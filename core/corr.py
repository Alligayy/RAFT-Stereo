import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

# 立体匹配的搜索沿右图水平方向进行。各层采样结果按通道堆叠，
# 得到 [B, num_levels * (2 * radius + 1), H, W_left]，供最细尺度的更新块使用。
try:
    # reg_cuda 使用的可选扩展，由 sampler/setup.py 构建。
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    # CUDA 采样的自动求导封装。前向沿相关性体的右图宽度轴做一维线性插值。
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        # 反向按插值权重将梯度分配回相关性体，坐标与搜索半径不返回梯度。
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    # reg_cuda：与 reg 一样预先计算逐行全配对相关性，使用 CUDA 扩展查询局部邻域。
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # 将每个左图位置对应的右图相关性向量单独成行，只在 W_right 轴做平均池化。
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        # CUDA 核只用横坐标，低一层金字塔的横坐标单位扩大一倍。
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        # 与 CorrBlock1D.corr 相同，固定批次与行，仅沿特征通道求和得到点积。
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    # alt：保存匹配特征，在每轮按搜索窗口采样右图特征并计算点积，以减少显存占用。
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        # grid_sample 使用归一化坐标，align_corners=True 时首尾像素中心正好映射为 -1 和 1。
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        # 按搜索窗口的偏移逐个采样。每次取一张右图特征网格，与左图特征逐像素做通道点积。
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        out_pyramid = []
        # 左图特征保持原网格，右图特征只沿水平方向逐层池化，纵坐标始终使用原网格单位。
        for i in range(self.num_levels):
            # 坐标最后一维的顺序为 (x, y)。堆叠结果的第一项是横向偏移，第二项固定为 0。
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class CorrBlock1D:
    # reg：预计算逐行全配对相关性并建立金字塔，每轮只在金字塔上查询。
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 相关性体的形状是 [B, H, W_left, 1, W_right]，存储同一行的所有左右位置对的分数。
        # 存储量随两图宽度的乘积增长，高分辨率输入可用 alt 按需计算。
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        # 每个左图位置有一条相关性曲线。沿右图宽度逐级平均池化，扩大每层覆盖的搜索范围。
        # 当前构造循环额外存了一层，__call__ 只读取前 num_levels 层。
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        # 相关性曲线已由左图位置和所在行确定，查询时只取当前估计的右图横坐标。
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            # 将估计坐标除以 2**i 换算到第 i 层，再取 [-r, r] 的整数偏移。
            # 这一层相邻的两个查询点在原特征网格上相隔 2**i 格。
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            # 复用二维采样器。每条相关性曲线的高度为 1，固定 y=0，实际只沿 x 插值。
            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        # 每个左图位置依次拼接从细到粗各层的 2*r+1 个分数，再将分数轴移到通道维。
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        # einsum 在 D 个特征通道上求和，保留批次、行、左图列和右图列四个轴。
        # 点积除以 sqrt(D)，控制匹配分数随特征通道数增长的尺度。
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class AlternateCorrBlock:
    # alt_cuda 的预留接口。构造函数直接抛出异常，后续代码在当前版本中不会执行。
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
