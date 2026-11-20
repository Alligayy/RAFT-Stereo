import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8


try:
    autocast = torch.cuda.amp.autocast
except:
    # 兼容缺少 autocast 的旧版 PyTorch，保留上下文管理器接口。
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTStereo(nn.Module):
    # 网络由特征与上下文编码、一维相关性查询、多尺度循环更新和学习式上采样组成。
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        # cnet 为每个启用的 GRU 尺度分别输出初始隐藏状态和固定的上下文特征。
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        # 每个尺度的上下文预投影为 3 组通道，分别加到更新门 z、重置门 r 和候选状态 q 上。
        # 投影层按 hidden_dims 的索引构建（从粗到细），默认各尺度均为 128 通道。
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        # 共享主干时，从 cnet 的中间特征生成左右图的 256 通道匹配特征。
        # 独立主干时，fnet 对两幅图提取同样用途的特征，并在两幅图之间共用权重。
        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        # 训练入口调用此方法，让 BatchNorm 使用已有的运行统计量。
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """在特征图网格上构造两份 [N, 2, H, W] 坐标，通道顺序为 (x, y)。"""
        N, _, H, W = img.shape

        # coords0 固定为左图网格，coords1 表示右图匹配坐标。两者相等时初始位移为零。
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """对低分辨率位移作学习式凸组合，上采样倍数为 2 ** n_downsample。"""
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample

        # 每个粗网格像素对应 factor × factor 个输出像素，每个输出像素各有 9 个权重。
        # softmax 沿 3×3 邻域归一化，使每组权重非负且和为 1。
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        # unfold 取出每个位置的 3×3 邻域。先乘 factor，将位移从特征格单位换为输入图像像素。
        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        # 加权求和后交错排列粗网格索引与子像素索引，拼成 [N, D, factor*H, factor*W]。
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """输入已极线校正的左右 RGB 图像，迭代估计水平位移 x_right - x_left。"""

        # 输入形状为 [B, 3, H, W]，将 [0, 255] 像素映射到 [-1, 1]。
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                # 按批次拼接左右图。cnet 只为左图生成上下文，但返回两幅图的共享主干特征。
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

            # 列表按空间分辨率从高到低排列。每个尺度的第一个输出初始化隐藏状态，第二个输出供整段迭代复用。
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # 固定上下文的卷积只计算一次，得到每个尺度的 [cz, cr, cq]。
            # zip 将从细到粗的上下文与从粗到细创建的投影层按位置配对，通道数不一致会导致形状错误。
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        # reg 预计算逐行全配对相关性，alt 在当前搜索窗口内按需计算。两条 PyTorch 路径使用 float32 特征。
        if self.args.corr_implementation == "reg":
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":
            # 复用全配对相关性体，邻域查询由 sampler/ 中的 CUDA 扩展完成。
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":
            # 保留的入口，当前 AlternateCorrBlock 构造时会抛出 NotImplementedError。
            corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        # 位移场和相关性查询都在最细的 GRU 网格上进行，尺度为输入图像的 1 / 2**n_downsample。
        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            # 可用同一网格、同一单位的 [dx, 0] 位移作为初值。
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            # 切断上一轮坐标的梯度路径，GRU 隐藏状态仍在各轮之间传递梯度。
            coords1 = coords1.detach()
            # 以估计的右图坐标为中心，查询多尺度匹配分数，并与当前位移一起送入更新块。
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # slow_fast_gru 先额外更新粗尺度状态，再执行一次完整更新并预测位移增量。
                # 三层时每轮粗、中、细 GRU 分别更新 3、2、1 次，两层时分别为 2、1 次。
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # 极线校正后对应点应在同一行。将 (dx, dy) 的纵向增量清零，只累加水平增量。
            delta_flow[:,1] = 0.0
            coords1 = coords1 + delta_flow

            # 测试模式仅上采样最后一轮，训练模式保留每轮结果供序列损失使用。
            if test_mode and itr < iters-1:
                continue

            if up_mask is None:
                # 无掩码时回退到固定 8 倍双线性插值。当前完整更新通常返回学习式掩码。
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            # 只保留水平分量。推理入口先填充输入再裁去输出的填充区域，恢复原图尺寸。
            # 数据集将常用视差 d = x_left - x_right 取负，作为这里的水平位移监督。
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            # 依次返回低分辨率双通道位移和上采样后的单通道水平位移。
            return coords1 - coords0, flow_up

        return flow_predictions
