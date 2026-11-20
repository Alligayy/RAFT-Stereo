import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

class FlowHead(nn.Module):
    # 将最细尺度的隐藏状态映射为每个网格像素的位移增量 (dx, dy)。
    # RAFTStereo.forward 会把 dy 清零，使更新沿水平方向进行。
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    # 用二维卷积计算逐位置的 GRU 门，在更新隐藏状态时利用同尺度的空间邻域。
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, cz, cr, cq, *x_list):
        # x_list 是本轮来自运动编码器或其他尺度的输入，沿通道维合并后与旧状态 h 拼接。
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        # cz、cr、cq 来自左图上下文，已在循环外计算。
        # z 控制采纳候选状态的比例，r 控制用多少旧状态生成候选状态 q。
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + cq)

        # 逐元素在旧状态与候选状态之间插值，作为本轮更新后的隐藏状态。
        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    # 备用 GRU，先用 1×5 卷积沿水平方向更新，再用 5×1 卷积沿竖直方向更新。
    # 当前 BasicMultiUpdateBlock 的三个尺度使用的是 ConvGRU。
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, *x):
        # 横向门控更新，扩展同一行内的感受野。
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # 纵向更新使用上一阶段更新过的隐藏状态。
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module):
    # 在最细网格上融合当前位移与局部匹配分数，生成供 GRU 使用的运动特征。
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        # 每层取左右各 radius 个位置和中心位置，各层结果沿通道维堆叠。
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+64, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        # 相关性与双通道位移走各自的卷积分支，再融合成 126 个通道。
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        # 直接保留原始双通道位移，合计输出 128 个通道。
        return torch.cat([out, flow], dim=1)

def pool2x(x):
    # 用带重叠的 3×3 平均池化，把较细层状态对齐到相邻的较粗尺度。
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    # 按目标状态的空间形状上采样粗尺度特征，便于按通道拼接。
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)

class BasicMultiUpdateBlock(nn.Module):
    # 交换相邻尺度的信息，并从最高分辨率的 GRU 读出位移增量和上采样掩码。
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        # hidden_dims 的索引从粗到细，forward 中 net 和 inp 的索引从细到粗。
        # 08/16/32 是历史命名，实际空间步幅依次为 2**n_downsample 的 1、2、4 倍。
        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2**self.args.n_downsample

        # 为每个低分辨率位置预测 factor**2 组权重，每组 9 个值对应 3×3 位移邻域。
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):

        # 按粗 → 中 → 细原地替换列表中的状态。较粗层汇聚尚未更新的相邻细层状态，
        # 较细层可直接使用本次调用已更新的相邻粗层状态。
        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            # 只有最细层读取相关性与当前位移，粗层通过相邻尺度的隐藏状态获取匹配信息。
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        # slow_fast_gru 的预更新只返回隐藏状态，完整更新才读取两个预测头。
        if not update:
            return net

        delta_flow = self.flow_head(net[0])

        # 先缩放掩码的未归一化权重，以平衡该分支的梯度，softmax 在上采样时执行。
        mask = .25 * self.mask(net[0])
        return net, mask, delta_flow
