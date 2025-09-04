## Conv4d.py
import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = (1, 1, 1, 1),
                 padding:[int, tuple] = (0, 0, 0, 0),
                 dilation:[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out



if __name__ == "__main__":
    input = torch.randn(2, 1, 5, 5, 5, 5).cuda()

    net = Conv4d(1, 1, kernel_size=(3, 1,1, 1), padding=(0, 0, 0, 0), stride=(1, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True ).cuda()
    out1 = net(input)




## crop.py
import torch

def center_crop_2d(tensor: torch.Tensor, target_size):
    """
    对2D张量进行中心裁剪。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (B, C, Lat, Lon)
        target_size (tuple[int]): 目标尺寸 (Lat, Lon)

    返回:
        裁剪后的张量。
    """
    _, _, current_lat, current_lon = tensor.shape
    lat_diff = current_lat - target_size[0]
    lon_diff = current_lon - target_size[1]

    crop_top = lat_diff // 2
    crop_bottom = lat_diff - crop_top

    crop_left = lon_diff // 2
    crop_right = lon_diff - crop_left

    return tensor[:, :, crop_top: current_lat - crop_bottom, crop_left: current_lon - crop_right]


def center_crop_3d(tensor: torch.Tensor, target_size):
    """
    对3D张量进行中心裁剪。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (B, C, Pl, Lat, Lon)
        target_size (tuple[int]): 目标尺寸 (Pl, Lat, Lon)

    返回:
        裁剪后的张量。
    """
    _, _, current_pl, current_lat, current_lon = tensor.shape
    pl_diff = current_pl - target_size[0]
    lat_diff = current_lat - target_size[1]
    lon_diff = current_lon - target_size[2]

    crop_front = pl_diff // 2
    crop_back = pl_diff - crop_front

    crop_top = lat_diff // 2
    crop_bottom = lat_diff - crop_top

    crop_left = lon_diff // 2
    crop_right = lon_diff - crop_left

    return tensor[:, :, crop_front: current_pl - crop_back, crop_top: current_lat - crop_bottom,
                  crop_left: current_lon - crop_right]


## enrth_position.py
import torch

def calculate_position_bias_indices(size):
    """
    参数:
        size (tuple[int]): [压力层数, 纬度, 经度]

    返回:
        bias_indices (torch.Tensor): [pl_dim * lat_dim * lon_dim, pl_dim * lat_dim * lon_dim]
    """
    pl_dim, lat_dim, lon_dim = size

    # 获取查询矩阵中压力层的索引
    pl_query_indices = torch.arange(pl_dim)
    # 获取键矩阵中压力层的索引
    pl_key_indices = -torch.arange(pl_dim) * pl_dim

    # 获取查询矩阵中纬度的索引
    lat_query_indices = torch.arange(lat_dim)
    # 获取键矩阵中纬度的索引
    lat_key_indices = -torch.arange(lat_dim) * lat_dim

    # 获取键值对中的经度索引
    lon_indices = torch.arange(lon_dim)

    # 计算各个维度上的索引组合
    grid_query = torch.stack(torch.meshgrid([pl_query_indices, lat_query_indices, lon_indices], indexing='ij'))
    grid_key = torch.stack(torch.meshgrid([pl_key_indices, lat_key_indices, lon_indices], indexing='ij'))
    flat_query = torch.flatten(grid_query, 1)
    flat_key = torch.flatten(grid_key, 1)

    # 计算每个维度上的索引差并重新排列
    index_difference = flat_query[:, :, None] - flat_key[:, None, :]
    index_difference = index_difference.permute(1, 2, 0).contiguous()

    # 调整索引以使其从0开始
    index_difference[:, :, 2] += lon_dim - 1
    index_difference[:, :, 1] *= 2 * lon_dim - 1
    index_difference[:, :, 0] *= (2 * lon_dim - 1) * lat_dim * lat_dim

    # 在三个维度上累加索引值
    bias_indices = index_difference.sum(-1)

    return bias_indices

## embed.py
import torch
from torch import nn

import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = (1, 1, 1, 1),
                 padding:[int, tuple] = (0, 0, 0, 0),
                 dilation:[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out


class ImageToPatch2D(nn.Module):
    """
    将2D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        height, width = img_dims
        patch_h, patch_w = patch_dims

        padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算高度和宽度的余数
        height_mod = height % patch_h
        width_mod = width % patch_w

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_dims[0] and W == self.img_dims[1], \
            f"输入图像尺寸 ({H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization is not None:
            x = self.normalization(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class ImageToPatch3D(nn.Module):
    """
    将3D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        depth, height, width = img_dims
        patch_d, patch_h, patch_w = patch_dims

        padding_front = padding_back = padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算深度、高度和宽度的余数
        depth_mod = depth % patch_d
        height_mod = height % patch_h
        width_mod = width % patch_w

        if depth_mod:
            pad_depth = patch_d - depth_mod
            padding_front = pad_depth // 2
            padding_back = pad_depth - padding_front

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad3d(
            (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        )
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape #
        assert C == self.img_dims[0] and H == self.img_dims[1] and W == self.img_dims[2], \
            f"输入图像尺寸 ({D}x{H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}x{self.img_dims[2]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization:
            x = self.normalization(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


class ImageToPatch4D(nn.Module):
    """
    将4D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸（时间、深度、高度、宽度）。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        feature, depth, time, height, width = img_dims
        patch_d, patch_t, patch_h, patch_w = patch_dims

        # 初始化填充变量
        padding_time_front = padding_time_back = padding_depth_front = padding_depth_back = 0
        padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算每个维度的余数并设置填充
        time_mod = time % patch_t # 2 % 2
        depth_mod = depth % patch_d 
        height_mod = height % patch_h
        width_mod = width % patch_w

        if time_mod:
            pad_time = patch_t - time_mod
            padding_time_front = pad_time // 2
            padding_time_back = pad_time - padding_time_front

        if depth_mod:
            pad_depth = patch_d - depth_mod
            padding_depth_front = pad_depth // 2
            padding_depth_back = pad_depth - padding_depth_front

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 填充层
        self.padding = nn.ConstantPad3d(
            (padding_left, padding_right, padding_top, padding_bottom,
             padding_depth_front, padding_depth_back, padding_time_front, padding_time_back),
            0
        )
        
        ## 下面这段新增的
        self.padding_params = (
            padding_left, padding_right,  # W维度
            padding_top, padding_bottom,  # H维度
            padding_time_front, padding_time_back,  # T维度
            padding_depth_front, padding_depth_back  # D维度
        )

        # Conv4d 投影层
        self.projection = Conv4d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, D, T, H, W = x.shape
        assert C == self.img_dims[0] and D == self.img_dims[1] and T == self.img_dims[2] and H == self.img_dims[3] and W == self.img_dims[4], \
            f"输入图像尺寸 ({C}x{D}x{T}x{H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}x{self.img_dims[2]}x{self.img_dims[3]}x{self.img_dims[4]}) 不符。"
        
        # x = self.padding(x) 替换下一块
        x = F.pad(x, self.padding_params, mode='constant', value=0)
        
        
        
        x = self.projection(x)
        if self.normalization:
            x = self.normalization(x.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)
        return x
    
## helper.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        
        if self.in_channels != self.out_channels:
            x1 = self.channel_up(x)

            return self.channel_up(x)# + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=1,padding=(0, 1, 1))

    def forward(self, x):

        x = F.interpolate(x, scale_factor=(1, 2, 2))

        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)

        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


## pad.py
import torch

def calculate_padding_3d(resolution, window_dims):
    """
    计算3D张量所需的填充尺寸。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 (Pl, Lat, Lon)
        window_dims (tuple[int]): 窗口的尺寸 (Pl, Lat, Lon)

    返回:
        padding (tuple[int]): 需要的填充尺寸 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    Pl, Lat, Lon = resolution
    win_pl, win_lat, win_lon = window_dims

    pad_left = pad_right = pad_top = pad_bottom = pad_front = pad_back = 0

    # 计算深度、纬度和经度的余数
    pl_mod = Pl % win_pl
    lat_mod = Lat % win_lat
    lon_mod = Lon % win_lon

    # 计算深度维度的填充
    if pl_mod:
        pl_pad_total = win_pl - pl_mod
        pad_front = pl_pad_total // 2
        pad_back = pl_pad_total - pad_front

    # 计算纬度维度的填充
    if lat_mod:
        lat_pad_total = win_lat - lat_mod
        pad_top = lat_pad_total // 2
        pad_bottom = lat_pad_total - pad_top

    # 计算经度维度的填充
    if lon_mod:
        lon_pad_total = win_lon - lon_mod
        pad_left = lon_pad_total // 2
        pad_right = lon_pad_total - pad_left

    return pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back


def calculate_padding_2d(resolution, window_dims):
    """
    计算2D张量所需的填充尺寸。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 (Lat, Lon)
        window_dims (tuple[int]): 窗口的尺寸 (Lat, Lon)

    返回:
        padding (tuple[int]): 需要的填充尺寸 (pad_left, pad_right, pad_top, pad_bottom)
    """
    # 将2D问题转换为3D以重用计算逻辑
    resolution_3d = [1] + list(resolution)
    window_dims_3d = [1] + list(window_dims)
    padding = calculate_padding_3d(resolution_3d, window_dims_3d)
    return padding[2:6]  # 只取2D相关的填充部分


## recovery.py
import torch
from torch import nn

import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class ConvTranspose4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:[int, tuple],
                 stride:[int, tuple] = 1,
                 padding:[int, tuple] = 0,
                 output_padding:[int, tuple] = 0,
                 groups:int = 1,
                 bias=False,
                 #padding_mode:str ='zeros',
                 dilation:[int, tuple] = 1):
        super(ConvTranspose4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        output_padding = _quadruple(output_padding) 

        if not all(op < st or op < dl for op, st, dl in zip(output_padding, stride, dilation)):
          raise ValueError('output padding must be smaller than either stride or dilation, got output padding={}, dilation={}, stride={}'.format(output_padding, dilation, stride))

        input_padding = tuple(d*(ks-1)-p for d, p, ks in zip(dilation, padding, kernel_size))

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_padding = input_padding
        self.output_padding = output_padding
        self.dilation = dilation

        self.groups = groups

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.ConvTranspose3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     output_padding=self.output_padding[1::],
                                     dilation=self.dilation[1::],
                                     bias=False,
                                     stride=self.stride[1::])
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_ip, d_ip, h_ip, w_ip) = self.input_padding
        (l_op, d_op, h_op, w_op) = self.output_padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i - 1) * l_s - 2 * l_p + l_d * (l_k - 1) + l_op + 1
        d_o = (d_i - 1) * d_s - 2 * d_p + d_d * (d_k - 1) + d_op + 1
        h_o = (h_i - 1) * h_s - 2 * h_p + h_d * (h_k - 1) + h_op + 1
        w_o = (w_i - 1) * w_s - 2 * w_p + w_d * (w_k - 1) + w_op + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - (l_p) + i
            # Calculate the range of input frame j corresponding to kernel frame i
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(0, l_i):
                # Calculate the output frame
                out_frame = l_s * j + zero_offset
                if out_frame < 0 or out_frame >= out.shape[2]:
                  #print("{} -> {} (no)".format((i,l_s * j), out_frame))
                  continue
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out

class RecoveryImage2D(nn.Module):
    """
    将Patch Embedding恢复为2D图像。

    参数:
        image_size (tuple[int]): 图像的纬度和经度 (Lat, Lon)
        patch_size (tuple[int]): Patch的尺寸 (Lat, Lon)
        input_channels (int): 输入的通道数。
        output_channels (int): 输出的通道数。
    """

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = image_size
        self.transposed_conv = nn.ConvTranspose2d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.transposed_conv(x)
        _, _, height, width = x.shape
        height_padding = height - self.image_size[0]
        width_padding = width - self.image_size[1]

        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top

        pad_left = width_padding // 2
        pad_right = width_padding - pad_left

        return x[:, :, pad_top:height - pad_bottom, pad_left:width - pad_right]


class RecoveryImage3D(nn.Module):
    """
    将Patch Embedding恢复为3D图像。

    参数:
        image_size (tuple[int]): 图像的深度、纬度和经度 (Pl, Lat, Lon)
        patch_size (tuple[int]): Patch的尺寸 (Pl, Lat, Lon)
        input_channels (int): 输入的通道数。
        output_channels (int): 输出的通道数。
    """

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = image_size
        self.transposed_conv = nn.ConvTranspose3d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.transposed_conv(x)
        _, _, depth, height, width = x.shape

        depth_padding = depth - self.image_size[0]
        height_padding = height - self.image_size[1]
        width_padding = width - self.image_size[2]

        pad_front = depth_padding // 2
        pad_back = depth_padding - pad_front

        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top

        pad_left = width_padding // 2
        pad_right = width_padding - pad_left

        return x[:, :, pad_front:depth - pad_back, pad_top:height - pad_bottom, pad_left:width - pad_right]
    
class RecoveryImage4D(nn.Module):
    """
    将Patch Embedding恢复为4D图像。

    参数:
        image_size (tuple[int]): 图像的层、深度、纬度和经度 (Pl, Depth, Lat, Lon)
        patch_size (tuple[int]): Patch的尺寸 (Pl, Depth, Lat, Lon)
        input_channels (int): 输入的通道数。
        output_channels (int): 输出的通道数。
    """

    def __init__(self, image_size, patch_size, input_channels, output_channels, target_size=None):
        super(RecoveryImage4D, self).__init__()
        self.image_size = image_size
        self.target_size = target_size if target_size else image_size
        self.conv_transpose4d = ConvTranspose4d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        # 使用4D转置卷积操作恢复图像
        x = self.conv_transpose4d(x)

        # 获取重构后的图像尺寸
        _, feature, depth, time, height, width = x.shape
        print(x.shape, 'xxx')

        # 计算各个维度需要裁剪的大小
        depth_padding = depth - self.image_size[1]
        time_padding = time - self.image_size[2]
        height_padding = height - self.image_size[3]
        width_padding = width - self.image_size[4]

        pad_front = depth_padding // 2
        pad_back = depth_padding - pad_front

        pad_depth_front = depth_padding // 2
        pad_depth_back = depth_padding - pad_depth_front

        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top

        pad_left = width_padding // 2
        pad_right = width_padding - pad_left
        
        if self.target_size != None:
            _, target_d, target_t, target_h, target_w = self.target_size

        # 返回裁剪后的图像，确保尺寸匹配
        return x[:, :, :target_d, :, pad_top:height - pad_bottom, pad_left:width - pad_right]

    
## shift_winodw.py
import torch

def partition_windows(tensor: torch.Tensor, window_dims):
    """
    将输入张量分割成多个窗口。

    参数:
        tensor: 输入张量，形状为 (B, Pl, Lat, Lon, C)
        window_dims (tuple[int]): 窗口尺寸 [win_pl, win_lat, win_lon]

    返回:
        分割后的窗口: 形状为 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
    """
    B, Pl, Lat, Lon, C = tensor.shape
    win_pl, win_lat, win_lon = window_dims
    tensor = tensor.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = tensor.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(
        -1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C
    )
    return windows

def reverse_partition(windows, window_dims, Pl, Lat, Lon):
    """
    将分割后的窗口重新组合成原始张量。

    参数:
        windows: 输入张量，形状为 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        window_dims (tuple[int]): 窗口尺寸 [win_pl, win_lat, win_lon]
        Pl: 压力层的大小
        Lat: 纬度的大小
        Lon: 经度的大小

    返回:
        组合后的张量: 形状为 (B, Pl, Lat, Lon, C)
    """
    win_pl, win_lat, win_lon = window_dims
    B = int(windows.shape[0] / (Lon / win_lon))
    tensor = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    tensor = tensor.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return tensor

def create_shifted_window_mask(resolution, window_dims, shift_dims):
    """
    在经度维度上，最左边和最右边的索引实际上是相邻的。
    如果半窗口出现在最左边和最右边的两个位置，它们会被直接合并成一个窗口。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 [压力层, 纬度, 经度]
        window_dims (tuple[int]): 窗口尺寸 [压力层, 纬度, 经度]
        shift_dims (tuple[int]): SW-MSA的移位大小 [压力层, 纬度, 经度]

    返回:
        注意力掩码: 形状为 (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon)
    """
    Pl, Lat, Lon = resolution
    win_pl, win_lat, win_lon = window_dims
    shift_pl, shift_lat, shift_lon = shift_dims

    mask_tensor = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))

    pl_segments = (slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None))
    lat_segments = (slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None))
    lon_segments = (slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None))

    counter = 0
    for pl in pl_segments:
        for lat in lat_segments:
            for lon in lon_segments:
                mask_tensor[:, pl, lat, lon, :] = counter
                counter += 1

    mask_tensor = mask_tensor[:, :, :, :Lon, :]

    masked_windows = partition_windows(mask_tensor, window_dims)  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1
    masked_windows = masked_windows.view(masked_windows.shape[0], masked_windows.shape[1], win_pl * win_lat * win_lon)
    attention_mask = masked_windows.unsqueeze(2) - masked_windows.unsqueeze(3)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0, float(0.0))
    return attention_mask

## main model
import torch
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import sys
sys.path.append('..')
from canglong.earth_position import calculate_position_bias_indices
from canglong.shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from canglong.embed import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D
from canglong.recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
from canglong.pad import calculate_padding_3d, calculate_padding_2d
from canglong.crop import center_crop_2d, center_crop_3d
input_constant = torch.load('../constant_masks/Earth.pt').cuda()

class UpSample(nn.Module):
    """
    Up-sampling operation.
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: 2 * in_lat - pad_bottom, pad_left: 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                        self.type_of_windows, num_heads)
        )

        earth_position_index = calculate_position_bias_indices(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = calculate_padding_3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = create_shifted_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = partition_windows(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = partition_windows(shifted_x, self.window_size)

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

        if self.roll:
            shifted_x = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        x = center_crop_3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
import torch
import torch.nn as nn
from canglong.helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

class Encoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(Encoder, self).__init__()
        channels = [64, 64, 64, 128, 128]
        attn_resolutions = [2]
        num_res_blocks = 1
        resolution = 256

        # 初始卷积层
        self.conv_in = nn.Conv3d(image_channels, channels[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        
        # 第一层（含残差块和注意力模块）
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks, resolution, attn_resolutions)
        
        # 下采样与第二层
        self.downsample1 = DownSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks, resolution // 2, attn_resolutions)

        # Further downsampling and third layer
        self.downsample2 = DownSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks, resolution // 4, attn_resolutions)

        # 中间层的残差块和注意力模块
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
        
        # 输出层的归一化、激活和最终卷积层
        self.norm_out = GroupNorm(channels[3])
        self.act_out = Swish()
        self.conv_out = nn.Conv3d(channels[3], latent_dim, kernel_size=3, stride=1, padding=(1,2,1))

    def _make_layer(self, in_channels, out_channels, num_res_blocks, resolution, attn_resolutions):
        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
            if resolution in attn_resolutions:
                layers.append(NonLocalBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv_in(x)

        # 第一层，并存储跳跃连接
        x = self.layer1(x)
        skip = x  # 保存第一层输出，用于后续跳跃连接

        # 下采样，进入第二层
        x = self.downsample1(x)
        x = self.layer2(x)

        # Further downsample and third layer
        x = self.downsample2(x)
        x = self.layer3(x)

        # 中间层的残差块和注意力模块
        x = self.mid_block1(x)
        #x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        # 最终的归一化、激活和卷积输出层
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)[:, :, :, :181, :360]
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, image_channels=14, latent_dim=64):
        super(Decoder, self).__init__()
        channels = [128, 128, 64, 64]  # Decoder 的通道配置
        num_res_blocks = 1  # 与 Encoder 对齐

        # 初始卷积层
        self.conv_in = nn.Conv3d(latent_dim, channels[0], kernel_size=3, stride=1, padding=1)
        
        # 第一层残差块
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks)
        
        # 上采样和第二层残差块
        self.upsample1 = UpSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks)

        self.upsample2 = UpSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks)
        
        # 中间层的残差块
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
        
        # 最终输出层
        self.norm_out = GroupNorm(channels[3])
        self.act_out = Swish()
        self.conv_out = nn.ConvTranspose3d(channels[3], image_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def _make_layer(self, in_channels, out_channels, num_res_blocks):
        # 创建指定数量的残差块
        layers = [ResidualBlock(in_channels, out_channels) for _ in range(num_res_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv_in(x)

        # 第一层残差块
        x = self.layer1(x)

        # 上采样和第二层残差块
        x = self.upsample1(x)  # 上采样后通道数保持不变

        x = self.layer2(x)     # 确保输入与 layer2 的期望通道数匹配

        x = self.upsample2(x)  # 上采样后通道数保持不变

        x = self.layer3(x)     # 确保输入与 layer2 的期望通道数匹配
        
        # 中间层的残差块
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        # 最终的归一化、激活和卷积输出层
        x = self.norm_out(x)
        x = self.act_out(x)

        x = self.conv_out(x)[:, :, :, :721, :1440]
        
        return x


class Canglong(nn.Module):
    """
    CAS Canglong PyTorch impl of: `CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction`
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.patchembed2d = ImageToPatch2D(
            img_dims=(721, 1440),
            patch_dims=(4, 4), # 8, 8
            in_channels=4,
            out_channels=embed_dim,
        )
        self.patchembed3d = ImageToPatch3D(
            img_dims=(14, 721, 1440),
            patch_dims=(1, 4, 4),
            in_channels=14,
            out_channels=embed_dim
        )
        self.patchembed4d = ImageToPatch4D(
            img_dims=(7, 5, 2, 721, 1440),
            patch_dims=(2, 2, 4, 4),
            in_channels=7,
            out_channels=embed_dim
        )
        self.encoder3d = Encoder(image_channels=17, latent_dim=96)

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=(6, 181, 360), output_resolution=(6, 91, 180))
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim, (6, 91, 180), (6, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.patchrecovery2d = RecoveryImage2D((721, 1440), (4, 4), 2 * embed_dim, 4) #8, 8
        self.decoder3d = Decoder(image_channels=17, latent_dim=2 * 96)
        self.patchrecovery3d = RecoveryImage3D(image_size=(16, 721, 1440), 
                                               patch_size=(1, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=16) #2, 8, 8
        self.patchrecovery4d = RecoveryImage4D(image_size=(7, 5, 1, 721, 1440), 
                                               patch_size=(2, 1, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=7,
                                               target_size=(7, 5, 1, 721, 1440))
        

        self.conv_constant = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=4, padding=2)
        self.input_constant = input_constant


    def forward(self, surface, upper_air):
        
        
        constant = self.conv_constant(self.input_constant)
        surface = self.encoder3d(surface)

        upper_air = self.patchembed4d(upper_air)
        print(upper_air.shape, 'upper_air_before')
        print(surface.shape, 'surface_before')
        print(constant.shape, 'constant')
        
        x = torch.concat([upper_air.squeeze(3), 
                          surface, 
                          constant.unsqueeze(2)], dim=2)
        print(x.shape, 'before earthlayer')
        
        B, C, Pl, Lat, Lon = x.shape

        x = x.reshape(B, C, -1).transpose(1, 2)
        
        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        print(output.shape, 'after earthlayer')
        output_surface = output[:, :, 3:5, :, :]  #  四五层是surface
        output_upper_air = output[:, :, 0:3, :, :]  # 前三层是upper air


        output_surface = self.decoder3d(output_surface)
        print(output_upper_air.unsqueeze(3).shape)
        output_upper_air = self.patchrecovery4d(output_upper_air.unsqueeze(3))
        
        return output_surface, output_upper_air


        # 简化输出处理来验证模型架构
        return output_surface, output_upper_air  # 只取前2层surface

    
model = Canglong().cuda()
input_upper_air = torch.randn(1, 7, 5, 2, 721, 1440).cuda()
input_surface = torch.randn(1, 17, 2, 721, 1440).cuda()
output_surface, output_upper_air = model(input_surface, input_upper_air)
print(output_surface.shape)
print(output_upper_air.shape)