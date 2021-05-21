""" Mission """

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from mmcv.cnn import build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import constant_init
from ..builder import NECKS


# swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# separable convolution
class SeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=False):
        super(SeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = relu

        self.sep = nn.Conv2d(in_channels,
                             in_channels,
                             3,
                             padding=1,
                             groups=in_channels,
                             bias=False)
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=bias)
        if relu:
            self.relu_fn = Swish()

    def forward(self, x):
        x = self.pw(self.sep(x))
        if self.relu:
            x = self.relu_fn(x)
        return x


class WeightedInputConv(nn.Module):
    # TODO weighted Convolution
    # Fast normalized fusion
    """
    inputs =  [features1, features2, features3]
    out = conv((w1*feature1 + w2*feature2 + w3*feature3) / (w1 + w2 + w3 + eps))
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_ins,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 eps=0.0001):
        super(WeightedInputConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_ins = num_ins
        self.eps = eps
        _, bn_layer = build_norm_layer(norm_cfg, out_channels)
        """
        1. convolution
        2. weight
        3. swish
        """
        # use separable conv
        self.conv_op = nn.Sequential(
            SeparableConv(
                in_channels,
                out_channels,
                bias=True,
                relu=False),
            bn_layer
        )

        # edge weight and swish
        self.weight = nn.Parameter(torch.Tensor(self.num_ins).fill_(1.0))
        self._swish = Swish()

    def forward(self, inputs):
        """
        1. relu (weight)
        2. / (w.sum + eps)
        3. w * feature
        4. swish
        5. convolution
        """
        w = F.relu(self.weight)
        w /= (w.sum() + self.eps)
        x = 0
        for i in range(self.num_ins):
            x += w[i] * inputs[i]

        output = self.conv_op(self._swish(x))
        return output


class ResampingConv(nn.Module):
    def __init__(self,
                 in_channels,
                 in_stride,
                 out_stride,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None):
        super(ResampingConv, self).__init__()
        self.in_channels = in_channels
        self.in_stride = in_stride
        self.out_stride = out_stride
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        if self.in_stride < self.out_stride:
            scale = int(self.out_stride // self.in_stride)
            assert scale == 2
            self.rescale_op = nn.MaxPool2d(
                scale + 1,
                stride=scale,
                padding=1)
        else:
            if self.in_stride > self.out_stride:
                scale = self.in_stride // self.out_stride
                self.rescale_op = functools.partial(
                    F.interpolate, scale_factor=scale, mode='nearest')
            else:
                self.rescale_op = None

        if self.in_channels != self.out_channels:
            self.conv_op = ConvModule(
                in_channels,
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                inplace=False)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.conv_op(x)
        x = self.rescale_op(x) if self.rescale_op else x
        return x


class bifpn(nn.Module):
    # feature path
    nodes_settings = [
        {'width_ratio': 64, 'inputs_offsets': [3, 4]},
        {'width_ratio': 32, 'inputs_offsets': [2, 5]},
        {'width_ratio': 16, 'inputs_offsets': [1, 6]},
        {'width_ratio': 8, 'inputs_offsets': [0, 7]},
        {'width_ratio': 16, 'inputs_offsets': [1, 7, 8]},
        {'width_ratio': 32, 'inputs_offsets': [2, 6, 9]},
        {'width_ratio': 64, 'inputs_offsets': [3, 5, 10]},
        {'width_ratio': 128, 'inputs_offsets': [4, 11]},
    ]

    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=[8, 16, 32, 64, 128],
                 num_outs=5,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(bifpn, self).__init__()
        assert num_outs >= 2
        assert len(strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_outs = num_outs

        self.channels_nodes = [i for i in in_channels]
        self.stride_nodes = [i for i in strides]
        self.resample_op_nodes = nn.ModuleList()
        self.new_op_nodes = nn.ModuleList()
        for _, fnode in enumerate(self.nodes_settings):
            new_node_stride = fnode['width_ratio']
            op_node = nn.ModuleList()
            for _, input_offset in enumerate(fnode['inputs_offsets']):
                input_node = ResampingConv(
                    self.channels_nodes[input_offset],
                    self.stride_nodes[input_offset],
                    new_node_stride,
                    out_channels,
                    norm_cfg=norm_cfg)
                op_node.append(input_node)
            new_op_node = WeightedInputConv(
                out_channels,
                out_channels,
                len(fnode['inputs_offsets']),
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.new_op_nodes.append(new_op_node)
            self.resample_op_nodes.append(op_node)
            self.channels_nodes.append(out_channels)
            self.stride_nodes.append(new_node_stride)

    def forward(self, inputs):
        assert len(inputs) == self.num_outs
        feats = [i for i in inputs]
        for fnode, op_node, new_op_node in zip(self.nodes_settings,
                                               self.resample_op_nodes, self.new_op_nodes):
            input_node = []
            for input_offset, resample_op in zip(fnode['inputs_offsets'], op_node):
                # reshape input before weighted conv
                input_node.append(resample_op(feats[input_offset]))

            # weighted convolution
            feats.append(new_op_node(input_node))

        outputs = feats[-self.num_outs:]
        return outputs


@NECKS.register_module
class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 strides=[8, 16, 32, 64, 128],
                 start_level=0,
                 end_level=-1,
                 stack=3,
                 norm_cfg=dict(type='SyncBN', momentum=0.01, eps=1e-3, requires_grad=True),
                 act_cfg=None):
        super(BiFPN, self).__init__()
        assert len(in_channels) >= 3
        assert len(strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.num_ins = len(in_channels)
        self.act_cfg = act_cfg
        self.stack = stack
        self.num_outs = num_outs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        # add extra conv layers (e.g., RetinaNet)
        bifpn_in_channels = in_channels[self.start_level:self.backbone_end_level]
        bifpn_strides = strides[self.start_level:self.backbone_end_level]
        bifpn_num_outs = self.num_outs
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_convs = None
        if extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for _ in range(extra_levels):
                self.extra_convs.append(
                    ResampingConv(
                        bifpn_in_channels[-1],
                        bifpn_strides[-1],
                        bifpn_strides[-1] * 2,
                        out_channels,
                        norm_cfg=norm_cfg))
                bifpn_in_channels.append(out_channels)
                bifpn_strides.append(bifpn_strides[-1] * 2)

        self.stack_bifpns = nn.ModuleList()
        for _ in range(stack):
            self.stack_bifpns.append(
                bifpn(
                    bifpn_in_channels,
                    out_channels,
                    strides=bifpn_strides,
                    num_outs=bifpn_num_outs,
                    conv_cfg=None,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

            bifpn_in_channels = [out_channels for _ in range(bifpn_num_outs)]

    @auto_fp16()
    def forward(self, inputs):
        
        print(len(inputs))
        print(len(self.in_channels))
        assert len(inputs) == len(self.in_channels)
        feats = list(inputs[self.start_level:self.backbone_end_level])
        # add extra feature (ex. input features=4, output features=5, add 1 extra features from last feature)
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                feats.append(self.extra_convs[i](feats[-1]))
        # weighted bi-directional feature pyramid network
        for idx, stack_bifpn in enumerate(self.stack_bifpns):
            feats = stack_bifpn(feats)
        return tuple(feats[:self.num_outs])

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, 1)
