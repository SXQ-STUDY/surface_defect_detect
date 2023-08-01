# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmseg.models.utils import DAPPM, BasicBlock, Bottleneck, resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType

from ..utils import make_divisible

class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.
    """
    def __init__(self,
                 channels,
                 ratio=16,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=make_divisible(channels // ratio, 8),
            kernel_size=1,
            stride=1,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=make_divisible(channels // ratio, 8),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class High_Low_semantic_fusion(nn.Module):
    def __init__(self, 
                 in_ch1=64, 
                 in_ch2=64, 
                 in_ch3=128, 
                 out_ch=128, 
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        self.se_layer = SELayer(in_ch1+in_ch2+in_ch3, ratio=16)
        self.conv_out = ConvModule(
            in_ch1+in_ch2+in_ch3, 
            out_ch, 
            kernel_size=1, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg
        )
        
    def prepare(self, x, y, z):
        x = self.prepare_x(x, y, z)
        y = self.prepare_y(x, y, z)
        z = self.prepare_z(x, y, z)
        return x, y, z

    def prepare_x(self, x, y, z):
        return x

    def prepare_y(self, x, y, z):
        return y
    
    def prepare_z(self, x, y, z):
        return z

    def fuse(self, x, y, z):
        cat_xyz = torch.cat([x, y, z], dim=1)
        se_xyz = self.se_layer(cat_xyz)
        return se_xyz

    def forward(self, x, y, z):
        x, y, z = self.prepare(x, y, z)
        out = self.fuse(x, y, z)
        out = self.conv_out(out)
        return out

class Fuse_edgeinfo(nn.Module):
    def __init__(self, 
                 edge_ch=64, 
                 se_ch=128, 
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        self.edge_ch = edge_ch
        self.se_ch = se_ch
        self.conv_edge = ConvModule(
            edge_ch, 
            se_ch, 
            kernel_size=1, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg
        )

    def fuse(self, edge_f, se_f):
        edge = self.conv_edge(edge_f)
        out = edge + se_f
        return out
    
    def forward(self, edge_f, se_f):
        out = self.fuse(edge_f, se_f)
        return out

@MODELS.register_module()
class MyDDRNet_with_edgebranch_with_highlowfuse(BaseModule):
    """DDRNet backbone.

    This backbone is the implementation of `Deep Dual-resolution Networks for
    Real-time and Accurate Semantic Segmentation of Road Scenes
    <http://arxiv.org/abs/2101.06085>`_.
    Modified from https://github.com/ydhongHIT/DDRNet.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        channels: (int): The base channels of DDRNet. Default: 32.
        ppm_channels (int): The channels of PPM module. Default: 128.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict to build norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict, optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        # bilateral fusion
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_1 = ConvModule(
            channels * 2,
            channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.down_2 = nn.Sequential(
            ConvModule(
                channels * 2,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None))

        # high resolution(spatial) branch
        self.spatial_branch_layers = nn.ModuleList()
        for i in range(3):
            self.spatial_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2,
                    planes=channels * 2,
                    num_blocks=2 if i < 2 else 1,
                ))

        self.spp = DAPPM(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        
        self.hl_fuse = High_Low_semantic_fusion(in_ch1=64, in_ch2=64, in_ch3=128, out_ch=128)
        self.edge_se_fuse = Fuse_edgeinfo(edge_ch=64, se_ch=128)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
            nn.ReLU(),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        out_size = (x.shape[-2] // 8, x.shape[-1] // 8)
        # stage 0-2 
        x = self.stem(x) # b*64*1/8h*1/8w
        # stage3 
        x_c = self.context_branch_layers[0](x) # b*128*1/16h*1/16w
        x_s = self.spatial_branch_layers[0](x) # b*64*1/8h*1/8w
        if self.training:
            c1 = x_c.clone()
            s1 = x_s.clone()
        comp_c = self.compression_1(self.relu(x_c)) # b*64*1/16h*1/16w
        x_c += self.down_1(self.relu(x_s)) # b*128*1/16h*1/16w
        x_s += resize( # b*64*1/8h*1/8w
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            c2 = x_c.clone()
            s2 = x_s.clone()
        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c)) # b*256*1/32h*1/32w
        x_s = self.spatial_branch_layers[1](self.relu(x_s)) # b*64*1/8h*1/8w
        comp_c = self.compression_2(self.relu(x_c)) # b*64*1/32h*1/32w
        x_c += self.down_2(self.relu(x_s)) # b*256*1/32h*1/32w
        x_s += resize( # b*64*1/8h*1/8w
            comp_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            c3 = x_c.clone()
            s3 = x_s.clone()
        # stage5
        x_s = self.spatial_branch_layers[2](self.relu(x_s)) # b*128*1/8h*1/8w
        x_c = self.context_branch_layers[2](self.relu(x_c)) # b*512*1/64h*1/64w
        x_c = self.spp(x_c) # b*128*1/64h*1/64w
        x_c = resize( # b*128*1/8h*1/8w
            x_c,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        
        final_s = x_s + x_c
        out = self.hl_fuse(s2, s3, final_s)
        out = self.edge_se_fuse(s1, out)

        return (c1, s1, c2, s2, c3, s3, out) if self.training else out
