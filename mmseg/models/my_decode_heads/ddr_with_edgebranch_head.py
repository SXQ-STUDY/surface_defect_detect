# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


@MODELS.register_module()
class DDRHead_with_edgebranch_with_borderloss(BaseDecodeHead):
    """Decode head for DDRNet.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        num_classes (int): Number of classes.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)
        # s1
        self.aux_head_s1 = self._make_base_head(self.in_channels // 2, self.channels)
        self.aux_cls_edge_s1 = nn.Conv2d(self.channels, 1, kernel_size=1)
        # c2
        self.aux_head_c2 = self._make_base_head(self.in_channels, self.channels)
        self.aux_cls_seg_c2 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)
        # c3
        self.aux_head_c3 = self._make_base_head(self.in_channels*2, self.channels)
        self.aux_cls_seg_c3 = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            c1, s1, c2, s2, c3, s3, fuse = inputs
            x_fuse = self.head(fuse)
            x_fuse = self.cls_seg(x_fuse)
            
            x_s1 = self.aux_head_s1(s1)
            x_s1 = self.aux_cls_edge_s1(x_s1)

            x_c2 = self.aux_head_c2(c2)
            x_c2 = self.aux_cls_seg_c2(x_c2)

            x_c3 = self.aux_head_c3(c3)
            x_c3 = self.aux_cls_seg_c3(x_c3)

            return x_fuse, x_s1, x_c2, x_c3
        else:
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]

        return nn.Sequential(*layers)
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs


    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        fuse_logit, s1_logit, c2_logit, c3_logit = seg_logits
        seg_label, edge_label = self._stack_batch_gt(batch_data_samples)

        fuse_logit = resize(
            fuse_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        s1_logit = resize(
            s1_logit,
            size=edge_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c2_logit = resize(
            c2_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c3_logit = resize(
            c3_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)
        edge_label = edge_label.squeeze(1)
        filler = torch.ones_like(seg_label) * self.ignore_index
        seg_edge_label = torch.where(edge_label==1, seg_label, filler)

        loss['loss_fuse'] = self.loss_decode[0](fuse_logit, seg_label)
        loss['loss_c2'] = self.loss_decode[1](c2_logit, seg_label)
        loss['loss_c3'] = self.loss_decode[2](c3_logit, seg_label)
        loss['loss_fuse_seg_edge'] = self.loss_decode[3](fuse_logit, seg_edge_label)

        loss['loss_s1'] = self.loss_decode[4](s1_logit, edge_label)

        loss['acc_seg'] = accuracy(
            fuse_logit, seg_label, ignore_index=self.ignore_index)

        return loss
