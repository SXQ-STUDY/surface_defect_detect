# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


# 未实现
def py_sigmoid_focal_loss1(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=0.5, # 阈值
                          alpha=0.6, # 
                          beta=1, #  exp(1-x) - beta
                          mi = 2,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    # 大于gamma采用(1-x)^2 小于gamma的采用exp(1-x)-beta

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    gt_logits = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    smaller_gamma_flag = gt_logits <= gamma
    bigger_gamma_flag = gt_logits > gamma
    focal_weight = (torch.exp(one_minus_pt*smaller_gamma_flag) - beta)*smaller_gamma_flag + one_minus_pt.pow(mi)*bigger_gamma_flag
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * focal_weight

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') 
    loss = loss * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss2(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=0.5,
                          alpha=0.6,
                          beta=1,
                          mi=2,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    # exp(1-x) - 1
    
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # 1
    # gt_logits = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    # one_minus_pt = 1 - gt_logits
    # 2 
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

    one_minus_pt = torch.exp(one_minus_pt) - 1
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') 
    loss = loss * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss3(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=0.5, # 阈值
                          alpha=0.6, # 
                          beta=1, #  exp(1-x) - beta
                          mi = 2,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    # 大于gamma采用(1-x)^2 小于gamma的采用exp(1-x)-beta

    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    gt_logits = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    one_minus_pt = 1 - gt_logits
    smaller_gamma_flag = gt_logits <= gamma
    bigger_gamma_flag = gt_logits > gamma
    focal_weight = (torch.exp(one_minus_pt*smaller_gamma_flag) - beta)*smaller_gamma_flag + one_minus_pt.pow(mi)*bigger_gamma_flag
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * focal_weight

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') 
    loss = loss * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def py_sigmoid_focal_loss4(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=0.5, # 阈值
                          alpha=0.6, # 
                          beta=1, #  exp(1-x) - beta
                          mi = 0,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):

    # 大于gamma采用原本的cross entropy 小于gamma的采用exp(1-x) -beta
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    #one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    gt_logits = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target
    one_minus_pt = 1 - gt_logits
    smaller_gamma_flag = gt_logits <= gamma
    bigger_gamma_flag = gt_logits > gamma
    focal_weight = (torch.exp(one_minus_pt*smaller_gamma_flag) - beta)*smaller_gamma_flag + one_minus_pt.pow(mi)*bigger_gamma_flag
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * focal_weight

    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') 
    loss = loss * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class SFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=0.5,
                 alpha=0.6,
                 beta=1,
                 mi=2,
                 exp_name='3',
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_sfocal'):
        super().__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.mi = mi
        self.exp_name = exp_name
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):

        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            one_hot_target = None
            if target.dim() == 1:
                target = F.one_hot(target, num_classes=num_classes)
            else:
                valid_mask = (target.argmax(dim=1) != ignore_index).view(-1, 1)
            
            if self.exp_name == '1':
                calculate_loss_func = py_sigmoid_focal_loss1
            elif self.exp_name == '2':
                calculate_loss_func = py_sigmoid_focal_loss2
            elif self.exp_name == '3':
                calculate_loss_func = py_sigmoid_focal_loss3
            elif self.exp_name == '4':
                calculate_loss_func = py_sigmoid_focal_loss4

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                beta=self.beta,
                mi=self.mi,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name + '-' + self.exp_name
