# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.distributions.uniform import Uniform

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model

from .deeplab_utils import ASPP, Decoder, CBAM, Decoder_DCN, SpatialAttention, ChannelAttention, Decoder_Attention
from .aff_utils import PathIndex
from .puzzle_utils import tile_features, merge_features

from tools.ai.torch_utils import resize_for_tensors
#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

"""dice:69.55"""
class DeepLabv3_Plus_MultiHead3_Reception_Attention(Backbone):
    def __init__(self, model_name, num_classes=2, mode='fix', use_group_norm=False, rotation=False, train=True):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn

        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder1 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='CBAM')
        self.decoder2 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='SA')
        self.decoder3 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='CA')

        self.CBAM = CBAM(256)
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(256)

        self.CBAM2 = CBAM(256)
        self.SA2 = SpatialAttention()
        self.CA2 = ChannelAttention(256)

        self.rotation = rotation
        self.train_mode = train

    def forward(self, x):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x

        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x_high_level = x

        x = self.aspp(x)

        x2 = x
        x3 = x
        # rotation
        if self.train_mode:
            if self.rotation:
                x2 = torch.rot90(x2, k=1, dims=[2,3])
                x3 = torch.rot90(x2, k=2, dims=[2,3])

        # purturbarions, feature noise
        if self.train_mode:
            uni_dist = Uniform(-0.3, 0.3)
            noise_vector2 = uni_dist.sample(x2.shape[1:]).cuda().unsqueeze(0)
            noise_vector3 = uni_dist.sample(x3.shape[1:]).cuda().unsqueeze(0)
            x2 = x2.mul(noise_vector2) + x2
            x3 = x3.mul(noise_vector3) + x3

            # different purturbations, dropout in the feature maps
            # x2_low_level = F.dropout2d(x_low_level, p=0.3)
            # x3_low_level = F.dropout2d(x_low_level, p=0.3)
            # attention = torch.mean(x2, dim=1, keepdim=True)
            # max_val, _ = torch.max(attention.view(x2.size(0), -1), dim=1, keepdim=True)
            # threshold = max_val * np.random.uniform(0.7, 0.9)
            # threshold = threshold.view(x2.size(0), 1, 1, 1).expand_as(attention)
            # drop_mask = (attention < threshold).float()
            # x2 = x2.mul(drop_mask)

            # x2 = F.dropout2d(x2, p=0.3)
            # x3 = F.dropout2d(x3, p=0.3)

        # x_CBAM = self.CBAM(x)
        # x_SA= self.SA(x2)
        # x_CA = self.CA(x3)

        # x1_low_level = self.CBAM2(x_low_level)
        # x2_low_level = self.SA2(x_low_level)
        # x3_low_level = self.CA2(x_low_level)

        # x1 = self.decoder1(x, x1_low_level)
        # x2 = self.decoder2(x_SA, x2_low_level)
        # x3 = self.decoder3(x_CA, x3_low_level)

        x1 = self.decoder1(x, x_low_level)
        x2 = self.decoder2(x, x_low_level)
        x3 = self.decoder3(x, x_low_level)

        x1 = resize_for_tensors(x1, inputs.size()[2:], align_corners=True)
        x2 = resize_for_tensors(x2, inputs.size()[2:], align_corners=True)
        x3 = resize_for_tensors(x3, inputs.size()[2:], align_corners=True)

        if self.train_mode:
            if self.rotation:
                x2 = torch.rot90(x2, k=3, dims=[2,3])
                x3 = torch.rot90(x3, k=2, dims=[2,3])
        # return x1, x2, x3
        return x1, x2, x3

