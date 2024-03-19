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

from .deeplab_utils import ASPP, Decoder, CBAM, SpatialAttention, ChannelAttention, Decoder_Attention
from .aff_utils import PathIndex
from .puzzle_utils import tile_features, merge_features

from tools.ai.torch_utils import resize_for_tensors
#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)


class MTNet(Backbone):
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
        if self.train_mode:
            uni_dist = Uniform(-0.3, 0.3)
            noise_vector2 = uni_dist.sample(x2.shape[1:]).cuda().unsqueeze(0)
            noise_vector3 = uni_dist.sample(x3.shape[1:]).cuda().unsqueeze(0)
            x2 = x2.mul(noise_vector2) + x2
            x3 = x3.mul(noise_vector3) + x3

        x1 = self.decoder1(x, x_low_level)
        x2 = self.decoder2(x, x_low_level)
        x3 = self.decoder3(x, x_low_level)

        x1 = resize_for_tensors(x1, inputs.size()[2:], align_corners=True)
        x2 = resize_for_tensors(x2, inputs.size()[2:], align_corners=True)
        x3 = resize_for_tensors(x3, inputs.size()[2:], align_corners=True)
        if self.train_mode:
            return x1, x2, x3
        else:
            return x1


class MTNet_Plus(Backbone):
    def __init__(self, model_name, num_classes=2, mode='fix', use_group_norm=False, train=True):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn

        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder1 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='CBAM')
        self.decoder2 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='SA')
        self.decoder3 = Decoder_Attention(num_classes, 256, norm_fn_for_extra_modules, attention_mode='CA')
        
        self.classifier = nn.Conv2d(3584, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.CBAM = CBAM(256)
        self.SA = SpatialAttention()
        self.CA = ChannelAttention(256)

        self.CBAM2 = CBAM(256)
        self.SA2 = SpatialAttention()
        self.CA2 = ChannelAttention(256)

        self.train_mode = train

    def forward(self, x):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x

        x = self.stage3(x)
        x_cam3 = x
        x = self.stage4(x)
        x_cam4 = x
        x = self.stage5(x)
        x_cam5 = x
        x_high_level = x

        # x_cam: torch.Size([16, 3584, 32, 32])
        x_cam = torch.cat([x_cam3, F.interpolate(x_cam4, size=x_cam3.shape[2:], align_corners=True, mode='bilinear'),\
                           F.interpolate(x_cam5, size=x_cam3.shape[2:], align_corners=True, mode='bilinear')], dim=1)
        
        features = self.classifier(x_cam)
        logits = F.avg_pool2d(features, kernel_size=(features.size(2), features.size(3)), padding=0)
        logits = logits.view(-1, self.num_classes)

        x = self.aspp(x)

        x2 = x
        x3 = x

        if self.train_mode:
            uni_dist = Uniform(-0.3, 0.3)
            noise_vector2 = uni_dist.sample(x2.shape[1:]).cuda().unsqueeze(0)
            noise_vector3 = uni_dist.sample(x3.shape[1:]).cuda().unsqueeze(0)
            x2 = x2.mul(noise_vector2) + x2
            x3 = x3.mul(noise_vector3) + x3

        x1 = self.decoder1(x, x_low_level)
        x2 = self.decoder2(x, x_low_level)
        x3 = self.decoder3(x, x_low_level)

        x1 = resize_for_tensors(x1, inputs.size()[2:], align_corners=True)
        x2 = resize_for_tensors(x2, inputs.size()[2:], align_corners=True)
        x3 = resize_for_tensors(x3, inputs.size()[2:], align_corners=True)
        if self.train_mode:
            return x1, x2, x3, logits, self.cam_normalize(features, size=x1.shape[2:])
        else:
            logits_soft = torch.softmax(logits, dim=1)
            logits_soft_max = torch.maximum(logits_soft[:,0], logits_soft[:,1])
            # print(logits_soft_max)
            # print(logits_soft_max.shape, self.cam_normalize(features, size=x1.shape[2:]).shape)
            # return x1, logits_soft_max[:,None,None,None]*self.cam_normalize(features, size=x1.shape[2:])
            return x1, self.cam_normalize(features, size=x1.shape[2:])

    def cam_normalize(self, cam, size):
        # cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=True)
        # cam = cam / (F.adaptive_max_pool2d(cam, 1) + 1e-5)
        cam = torch.softmax(cam, dim=1)
        return cam