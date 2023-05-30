from mmseg.models.backbones.swin import SwinTransformer
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import torch
import torch.nn.functional as F
from torch import nn

class Deeplabv3p_res50(nn.Module):
    def __init__(self, num_classes):
        super(Deeplabv3p_res50, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        backbone_norm_cfg = dict(type='LN', requires_grad=True)
        self.feature_extractor = EncoderDecoder(
            backbone=dict(
            type='ResNetV1c',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),

            decode_head = dict(
                type='DepthwiseSeparableASPPHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        )
        self.feature_extractor.init_weights()

        self.head =DepthwiseSeparableASPPHead(
                in_channels=2048,
                in_index=3,
                channels=512,
                dilations=(1, 12, 24, 36),
                c1_in_channels=256,
                c1_channels=48,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        print(feature.shape)
        # out = self.head(feature)
        # out = F.interpolate(out, x.shape[2:], mode='bilinear',align_corners=True)
        return feature


