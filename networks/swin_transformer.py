from mmseg.models.backbones.swin import SwinTransformer
from mmseg.models.decode_heads.uper_head import UPerHead
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import torch
import torch.nn.functional as F
from torch import nn

class Swin_Transformer(nn.Module):
    def __init__(self, num_classes):
        super(Swin_Transformer, self).__init__()
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'
        init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file)
        norm_cfg = dict(type='BN', requires_grad=True)
        backbone_norm_cfg = dict(type='LN', requires_grad=True)
        self.feature_extractor = SwinTransformer(
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg,
                init_cfg=init_cfg
                )
        self.feature_extractor.init_weights()

        self.head = UPerHead(
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        out = self.head(feature)
        out = F.interpolate(out, x.shape[2:], mode='bilinear',align_corners=True)
        return out

class Swin_tiny(nn.Module):
    def __init__(self, num_classes):
        super(Swin_tiny, self).__init__()
        checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
        init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file)
        norm_cfg = dict(type='BN', requires_grad=True)
        backbone_norm_cfg = dict(type='LN', requires_grad=True)
        self.feature_extractor = SwinTransformer(
                pretrain_img_size=224,
                embed_dims=96,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=backbone_norm_cfg,
                init_cfg=init_cfg
                )
        self.feature_extractor.init_weights()

        self.head = UPerHead(
                in_channels=[96, 192, 384, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        out = self.head(feature)
        out = F.interpolate(out, x.shape[2:], mode='bilinear',align_corners=True)
        return out