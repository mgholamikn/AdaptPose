# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
this folder and code is modified base on VideoPose code,
https://github.com/facebookresearch/VideoPose3D
the VPose model for single frame setting.
'''


import torch
import torch.nn as nn



# class TemporalModelBase(nn.Module):
#     """
#     Do not instantiate this class.
#     """
    
#     def __init__(self, num_joints_in, in_features, num_joints_out,
#                  filter_widths, causal, dropout, channels):
#         super().__init__()
        
#         # Validate input
#         for fw in filter_widths:
#             assert fw % 2 != 0, 'Only odd filter widths are supported'
        
#         self.num_joints_in = num_joints_in
#         self.in_features = in_features
#         self.num_joints_out = num_joints_out
#         self.filter_widths = filter_widths
        
#         self.drop = nn.Dropout(dropout)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.pad = [ filter_widths[0] // 2 ]
#         self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
#         self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

#     def set_bn_momentum(self, momentum):
#         self.expand_bn.momentum = momentum
#         for bn in self.layers_bn:
#             bn.momentum = momentum
            
#     def receptive_field(self):
#         """
#         Return the total receptive field of this model as # of frames.
#         """
#         frames = 0
#         for f in self.pad:
#             frames += f
#         return 1 + 2*frames
    
#     def total_causal_shift(self):
#         """
#         Return the asymmetric offset for sequence padding.
#         The returned value is typically 0 if causal convolutions are disabled,
#         otherwise it is half the receptive field.
#         """
#         frames = self.causal_shift[0]
#         next_dilation = self.filter_widths[0]
#         for i in range(1, len(self.filter_widths)):
#             frames += self.causal_shift[i] * next_dilation
#             next_dilation *= self.filter_widths[i]
#         return frames
        
#     def forward(self, x):
#         """
#         input: bx16x2 / bx32
#         output: bx16x3
#         """
#         if len(x.shape) == 2:
#             x = x.view(x.shape[0], 16, 2)
#         # pre-processing
#         # x = x.view(x.shape[0], x.shape[2], 16, 2)
#         x = x.view(x.shape[0], 1, 16, 2)

#         assert len(x.shape) == 4
#         assert x.shape[-2] == self.num_joints_in
#         assert x.shape[-1] == self.in_features
        
#         sz = x.shape[:3]
#         x = x.view(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
        
#         x = self._forward_blocks(x)
#         x = x.permute(0, 2, 1)
#         x = x.view(sz[0], -1, self.num_joints_out, 3)

#         # post process
#         x = x.view(sz[0], self.num_joints_out * 3)
#         # out: 15 joint ==> 16 joint
#         out = torch.cat([torch.zeros_like(x)[:,:3], x], 1).view(sz[0], 16, 3)  # Pad hip joint (0,0,0)
#         return out

# class TemporalModel(TemporalModelBase):
#     """
#     Reference 3D pose estimation model with temporal convolutions.
#     This implementation can be used for all use-cases.
#     """
    
#     def __init__(self, num_joints_in, in_features, num_joints_out,
#                  filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
#         """
#         Initialize this model.
        
#         Arguments:
#         num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
#         in_features -- number of input features for each joint (typically 2 for 2D input)
#         num_joints_out -- number of output joints (can be different than input)
#         filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
#         causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
#         dropout -- dropout probability
#         channels -- number of convolution channels
#         dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
#         """
#         super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
#         self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
#         layers_conv = []
#         layers_bn = []
        
#         self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
#         next_dilation = filter_widths[0]
#         for i in range(1, len(filter_widths)):
#             self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
#             self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
#             layers_conv.append(nn.Conv1d(channels, channels,
#                                          filter_widths[i] if not dense else (2*self.pad[-1] + 1),
#                                          dilation=next_dilation if not dense else 1,
#                                          bias=False))
#             layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
#             layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
#             layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
#             next_dilation *= filter_widths[i]
            
#         self.layers_conv = nn.ModuleList(layers_conv)
#         self.layers_bn = nn.ModuleList(layers_bn)
        
#     def _forward_blocks(self, x):
#         x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
#         for i in range(len(self.pad) - 1):
#             pad = self.pad[i+1]
#             shift = self.causal_shift[i+1]
#             res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
#             x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
#             x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
#         x = self.shrink(x)
#         return x
from utils.gan_utils import get_bone_unit_vecbypose3d, get_pose3dbyBoneVec, get_BoneVecbypose3d

def kcs_layer_hb_2d(x, num_joints=16):
    """
    torso part
    """
    bv = get_BoneVecbypose3d(x)
    mask = torch.zeros_like(bv)
    hb_idx = [0, 1,2,3,4,5, 6, 7, 8, 9,10,11, 12,13,14]
    mask[:, hb_idx, :] = 1
    bv = bv * mask
    
    Psi = torch.matmul(bv, bv.permute(0,1, 3, 2).contiguous())
    return Psi

class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames
    
    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames
        
    def forward(self, x):
        if len(x.shape) == 5:
            x=x[:,0]
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features
        
        sz = x.shape[:3]
        
        # Psi=kcs_layer_hb_2d(x,num_joints=self.num_joints_in)
        # x1 = Psi.view(Psi.shape[0], Psi.shape[1], -1)
        # x = get_BoneVecbypose3d(x)
     
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, (self.num_joints_out), 3)
        # x = x[:,13]
        # post process
        x = x.view(sz[0],  (self.num_joints_out) * 3)
        # x = get_pose3dbyBoneVec(x)
        # out: 15 joint ==> 16 joint
        out = torch.cat([torch.zeros_like(x)[:,:3], x], 1).view(sz[0], 16, 3)  # Pad hip joint (0,0,0)

        return out
  

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        
        return x

class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        
        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
 
        x = self.shrink(x)

        return x


# ########################################################
# ####               PoseFormer              #############
# ########################################################

# ## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

# import math
# import logging
# from functools import partial
# from collections import OrderedDict
# from einops import rearrange, repeat

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
#         #NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class PoseTransformer(nn.Module):
#     def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
#                  num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
#         """    ##########hybrid_backbone=None, representation_size=None,
#         Args:
#             num_frame (int, tuple): input frame number
#             num_joints (int, tuple): joints number
#             in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
#             embed_dim_ratio (int): embedding dimension ratio
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             drop_rate (float): dropout rate
#             attn_drop_rate (float): attention dropout rate
#             drop_path_rate (float): stochastic depth rate
#             norm_layer: (nn.Module): normalization layer
#         """
#         super().__init__()

#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
#         out_dim = num_joints * 3     #### output dimension is num_joints * 3

#         ### spatial patch embedding
#         self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
#         self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

#         self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)


#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

#         self.Spatial_blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])

#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
#             for i in range(depth)])

#         self.Spatial_norm = norm_layer(embed_dim_ratio)
#         self.Temporal_norm = norm_layer(embed_dim)

#         ####### A easy way to implement weighted mean
#         self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

#         self.head = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Linear(embed_dim , out_dim),
#         )


#     def Spatial_forward_features(self, x):
#         b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
#         x = rearrange(x, 'b c f p  -> (b f) p  c', )

#         x = self.Spatial_patch_to_embedding(x)
#         x += self.Spatial_pos_embed
#         x = self.pos_drop(x)

#         for blk in self.Spatial_blocks:
#             x = blk(x)

#         x = self.Spatial_norm(x)
#         x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
#         return x

#     def forward_features(self, x):
#         b  = x.shape[0]
#         x += self.Temporal_pos_embed
#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x = blk(x)

#         x = self.Temporal_norm(x)
#         ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
#         x = self.weighted_mean(x)
#         x = x.view(b, 1, -1)
#         return x


#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
#         b, _, _, p = x.shape
#         ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
#         x = self.Spatial_forward_features(x)
#         x = self.forward_features(x)
#         x = self.head(x)

#         x = x.view(b, 1, p, -1)

#         return x