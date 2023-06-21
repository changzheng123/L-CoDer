# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from transformers import BertTokenizer, BertConfig
from transformers import BertModel

import sys

MAX_CAP_LEN = 20 
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class Sobel_conv(nn.Module):
    def __init__(self):
        super(Sobel_conv, self).__init__()
        kernel_v = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
    
    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)#(3, B, num_heads, N, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)#( B, num_heads, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_crossmodal(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.k_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.v_linear = nn.Linear(dim, all_head_dim , bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, obj, col, occm=None):
        B, N_p, C = x.shape 
        B, N_o, C = obj.shape

        qkv_bias = None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = F.linear(input=x, weight=self.q_linear.weight, bias=q_bias).reshape(B, N_p, self.num_heads, -1).permute( 0, 2, 1, 3)
        k = F.linear(input=obj, weight=self.k_linear.weight, bias=k_bias).reshape(B, N_o, self.num_heads, -1).permute( 0, 2, 1, 3)
        v = F.linear(input=col, weight=self.v_linear.weight, bias=v_bias).reshape(B, N_o, self.num_heads, -1).permute( 0, 2, 1, 3)

        q = q * self.scale # [B, num_heads, N_p, dim]
        attn = (q @ k.transpose(-2, -1)) # [B, num_heads, N_p, N_o]

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_p, -1)
        # print("bolck_corss:",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn.transpose(1, 2).reshape(B, N_p, -1)#[B, N_p, num_head*N_o]

class Attention_po_pc(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.k_linear = nn.Linear(dim, all_head_dim , bias=False)
        self.v_linear = nn.Linear(dim, all_head_dim , bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, po, pc, occm=None):
        # obj.shape(col) = B, N_l, C 
        # x.shape= B, N_p, C 
        B, N, C = po.shape
        qkv_bias = None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = F.linear(input=po, weight=self.q_linear.weight, bias=q_bias).reshape(B, N, self.num_heads, -1).permute( 0, 2, 1, 3)
        k = F.linear(input=pc, weight=self.k_linear.weight, bias=k_bias).reshape(B, N, self.num_heads, -1).permute( 0, 2, 1, 3)
        v = F.linear(input=pc, weight=self.v_linear.weight, bias=v_bias).reshape(B, N, self.num_heads, -1).permute( 0, 2, 1, 3)

        q = q * self.scale #[B, num_heads, N, dim]
        attn = (q @ k.transpose(-2, -1)) # [B, num_heads, N, N]

        
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # print("bolck_corss:",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn.transpose(1, 2).reshape(B, N, -1)

class Attention_poc(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mode=''):
        # print(attn_mode)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B , n_head, N, dim
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale #( B, num_heads, N, N)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        self.attn_ori = attn.clone().detach()
        split1 = 196
        split2 = 216
        if attn_mode == "whole":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2] # exchange
            attn[:, :, :split1, split1:split2] -= 100 # po
            attn[:, :, split1:split2, split2:] -= 100 # oc
            attn[:, :, split2:, split1:split2] -= 100 # co
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        elif attn_mode == "selflanguage":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2]
            attn[:, :, :split1, split1:split2] -= 1000
            attn[:, :, split1:split2, split2:] -= 1000
            attn[:, :, split2:, split1:split2] -= 1000
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        else:
            print("attn_mode ERROR")
            sys.exit(0)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_poc_clip(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mode=''):
        # print(attn_mode)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B , n_head, N, dim
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale #( B, num_heads, N, N)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        self.attn_ori = attn.clone().detach()
        split1 = 196
        split2 = 216
        if attn_mode == "whole":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2] # exchange
            attn[:, :, :split1, split1:split2] -= 100 # po
            attn[:, :, split1:split2, split2:] -= 100 # oc
            attn[:, :, split2:, split1:split2] -= 100 # co
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        elif attn_mode == "whole_tsf":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2] @ attn[:, :, split1:split2, split2:,]# exchange
            attn[:, :, :split1, split1:split2] -= 100 # po
            attn[:, :, split1:split2, split2:] -= 100 # oc
            attn[:, :, split2:, split1:split2] -= 100 # co
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        elif attn_mode == "splitSoftmax":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2] # exchange
            attn[:, :, :split1, split1:split2] -= 100 # po
            attn[:, :, split1:split2, split2:] -= 100 # oc
            attn[:, :, split2:, 0:split2] -= 100 # c-po
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        elif attn_mode == "fixlanguage":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2] # exchange
            attn[:, :, :split1, split1:split2] -= 1000 # po
            attn[:, :, split1:split2, :] -= 1000 # o-poc
            attn[:, :, split2:, :] -= 1000 # c-poc
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        elif attn_mode == "selflanguage":    
            attn[:, :, :split1, split2:] = attn[:, :, :split1, split1:split2]
            attn[:, :, :split1, split1:split2] -= 1000
            attn[:, :, split1:split2, split2:] -= 1000
            attn[:, :, split2:, split1:split2] -= 1000
            attn = attn.softmax(dim=-1)
            self.attn_sm = attn.clone().detach()
            attn = self.attn_drop(attn)
        else:
            print("attn_mode ERROR")
            sys.exit(0)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_mae_off(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_mae_off(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_mae_off(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block_patch(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block_crossmodal(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, mask_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_col = norm_layer(dim)
        self.norm_obj = norm_layer(dim)
        self.proj_obj = nn.Linear(768,dim)
        self.proj_col = nn.Linear(768,dim)
        assert mask_dim != None, f"mask_dim == None"
        self.mask_proj = nn.Sequential(nn.Linear(MAX_CAP_LEN*num_heads,mask_dim),nn.ReLU(),nn.Linear(mask_dim,mask_dim)) 
        self.attn = Attention_crossmodal(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, obj, col, occm,attn_mask=None):
        obj = self.proj_obj(obj)
        col = self.proj_col(col)
        if self.gamma_1 is None:
            x_ = self.norm1(x)
            obj = self.norm_obj(obj)
            col = self.norm_col(col)
            x_, attn_map = self.attn(x_,obj,col,occm) # attn_map.shape = B x num_head x N_p x N_l
            # print('attn_map.shape',attn_map.shape)
            x_ = x + self.drop_path(x_)
            x_ = x_ + self.drop_path(self.mlp(self.norm2(x_))) 
            mask = self.mask_proj(attn_map)
        else:
            x_ = self.norm1(x)
            obj = self.norm_obj(obj)
            col = self.norm_col(col)
            x_, attn_map = self.attn(x_,obj,col,occm) # attn_map.shape = B x N_p x N_l
            x_ = x + self.drop_path(self.gamma_1 * x_)
            x_ = x_ + self.drop_path(self.gamma_2 *self.mlp(self.norm2(x_))) 
            mask = self.mask_proj(attn_map)
        return x_, mask

class Block_po_pc(nn.Module):
   
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, mask_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # assert mask_dim != None, f"mask_dim == None"
        # self.mask_proj = nn.Sequential(nn.Linear(MAX_CAP_LEN,mask_dim),nn.ReLU(),nn.Linear(mask_dim,mask_dim)) 
        self.attn = Attention_po_pc(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, po, pc, occm, attn_mask=None):
        # 这里的obj和col经过了type编码

        if self.gamma_1 is None:            
            po_ = self.norm1(po)
            pc_ = self.norm2(pc)
            x, attn_map = self.attn(po_,pc_,occm) # attn_map.shape = B x N_p x N_l
            x = po + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x))) 
            # mask = self.mask_proj(attn_map)
        else:
            po_ = self.norm1(po)
            pc_ = self.norm2(pc)
            x, attn_map = self.attn(po_,pc_,occm) # attn_map.shape = B x N_p x N_l
            print('attn_map',attn_map.shape)
            x = po + self.drop_path(self.gamma_1*x)
            x = x + self.drop_path(self.mlp(self.gamma_2 *self.norm2(x))) 
            # mask = self.mask_proj(attn_map)
        return x

class Block_poc(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_poc(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mode=''):
        # print(attn_mode)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x),attn_mode))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),attn_mode))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Block_poc_clip(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_poc_clip(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mode=''):
        # print(attn_mode)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x),attn_mode))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),attn_mode))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class NonLinear(nn.Module):
    def __init__(self, input_size, hidden_size, activation=None):
        super(NonLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError("activation must be callable: type={}".format(type(activation)))
            self._activate = activation


    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        self.linear.weight.data.copy_(torch.from_numpy(W))

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            input1 = torch.cat((input1, torch.tensor(ones)), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, torch.tensor(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1) #

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        # biaffine = torch.sigmoid(biaffine)
        return biaffine

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'

class Bert_encoder(nn.Module):
    def __init__(self,decoder_emb):
        super().__init__()
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        model_name = 'bert-base-uncased'
        model_config = BertConfig.from_pretrained(model_name)
        # 修改配置
        model_config.output_hidden_states = True
        # model_config.output_attentions = True
        self.bert_model = BertModel.from_pretrained(model_name,config = model_config)

        self.mlp_arc_object = NonLinear(
            input_size = 768,
            hidden_size = decoder_emb,
            activation = nn.ReLU())
        self.mlp_arc_color = NonLinear(
            input_size = 768,
            hidden_size = decoder_emb,
            activation = nn.ReLU())
        self.arc_biaffine = Biaffine(decoder_emb, decoder_emb, 1, bias=(True, False))
        

    def forward(self,txts,vis=None):
        token_ids = []
        for txt in txts:
            token_id = self.tokenizer.encode(txt,add_special_tokens=False,max_length=MAX_CAP_LEN, pad_to_max_length=True)
            token_ids.append(token_id)
        token_tensor = torch.LongTensor(token_ids).cuda()
        cap_emb = self.bert_model(token_tensor)['last_hidden_state']
        obj_emb = self.mlp_arc_object(cap_emb) # b x N_l x dim
        col_emb = self.mlp_arc_color(cap_emb)
        
        # arc_logit = self.arc_biaffine(col_emb.detach(),obj_emb.detach())
        # arc_logit = arc_logit.squeeze(-1)
        # arc_logit = torch.sigmoid(arc_logit)
        arc_logit =None

        return obj_emb, col_emb, arc_logit

def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv3x3_in_relu(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.InstanceNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block

def conv3x3_tanh(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding and tanh"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.Tanh()
    )
    return block

class Conv_Upsample(nn.Module):
    def __init__(self,):
        super(Conv_Upsample, self).__init__()

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.r1 = nn.ReLU(True)
        self.c1 = conv3x3_in_relu(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.r2 = nn.ReLU(True)
        self.c2 = conv3x3_in_relu(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.r3 = nn.ReLU(True)
        self.c3 = conv3x3_in_relu(64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.r4 = nn.ReLU(True)
        self.c4 = conv3x3_tanh(32, 2)

    def forward(self, p):
        """
        :param img_l: batch x 1 x ih x iw
        """
        
        output = self.up1(p)
        output = self.r1(output)
        output = self.c1(output)# 256 x 28 x28
        
        output = self.up2(output)
        output = self.r2(output)
        output = self.c2(output)# 128 x 56 x 56
        
        output = self.up3(output)
        output = self.r3(output)
        output = self.c3(output)# 64 x 112 x 112

        output = self.up4(output)# 32 x 224 x 224
        output = self.r4(output)
        output = self.c4(output)# 2 x 224 x 224

        return output

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 