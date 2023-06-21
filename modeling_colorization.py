# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math

from numpy.core.shape_base import block
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from functools import partial


from modeling_finetune import Block_mae_off, Mlp, _cfg, PatchEmbed, get_sinusoid_encoding_table, Bert_encoder, Block_poc, Biaffine, Conv_Upsample
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


###############################################
#                    encoder
################################################

class Colorization_VisionTransformerEncoder_off(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block_mae_off(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x = x + self.pos_embed[:, 1:, :]

        B, _, C = x.shape
        # x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

###############################################
#                    decoder
################################################

class Colorization_VisionTransformerDecoder_fusion_x(nn.Module):# 主实验decoder
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,depth_mlp=4, attn_mode = ''
                 ,upsample = False):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 2 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.upsample = upsample
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.attn_mode=attn_mode
        ########################################
        self.depth = depth
        blocks_poc = []
        for i in range(self.depth):
            blocks_poc.append(Block_poc(dim=embed_dim, num_heads=num_heads,       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,init_values=init_values))
        self.blocks_poc = nn.ModuleList(blocks_poc)
        
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # out dim = 512
        
        if self.upsample: # 上采样
            self.conv_upsample = Conv_Upsample()
        else: # use mlp
            self.depth_mlp = depth_mlp
            blocks_mlp = []
            for i in range(self.depth_mlp):
                blocks_mlp.append(Mlp(embed_dim))
            self.blocks_mlp = nn.ModuleList(blocks_mlp)

            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                            padding=1, bias=False)
        ########################################
        
        self.token_type_embeddings = nn.Embedding(3, embed_dim)
        
        self.biafine = self.arc_biaffine = Biaffine(embed_dim, embed_dim, 1, bias=(True, False))

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, obj, col, occm=None):

        x_type = self.token_type_embeddings(torch.zeros((x.size()[0],x.size()[1])).cuda().long())
        obj_type = self.token_type_embeddings(torch.full_like(obj[:,:,0], 1).cuda().long())
        col_type = self.token_type_embeddings(torch.full_like(col[:,:,0], 2).cuda().long())
        # print('x_type',x_type)
        x = x + x_type # B x L_p x C(emdding_dim)
        obj = obj + obj_type
        col = col + col_type
        # 过transformer
        poc = torch.cat([x, obj,col], dim=1)
        for i in range(self.depth):
            poc = self.blocks_poc[i](poc, self.attn_mode)
        p = poc[:,0:x.shape[1],:]
        o = poc[:,0:obj.shape[1],:]
        c = poc[:,0:obj.shape[1],:]
        # print('self.upsample',self.upsample)
        ################ 是否上采样
        if self.upsample: # deconv 上采样
            # B x N x dim(768) -> B x N x dim(512)
            p = self.head(self.norm(p))
            bs = p.shape[0]
            size = int(math.sqrt(p.shape[1]))
            dim = p.shape[-1]
            # B x dim(512) x N 
            p = p.permute(0,2,1)
            p = p.reshape(bs,dim,size,size)
            p = self.conv_upsample(p)
        else:
            # print('p.shape:',p.shape)
            for i in range(self.depth_mlp):# 过mlp
                p = self.blocks_mlp[i](p)
            
            p = self.head(self.norm(p)) # return ab [B, N, 2*16^2]
            # 最后过一层conv    
            p = rearrange(p, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=int(math.sqrt(p.shape[1])), w=int(math.sqrt(p.shape[1])),c=2,p1=int(math.sqrt(p.shape[-1]/2)))
            p = self.conv(p)
        occm_pred = self.biafine(o,c)
        return p, occm_pred

#################################################
#                    main model
#################################################
class Colorization_VisionTransformer_fusion_x(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=768, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=True,
                 attn_mode='',
                 upsample = False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        
        self.encoder = Colorization_VisionTransformerEncoder_off(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = Colorization_VisionTransformerDecoder_fusion_x(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            attn_mode = attn_mode,
            upsample = upsample)

        self.depth = encoder_depth + decoder_depth

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        
        # text_encoder
        self.text_encoder = Bert_encoder(decoder_embed_dim)

        # occm predictor
        # self.occm_pred = Occm_pred()

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, cap):
        
        x_vis = self.encoder(x) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        # print("x_vis.shape",x_vis.shape)
        # 对文本编码,并预测occm
        obj, col, occm = self.text_encoder(cap,x_vis)
        # 加入type_embedding

        #  the shape of x is [B, N, 2 * 16 * 16]
        x, occm_pred = self.decoder(x_vis, obj, col, occm) # [B, N, 2* 16 * 16]
        return x, occm_pred



#  register model
@register_model
def colorization_vit_large_patch16_224_fusion_whole_up(pretrained=False, **kwargs):
    model = Colorization_VisionTransformer_fusion_x(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=512,
        decoder_embed_dim=1024, 
        decoder_depth=12,
        decoder_num_heads=16,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        attn_mode = 'whole',
        upsample = True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 