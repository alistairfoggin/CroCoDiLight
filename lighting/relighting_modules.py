from functools import partial

import torch
from torch import nn

from croco.models.blocks import Block, Mlp, Attention, DropPath


class LightingExtractor(nn.Module):
    def __init__(self, patch_size=1024, num_heads=16, mlp_ratio=2, extractor_depth=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 rope=None):
        super(LightingExtractor, self).__init__()
        # B x num_tokens x patch_size
        self.dynamic_token = nn.Parameter(torch.randn(1, 1, patch_size), requires_grad=True)
        self.base_blocks = nn.ModuleList()
        for _ in range(extractor_depth):
            self.base_blocks.append(
                Block(patch_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope))

    def forward(self, x, xpos):
        x = torch.cat((x, self.dynamic_token.expand(x.shape[0], 1, self.dynamic_token.shape[2])), dim=1)
        dyn_pos = torch.ones(xpos.shape[0], 1, xpos.shape[2], dtype=xpos.dtype, device=xpos.device) * -1
        xpos_extra = torch.cat((xpos, dyn_pos), dim=1)
        # x: [B, 196+1, 1024]
        for blk in self.base_blocks:
            x = blk(x, xpos_extra)
        # static: [B, 196, 1024]
        static = x[:, :-1, :]
        # dynamic: [B, 1, 1024]
        dynamic = x[:, -1:, :]
        return static, dynamic, dyn_pos


class LightingEntangler(nn.Module):
    def __init__(self, patch_size=1024, num_heads=16, mlp_ratio=2, extractor_depth=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=None):
        super(LightingEntangler, self).__init__()
        self.base_blocks = nn.ModuleList()
        for _ in range(extractor_depth):
            self.base_blocks.append(
                Block(patch_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope))

    def forward(self, x, xpos, dyn):
        # x = x + dyn
        x = torch.cat((x, dyn.expand(x.shape[0], 1, dyn.shape[2])), dim=1)
        dyn_pos = torch.ones(xpos.shape[0], 1, xpos.shape[2], dtype=xpos.dtype, device=xpos.device) * -1
        xpos_extra = torch.cat((xpos, dyn_pos), dim=1)
        for blk in self.base_blocks:
            x = blk(x, xpos_extra)
        # return x
        # static: [B, 196, 1024]
        static = x[:, :-1, :]
        # dynamic: [B, 1, 1024]
        # dynamic = x[:, -1:, :]
        return static  # , dynamic

    def get_dynamic(self, x, xpos, dyn):
        x = torch.cat((x, dyn.expand(x.shape[0], 1, dyn.shape[2])), dim=1)
        dyn_pos = torch.ones(xpos.shape[0], 1, xpos.shape[2], dtype=xpos.dtype, device=xpos.device) * -1
        xpos_extra = torch.cat((xpos, dyn_pos), dim=1)
        for blk in self.base_blocks:
            x = blk(x, xpos_extra)
        dynamic = x[:, -1:, :]
        return dynamic


class LightingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)

    def forward(self, x, xpos, dynamic_feature):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(torch.cat((self.norm2(x), dynamic_feature.expand(-1, x.shape[1], -1)), dim=-1)))
        return x


class LightingDecoder(nn.Module):
    def __init__(self, embedding_dim=1024, num_heads=16, mlp_ratio=4, patch_size=16, num_blocks=4, rope=None):
        super(LightingDecoder, self).__init__()
        self.dec_blocks = nn.ModuleList()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        for _ in range(num_blocks):
            self.dec_blocks.append(
                LightingBlock(embedding_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=rope))
        self.dec_norm = norm_layer(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, patch_size ** 2 * 3)

    def forward(self, x, xpos, dynamic_feature):
        # static_features: [B, 196, 1024]
        # dynamic_feature: [B,   1, 1024]
        for blk in self.dec_blocks:
            x = blk(x, xpos, dynamic_feature)
        x = self.dec_norm(x)
        out = self.prediction_head(x)
        return out


img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def rescale_image(img):
    mean = torch.tensor(img_mean, device=img.device, dtype=img.dtype).reshape(1, -1, 1, 1)
    std = torch.tensor(img_std, device=img.device, dtype=img.dtype).reshape(1, -1, 1, 1)
    return torch.clamp(img * std + mean, min=0., max=1.)
