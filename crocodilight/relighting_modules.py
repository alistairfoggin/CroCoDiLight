from functools import partial

import torch
from torch import nn

from croco.models.blocks import Block

from crocodilight.dataloader import img_mean, img_std


class DelightingTransformer(nn.Module):
    def __init__(self, patch_size=1024, num_heads=16, mlp_ratio=2, extractor_depth=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 rope=None):
        super(DelightingTransformer, self).__init__()
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


class RelightingTransformer(nn.Module):
    def __init__(self, patch_size=1024, num_heads=16, mlp_ratio=2, extractor_depth=2,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), rope=None):
        super(RelightingTransformer, self).__init__()
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


def rescale_image(img):
    mean = torch.tensor(img_mean, device=img.device, dtype=img.dtype).reshape(1, -1, 1, 1)
    std = torch.tensor(img_std, device=img.device, dtype=img.dtype).reshape(1, -1, 1, 1)
    return torch.clamp(img * std + mean, min=0., max=1.)
