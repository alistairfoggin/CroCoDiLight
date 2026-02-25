from functools import partial

import torch
from torch import nn
from blended_tiling import TilingModule

from crocodilight.relighting_modules import DelightingTransformer, RelightingTransformer
from croco.models.blocks import Block
from croco.models.croco import CroCoNet
from croco.models.head_downstream import PixelwiseTaskWithDPT


def load_relight_model(checkpoint_path, device='cpu'):
    """Load a RelightModule from a consolidated checkpoint.

    Args:
        checkpoint_path: Path to a consolidated .pth file containing
            'croco_kwargs' and 'model' (RelightModule state dict).
        device: Device to load the model onto.

    Returns:
        A RelightModule in evaluation mode on the specified device.
    """
    ckpt = torch.load(checkpoint_path, device)
    croco_decode = CroCoDecode(**ckpt['croco_kwargs']).to(device)
    croco_decode.setup()
    croco_relight = RelightModule(croco_decode).to(device)
    croco_relight.load_state_dict(ckpt['model'])
    return croco_relight


class CroCoDecode(CroCoNet):
    def setup(self):
        self.freeze_encoder()
        self._set_mono_decoder(self.enc_embed_dim, self.enc_embed_dim, 16, 12, 2, partial(nn.LayerNorm, eps=1e-6))
        hooks_idx = [1, 4, 9, 11]
        head = PixelwiseTaskWithDPT(hooks_idx=hooks_idx, num_channels=3)
        head.setup(self)
        self.head = head

    def freeze_encoder(self, do_freeze=True):
        self.enc_blocks.requires_grad_(not do_freeze)
        self.patch_embed.requires_grad_(not do_freeze)
        self.enc_norm.requires_grad_(not do_freeze)

    def freeze_decoder(self, do_freeze=True):
        self.out_dec_blocks.requires_grad_(not do_freeze)
        self.out_decoder_embed.requires_grad_(not do_freeze)
        self.out_dec_norm.requires_grad_(not do_freeze)
        self.head.requires_grad_(not do_freeze)

    def _set_mono_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer):
        self.out_dec_depth = dec_depth
        self.out_dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.out_decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.out_dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.out_dec_norm = norm_layer(dec_embed_dim)

    def _mono_decoder(self, feat, pos, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # encoder to decoder layer
        f1_ = self.out_decoder_embed(feat)
        # apply Transformer blocks
        out = f1_
        if return_all_blocks:
            _out, out = out, []
            for blk in self.out_dec_blocks:
                _out = blk(_out, pos)
                out.append(_out)
            out[-1] = self.out_dec_norm(out[-1])
        else:
            for blk in self.out_dec_blocks:
                out = blk(out, pos)
            out = self.out_dec_norm(out)
        return out

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension
             than to encode them separately
        """
        out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0), do_mask=False,
                                         return_all_blocks=return_all_blocks)
        return out, pos

    def encode_decode(self, img):
        _, _, H, W = img.size()
        img_info = {'height': H, 'width': W}
        latents, pos, _ = self._encode_image(img, do_mask=False, return_all_blocks=False)
        decout = self._mono_decoder(latents, pos, return_all_blocks=True)
        return self.head(decout, img_info)

    def decode(self, latents, pos, img_info):
        decout = self._mono_decoder(latents, pos, return_all_blocks=True)
        return self.head(decout, img_info)


class LightingMapper(nn.Module):
    def __init__(self, patch_size=1024, extractor_depth=8, rope=None):
        super(LightingMapper, self).__init__()
        self.latent_mapper = RelightingTransformer(patch_size=patch_size, extractor_depth=extractor_depth, rope=rope)

    def freeze_mapper(self, do_freeze=True):
        self.latent_mapper.requires_grad_(not do_freeze)

    def load_mapper(self, state_dict):
        self.latent_mapper.load_state_dict(state_dict)

    def forward(self, static, pos, dyn, tiling_module=None):
        dyn_latents = self.latent_mapper.get_dynamic(static, pos, dyn)
        return dyn_latents


class RelightModule(nn.Module):
    def __init__(self, croco: CroCoDecode):
        super(RelightModule, self).__init__()
        self.croco = croco
        self.lighting_extractor = DelightingTransformer(patch_size=croco.enc_embed_dim, extractor_depth=8,
                                                        rope=croco.rope)
        self.lighting_entangler = RelightingTransformer(patch_size=croco.enc_embed_dim, extractor_depth=8,
                                                        rope=croco.rope)

    def freeze_components(self, encoder=True, decoder=True, extractor=False, entangler=False):
        """
        Freeze/unfreeze components for different training scenarios.

        Args:
            encoder: If True, freeze the CroCo encoder (default: True)
            decoder: If True, freeze the CroCo decoder (default: True)
            extractor: If True, freeze the CroCoDiLight extractor/disentangler (default: False)
            entangler: If True, freeze the CroCoDiLight entangler (default: False)

        Common configurations:
            - Relighting training: freeze_components(True, True, False, False)
            - Inference only: freeze_components(True, True, True, True)
        """
        self.croco.freeze_encoder(encoder)
        self.croco.freeze_decoder(decoder)
        self.lighting_extractor.requires_grad_(not extractor)
        self.lighting_entangler.requires_grad_(not entangler)

    def forward(self, img1, img2, do_tiling=True):
        if img1.shape[0] == 1 and do_tiling:
            tiling_module = TilingModule(tile_size=448, tile_overlap=0.2, base_size=img1.shape[2:])
            img1_resized = tiling_module.split_into_tiles(img1)
            img2_resized = tiling_module.split_into_tiles(img2)
        else:
            tiling_module = None
            img1_resized = img1
            img2_resized = img2
        _, _, H, W = img1_resized.size()  # Usually H, W = 448, 448
        img_info = {'height': H, 'width': W}

        feat, pos = self.croco.encode_image_pairs(img1_resized, img2_resized, return_all_blocks=False)
        static, dyn, dyn_pos = self.lighting_extractor(feat, pos)

        # Swap dyn1 and dyn2
        dyn1, dyn2 = dyn.chunk(2, dim=0)
        swapped_dyn = torch.cat((dyn2, dyn1), dim=0)

        # Relight img 1 to be like img 2
        relit_feat = self.lighting_entangler(static, pos, swapped_dyn)
        relit_img = self.croco.decode(relit_feat, pos, img_info)

        relit_img1, relit_img2 = relit_img.chunk(2, dim=0)
        if tiling_module is not None:
            relit_img1 = tiling_module.rebuild_with_masks(relit_img1)
            relit_img2 = tiling_module.rebuild_with_masks(relit_img2)

        return (relit_img1, relit_img2,
                static, pos, dyn, tiling_module)

    def apply_mapper(self, img, mapper: LightingMapper, use_consistency=True, return_dyn=False):
        tiling_module = TilingModule(tile_size=448, tile_overlap=0.2, base_size=img.shape[2:])
        img_resized = tiling_module.split_into_tiles(img)

        _, _, H, W = img_resized.size()
        img_info = {'height': H, 'width': W}

        feat, pos, _ = self.croco._encode_image(img_resized, do_mask=False, return_all_blocks=False)
        static, dyn, dyn_pos = self.lighting_extractor(feat, pos)
        mapped_dyn = mapper(static, pos, dyn, tiling_module) if use_consistency else mapper(static, pos, dyn, None)
        mapped_feat = self.lighting_entangler(static, pos, mapped_dyn)
        mapped_img = self.croco.decode(mapped_feat, pos, img_info)

        mapped_img = tiling_module.rebuild_with_masks(mapped_img)
        return mapped_img if not return_dyn else (mapped_img, dyn)

    def apply_gt_mapper(self, img, gt_img):
        """
        This is only used for the Oracle Shadow Removal metric where you have the shadow removal GT image for reference.
        """
        tiling_module = TilingModule(tile_size=448, tile_overlap=0.2, base_size=img.shape[2:])
        img_resized = tiling_module.split_into_tiles(img)
        gt_img_resized = tiling_module.split_into_tiles(gt_img)
        _, _, H, W = img_resized.size()
        img_info = {'height': H, 'width': W}

        feat, pos = self.croco.encode_image_pairs(img_resized, gt_img_resized, return_all_blocks=False)
        static, dyn, dyn_pos = self.lighting_extractor(feat, pos)
        static, _ = static.chunk(2, dim=0)
        dyn, dyn_gt = dyn.chunk(2, dim=0)
        pos, _ = pos.chunk(2, dim=0)

        mapped_dyn = dyn_gt
        mapped_feat = self.lighting_entangler(static, pos, mapped_dyn)
        mapped_img = self.croco.decode(mapped_feat, pos, img_info)
        if tiling_module is not None:
            mapped_img = tiling_module.rebuild_with_masks(mapped_img)
        return mapped_img
