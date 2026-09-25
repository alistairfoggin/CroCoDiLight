"""Shared inference utilities for CroCoDiLight scripts.

Centralizes model loading, image transforms, tensor I/O, and feature extraction
patterns that were previously copy-pasted across 5+ scripts.
"""

import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from blended_tiling import TilingModule
from PIL import Image

from crocodilight.dataloader import img_mean, img_std
from crocodilight.relighting_model import (
    LightingMapper,
    RelightModule,
    load_relight_model,
)
from crocodilight.relighting_modules import rescale_image

def get_device(device_str=None):
    """Auto-detect CUDA or use specified device string.

    Args:
        device_str: Optional device string (e.g. 'cuda:0', 'cpu').
            If None, auto-detects CUDA availability.

    Returns:
        torch.device
    """
    if device_str is not None:
        return torch.device(device_str)
    return torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

def load_model(model_path: Path | str="pretrained_models/CroCoDiLight.pth", device=None):
    """Load RelightModule onto device, set to inference mode.

    Args:
        model_path: Path to the consolidated model checkpoint.
        device: torch.device or None for auto-detection.

    Returns:
        RelightModule in inference mode on the specified device.
    """
    device = get_device(device)
    model = load_relight_model(model_path, device)
    model.eval()
    return model


def load_mapper(model: RelightModule, mapper_path: Path | str, device=None):
    """Create a LightingMapper compatible with model, load weights, set to inference mode.

    Args:
        model: A loaded RelightModule (used to read encoder dimensions).
        mapper_path: Path to the mapper .pth weights.
        device: torch.device or None for auto-detection.

    Returns:
        LightingMapper in inference mode on the specified device.
    """
    device = get_device(device)
    mapper = LightingMapper(
        patch_size=model.croco.enc_embed_dim,
        extractor_depth=8,
        rope=model.croco.rope,
    ).to(device)
    mapper.load_mapper(torch.load(mapper_path, 'cpu'))
    mapper.eval()
    return mapper


def load_image(image_path: Path | str, device=None, resize=None, center_crop=None):
    """Load an image path as a normalized ``1,C,H,W`` tensor.

    Args:
        image_path: Path to image file.
        device: torch.device to move tensor to.
        resize: Optional resize dimension.
        center_crop: Optional center-crop dimension.

    Returns:
        torch.Tensor of shape (1, C, H, W).
    """
    with Image.open(image_path) as image:
        return pil_to_tensor(
            image, device=device, resize=resize, center_crop=center_crop
        )


def save_tensor_image(tensor: torch.Tensor, path: Path | str):
    """Denormalize tensor (rescale_image) and save as image file.

    Args:
        tensor: Image tensor of shape (1, C, H, W) or (C, H, W).
        path: Output file path.
    """
    tensor_to_pil(tensor).save(path)


def pil_to_tensor(image: Image.Image, device=None, resize=None, center_crop=None):
    """Convert a PIL Image to a normalised (1, C, H, W) tensor.

    Applies ImageNet normalisation matching the existing transform pipeline.

    Args:
        image: PIL Image (RGB).
        device: torch.device to place tensor on, or None for CPU.
        resize: Optional resize dimension.
        center_crop: Optional center-crop dimension.

    Returns:
        torch.Tensor of shape (1, C, H, W), ImageNet-normalised.
    """
    image = image.convert('RGB')
    ops: list[transforms.Transform] = [transforms.ToImage()]
    if resize is not None:
        ops.append(transforms.Resize(resize))
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops.extend([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    tensor = transforms.Compose(ops)(image).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor):
    """Convert a model output tensor back to a PIL Image.

    Denormalises using rescale_image() and converts to uint8 PIL.

    Args:
        tensor: Image tensor of shape (1, C, H, W) or (C, H, W).

    Returns:
        PIL Image (RGB).
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    img = rescale_image(tensor)[0].cpu().detach()
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    return Image.fromarray(img_np)


def _pad_to_min_size(img_tensor: torch.Tensor, min_size=448):
    """Pad image with zeros if smaller than min_size.

    Args:
        img_tensor: Tensor of shape (1, C, H, W).
        min_size: Minimum spatial dimension.

    Returns:
        (padded_tensor, pad_info) where pad_info is a dict with 'original_h' and 'original_w',
        or (img_tensor, None) if no padding was needed.
    """
    _, _, H, W = img_tensor.shape
    if H >= min_size and W >= min_size:
        return img_tensor, None
    new_H = max(H, min_size)
    new_W = max(W, min_size)
    padded = torch.zeros((1, 3, new_H, new_W), device=img_tensor.device, dtype=img_tensor.dtype)
    padded[:, :, :H, :W] = img_tensor
    return padded, {'original_h': H, 'original_w': W}


def _unpad(tensor: torch.Tensor, pad_info):
    """Crop padded tensor back to original size.

    Args:
        tensor: Padded tensor of shape (1, C, H, W).
        pad_info: Dict from _pad_to_min_size, or None (returns tensor unchanged).

    Returns:
        Cropped tensor matching original dimensions.
    """
    if pad_info is None:
        return tensor
    return tensor[:, :, :pad_info['original_h'], :pad_info['original_w']]


def _extract_features_from_tensor(model: RelightModule, img: torch.Tensor, tile_size: int=448, tile_overlap: float=0.2):
    """Core feature extraction from a pre-processed (1, C, H, W) tensor.

    Args:
        model: Loaded RelightModule.
        img: Image tensor of shape (1, C, H, W), already transformed/normalised.
        tile_size: Tile size for blended tiling.
        tile_overlap: Overlap fraction between tiles.

    Returns:
        (static, dyn, pos, tiling_module) feature tensors.
    """
    tiling_module = TilingModule(tile_size=tile_size, tile_overlap=tile_overlap, base_size=img.shape[2:])
    img = tiling_module.split_into_tiles(img)
    with torch.no_grad():
        feat, pos, _ = model.croco._encode_image(img, do_mask=False, return_all_blocks=False)
        static, dyn, _ = model.lighting_extractor(feat, pos)
    return static, dyn, pos, tiling_module


def extract_features(model: RelightModule, image_path: Path | str, device=None, resize=None,
                     tile_size=448, tile_overlap=0.2):
    """Extract static/dynamic features from image file using tiling.

    Args:
        model: Loaded RelightModule.
        image_path: Path to image file.
        device: torch.device.
        resize: Optional resize dimension.
        tile_size: Tile size for blended tiling.
        tile_overlap: Overlap fraction between tiles.

    Returns:
        (static, dyn, pos, tiling_module) feature tensors.
    """
    img = load_image(image_path, device, resize=resize)
    return _extract_features_from_tensor(model, img, tile_size, tile_overlap)


def extract_lighting(model: RelightModule, reference: torch.Tensor, tile_size: int=448, tile_overlap=0.2):
    """Extract lighting features from one normalized ``1,C,H,W`` tensor."""
    if reference.shape[0] != 1:
        raise ValueError("extract_lighting expects a single reference image")
    reference, _ = _pad_to_min_size(reference, tile_size)
    _, lighting, _, _ = _extract_features_from_tensor(
        model, reference, tile_size, tile_overlap
    )
    return lighting


def relight(
        model: RelightModule,
        images: torch.Tensor,
        lighting: torch.Tensor,
        tile_size=448, tile_overlap=0.2):
    """Apply lighting features to a normalized ``N,C,H,W`` image tensor."""
    outputs = []
    with torch.inference_mode():
        for image in images.split(1):
            image, pad_info = _pad_to_min_size(image, tile_size)
            static, _, pos, tiling_module = _extract_features_from_tensor(
                model, image, tile_size, tile_overlap
            )
            features = model.lighting_entangler(static, pos, lighting)
            output = model.croco.decode(
                features, pos, {"height": tile_size, "width": tile_size}
            )
            output = tiling_module.rebuild_with_masks(output)
            outputs.append(_unpad(output, pad_info))
    return torch.cat(outputs, dim=0)


def apply_mapper(model, images, mapper):
    """Apply a shadow, albedo, or other compatible mapper to an image batch."""
    outputs = []
    with torch.inference_mode():
        for image in images.split(1):
            image, pad_info = _pad_to_min_size(image)
            output = model.apply_mapper(image, mapper, use_consistency=False)
            outputs.append(_unpad(output, pad_info))
    return torch.cat(outputs, dim=0)


def process_input(input_path, output_path, process_fn):
    """Auto-detect file vs folder. Call process_fn for each image.

    If input_path is a file, processes that single file.
    If input_path is a directory, processes all image files within it.

    Args:
        input_path: Path to an image file or directory of images.
        output_path: Path to output file or directory.
        process_fn: Callable(input_path, output_path) for each image.
    """
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    if os.path.isfile(input_path):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        process_fn(input_path, output_path)
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        files = sorted(f for f in os.listdir(input_path) if f.lower().endswith(image_exts))
        for fname in files:
            in_path = os.path.join(input_path, fname)
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(output_path, out_name)
            process_fn(in_path, out_path)
            print(f"Processed {fname}, saved to {out_path}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
