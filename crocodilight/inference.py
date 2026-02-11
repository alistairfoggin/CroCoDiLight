"""Shared inference utilities for crocodilight scripts.

Centralizes model loading, image transforms, tensor I/O, and feature extraction
patterns that were previously copy-pasted across 5+ scripts.
"""

import os

import cv2
import torch
import torchvision.transforms.v2 as transforms
from blended_tiling import TilingModule
from PIL import Image

from crocodilight.relighting_modules import img_mean, img_std, rescale_image
from crocodilight.relighting_model import load_relight_model, LightingMapper


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


def load_model(model_path="pretrained_models/crocodilight.pth", device=None):
    """Load RelightModule onto device, set to inference mode.

    Args:
        model_path: Path to the consolidated model checkpoint.
        device: torch.device or None for auto-detection.

    Returns:
        RelightModule in inference mode on the specified device.
    """
    if device is None:
        device = get_device()
    model = load_relight_model(model_path, device)
    model.eval()
    return model


def load_mapper(model, mapper_path, device=None):
    """Create a LightingMapper compatible with model, load weights, set to inference mode.

    Args:
        model: A loaded RelightModule (used to read encoder dimensions).
        mapper_path: Path to the mapper .pth weights.
        device: torch.device or None for auto-detection.

    Returns:
        LightingMapper in inference mode on the specified device.
    """
    if device is None:
        device = get_device()
    mapper = LightingMapper(
        patch_size=model.croco.enc_embed_dim,
        extractor_depth=8,
        rope=model.croco.rope,
    ).to(device)
    mapper.load_mapper(torch.load(mapper_path, 'cpu'))
    mapper.eval()
    return mapper


def get_transform(resize=None, center_crop=None):
    """Standard ImageNet-normalized transform with optional resize/crop.

    Args:
        resize: Optional int for Resize transform.
        center_crop: Optional int for CenterCrop transform.

    Returns:
        torchvision.transforms.Compose
    """
    ops = [transforms.ToImage()]
    if resize is not None:
        ops.append(transforms.Resize(resize))
    if center_crop is not None:
        ops.append(transforms.CenterCrop(center_crop))
    ops.extend([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    return transforms.Compose(ops)


def load_and_transform(image_path, transform, device):
    """Load image, apply transform, add batch dim, move to device.

    Args:
        image_path: Path to image file.
        transform: torchvision transform to apply.
        device: torch.device to move tensor to.

    Returns:
        torch.Tensor of shape (1, C, H, W).
    """
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def save_tensor_image(tensor, path):
    """Denormalize tensor (rescale_image) and save as image file.

    Args:
        tensor: Image tensor of shape (1, C, H, W) or (C, H, W).
        path: Output file path.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    img_np = (rescale_image(tensor)[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype('uint8')
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_np)


def pad_to_min_size(img_tensor, min_size=448):
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


def unpad(tensor, pad_info):
    """Crop padded tensor back to original size.

    Args:
        tensor: Padded tensor of shape (1, C, H, W).
        pad_info: Dict from pad_to_min_size, or None (returns tensor unchanged).

    Returns:
        Cropped tensor matching original dimensions.
    """
    if pad_info is None:
        return tensor
    return tensor[:, :, :pad_info['original_h'], :pad_info['original_w']]


def extract_features(model, image_path, device, transform, resize=None,
                     tile_size=448, tile_overlap=0.2):
    """Extract static/dynamic features from image using tiling.

    Args:
        model: Loaded RelightModule.
        image_path: Path to image file.
        device: torch.device.
        transform: Image transform to apply.
        resize: Optional resize dimension.
        tile_size: Tile size for blended tiling.
        tile_overlap: Overlap fraction between tiles.

    Returns:
        (static, dyn, pos, tiling_module) feature tensors.
    """
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    if resize is not None:
        img = transforms.Resize(resize)(img)
    tiling_module = TilingModule(tile_size=tile_size, tile_overlap=tile_overlap, base_size=img.shape[2:])
    img = tiling_module.split_into_tiles(img).to(device)
    with torch.no_grad():
        feat, pos, _ = model.croco._encode_image(img, do_mask=False, return_all_blocks=False)
        static, dyn, dyn_pos = model.lighting_extractor(feat, pos)
    return static, dyn, pos, tiling_module


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
