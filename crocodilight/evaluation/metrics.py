"""Shared image quality metrics for crocodilight evaluation scripts.

Extracted from scripts/shadow_eval.py into a reusable module.
"""

import os

import cv2
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def list_images(folder):
    """List image files in a folder, sorted alphabetically.

    Args:
        folder: Path to directory.

    Returns:
        Sorted list of image filenames.
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def read_img(path):
    """Read an image file as RGB uint8 numpy array.

    Args:
        path: Path to image file.

    Returns:
        numpy array of shape (H, W, 3) in RGB order, dtype uint8.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def mae_rmse_lab(img1, img2):
    """Compute MAE and RMSE in CIELAB color space.

    Args:
        img1: RGB uint8 numpy array.
        img2: RGB uint8 numpy array.

    Returns:
        (mae, rmse) as floats.
    """
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = lab1 - lab2
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse


def psnr_score(img1, img2):
    """Compute PSNR between two images.

    Args:
        img1: uint8 numpy array.
        img2: uint8 numpy array.

    Returns:
        PSNR value as float.
    """
    return cv2.PSNR(img1, img2)


def ssim_score(img1, img2):
    """Compute SSIM on the Y channel (YCrCb color space).

    Args:
        img1: RGB uint8 numpy array.
        img2: RGB uint8 numpy array.

    Returns:
        SSIM value as float.
    """
    y1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    y2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    return ssim(y1, y2, data_range=255.0)


def create_lpips_fn(device=None):
    """Create an LPIPS (AlexNet) scoring function on the given device.

    Args:
        device: torch.device or None for auto-detection.

    Returns:
        lpips.LPIPS model on the specified device.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    return lpips.LPIPS(net='alex').to(device)


def calc_lpips(img1, img2, lpips_fn):
    """Compute LPIPS distance between two images.

    Args:
        img1: RGB uint8 numpy array.
        img2: RGB uint8 numpy array.
        lpips_fn: LPIPS model from create_lpips_fn().

    Returns:
        LPIPS distance as float.
    """
    device = next(lpips_fn.parameters()).device
    t1 = torch.from_numpy(img1.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(img2.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return lpips_fn(t1, t2).item()
