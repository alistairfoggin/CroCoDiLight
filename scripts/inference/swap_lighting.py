"""Lighting swap inference script.

Swaps lighting between two images, or applies a mapper to a single image.

Usage:
    # Swap lighting between two images:
    python scripts/inference/swap_lighting.py --image1 img1.png --image2 img2.png --output-dir out/

    # Apply mapper to a single image:
    python scripts/inference/swap_lighting.py --image1 img.png --mapper pretrained_models/CroCoDiLight_shadow_mapper.pth --output-dir out/
"""

import argparse
import os

import torch

from crocodilight.inference import (
    get_device, load_model, load_mapper, get_transform,
    load_and_transform, save_tensor_image,
)
def main():
    parser = argparse.ArgumentParser(description="Swap lighting between images or apply a mapper")
    parser.add_argument("--image1", required=True, help="First input image")
    parser.add_argument("--image2", default=None, help="Second input image (for swap mode)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default="pretrained_models/crocodilight.pth", help="Model checkpoint path")
    parser.add_argument("--mapper", default=None, help="Mapper weights path (for single-image mode)")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=448, help="Resize images to this size")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    transform = get_transform(resize=args.resize, center_crop=args.resize)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mapper:
        # Single-image mapper mode
        mapper = load_mapper(model, args.mapper, device)
        img = load_and_transform(args.image1, transform, device)
        with torch.no_grad():
            mapped_img = model.apply_mapper(img, mapper)
        save_tensor_image(mapped_img, os.path.join(args.output_dir, "mapped.png"))
        save_tensor_image(img, os.path.join(args.output_dir, "input.png"))
    else:
        # Two-image swap mode
        if args.image2 is None:
            parser.error("--image2 is required for swap mode (or use --mapper for single-image mode)")
        img1 = load_and_transform(args.image1, transform, device)
        img2 = load_and_transform(args.image2, transform, device)
        with torch.no_grad():
            img1_relit, img2_relit, *_ = model(img1, img2, do_tiling=False)
        save_tensor_image(img1_relit, os.path.join(args.output_dir, "image1_relit.png"))
        save_tensor_image(img2_relit, os.path.join(args.output_dir, "image2_relit.png"))


if __name__ == "__main__":
    main()
