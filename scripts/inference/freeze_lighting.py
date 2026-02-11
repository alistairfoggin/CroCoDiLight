"""Freeze lighting inference script.

Applies the lighting from a reference image to varying content images.
The reference image provides fixed dynamic (lighting) features, while each
input image provides its own static (content) features.

Usage:
    python scripts/inference/freeze_lighting.py --reference ref.png --input <folder> --output <folder>
"""

import argparse

import torch

from crocodilight.inference import (
    get_device, load_model, get_transform, extract_features,
    save_tensor_image, process_input,
)


def main():
    parser = argparse.ArgumentParser(description="Apply fixed lighting from a reference to varying content images")
    parser.add_argument("--reference", required=True, help="Reference image providing the lighting")
    parser.add_argument("--input", required=True, help="Input image or folder (provides content)")
    parser.add_argument("--output", required=True, help="Output image or folder")
    parser.add_argument("--model", default="pretrained_models/crocodilight.pth", help="Model checkpoint path")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=None, help="Resize images before processing")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    transform = get_transform()
    img_info = {"height": 448, "width": 448}

    # Extract crocodilight features from reference
    _, dyn_ref, _, _ = extract_features(model, args.reference, device, transform, resize=args.resize)

    def process(img_path, out_path):
        static, _, pos, tiling_module = extract_features(model, img_path, device, transform, resize=args.resize)
        with torch.no_grad():
            feat = model.lighting_entangler(static, pos, dyn_ref)
            out_img = model.croco.decode(feat, pos, img_info)
            out_img = tiling_module.rebuild_with_masks(out_img)
        save_tensor_image(out_img, out_path)

    process_input(args.input, args.output, process)


if __name__ == "__main__":
    main()
