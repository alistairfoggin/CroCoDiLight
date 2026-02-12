"""Transfer lighting inference script.

Applies the content (static features) from a reference image while varying
the lighting from each input image.

Usage:
    python scripts/inference/transfer_lighting.py --reference ref.png --input <folder> --output <folder>
"""

import argparse

import torch

from crocodilight.inference import (
    get_device, load_model, extract_features,
    save_tensor_image, process_input,
)


def main():
    parser = argparse.ArgumentParser(description="Apply fixed content from a reference with varying lighting")
    parser.add_argument("--reference", required=True, help="Reference image providing the content")
    parser.add_argument("--input", required=True, help="Input image or folder (provides lighting)")
    parser.add_argument("--output", required=True, help="Output image or folder")
    parser.add_argument("--model", default="pretrained_models/crocodilight.pth", help="Model checkpoint path")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=None, help="Resize images before processing")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    img_info = {"height": 448, "width": 448}

    # Extract content features from reference
    static_ref, _, pos_ref, _ = extract_features(model, args.reference, device, resize=args.resize)

    def process(img_path, out_path):
        _, dyn, _, tiling_module = extract_features(model, img_path, device, resize=args.resize)
        with torch.no_grad():
            feat = model.lighting_entangler(static_ref, pos_ref, dyn)
            out_img = model.croco.decode(feat, pos_ref, img_info)
            out_img = tiling_module.rebuild_with_masks(out_img)
        save_tensor_image(out_img, out_path)

    process_input(args.input, args.output, process)


if __name__ == "__main__":
    main()
