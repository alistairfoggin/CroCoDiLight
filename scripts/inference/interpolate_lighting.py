"""Lighting interpolation inference script.

Interpolates dynamic (lighting) features between two images across N steps,
producing a smooth lighting transition sequence.

Usage:
    python scripts/inference/interpolate_lighting.py --frame-a a.png --frame-b b.png --output-dir out/ --steps 5
"""

import argparse
import os

import torch

from crocodilight.inference import (
    get_device, load_model, extract_features,
    save_tensor_image,
)


def main():
    parser = argparse.ArgumentParser(description="Interpolate crocodilight between two images")
    parser.add_argument("--frame-a", required=True, help="First frame")
    parser.add_argument("--frame-b", required=True, help="Second frame")
    parser.add_argument("--output-dir", required=True, help="Output directory for interpolated frames")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--model", default="pretrained_models/CroCoDiLight.pth", help="Model checkpoint path")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=1024, help="Resize images before processing")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    img_info = {"height": 448, "width": 448}
    os.makedirs(args.output_dir, exist_ok=True)

    static_a, dyn_a, pos_a, tiling_module_a = extract_features(
        model, args.frame_a, device, resize=args.resize)
    _, dyn_b, _, _ = extract_features(
        model, args.frame_b, device, resize=args.resize)

    for idx, alpha in enumerate(torch.linspace(0, 1, args.steps)):
        dyn_interp = dyn_a * (1 - alpha) + dyn_b * alpha

        with torch.no_grad():
            feat_interp = model.lighting_entangler(static_a, pos_a, dyn_interp)
            out_img = model.croco.decode(feat_interp, pos_a, img_info)
            out_img = tiling_module_a.rebuild_with_masks(out_img)

        save_tensor_image(out_img, os.path.join(args.output_dir, f"dyn_interp_{idx:02d}.png"))

    print(f"Saved {args.steps} interpolated frames to {args.output_dir}")


if __name__ == "__main__":
    main()
