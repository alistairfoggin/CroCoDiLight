"""Shadow removal inference script.

Applies the shadow removal mapper to input images.

Usage:
    python scripts/inference/shadow_removal.py --input <path> --output <path>
"""

import argparse

from crocodilight.inference import (
    get_device, load_model, load_mapper,
    load_image, save_tensor_image, process_input, apply_mapper,
)


def main():
    parser = argparse.ArgumentParser(description="Remove shadows from images using CroCoDiLight")
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--output", required=True, help="Output image or folder")
    parser.add_argument("--model", default="pretrained_models/CroCoDiLight.pth", help="Model checkpoint path")
    parser.add_argument("--mapper", default="pretrained_models/CroCoDiLight_shadow_mapper.pth", help="Shadow mapper weights path")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=None, help="Resize images before processing")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    mapper = load_mapper(model, args.mapper, device)
    def process(img_path, out_path):
        img = load_image(img_path, device, resize=args.resize)
        result = apply_mapper(model, img, mapper)
        save_tensor_image(result, out_path)

    process_input(args.input, args.output, process)


if __name__ == "__main__":
    main()
