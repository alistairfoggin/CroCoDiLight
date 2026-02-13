"""Albedo estimation inference script.

Applies the albedo mapper to input images, with padding for small images.

Usage:
    python scripts/inference/albedo_estimation.py --input <path> --output <path>
"""

import argparse

import torch

from crocodilight.inference import (
    get_device, load_model, load_mapper, get_transform,
    load_and_transform, save_tensor_image, process_input,
    pad_to_min_size, unpad,
)


def main():
    parser = argparse.ArgumentParser(description="Estimate albedo from images using crocodilight")
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--output", required=True, help="Output image or folder")
    parser.add_argument("--model", default="pretrained_models/CroCoDiLight.pth", help="Model checkpoint path")
    parser.add_argument("--mapper", default="pretrained_models/CroCoDiLight_albedo_mapper.pth", help="Albedo mapper weights path")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    parser.add_argument("--resize", type=int, default=None, help="Resize images before processing")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    mapper = load_mapper(model, args.mapper, device)
    transform = get_transform()

    def process(img_path, out_path):
        img = load_and_transform(img_path, transform, device, resize=args.resize)
        img, pad_info = pad_to_min_size(img)
        with torch.no_grad():
            result = model.apply_mapper(img, mapper, use_consistency=False)
        result = unpad(result, pad_info)
        save_tensor_image(result, out_path)

    process_input(args.input, args.output, process)


if __name__ == "__main__":
    main()
