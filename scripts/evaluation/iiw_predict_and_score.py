"""IIW predict-and-score script.

Runs albedo estimation inference on the IIW dataset and computes WHDR scores
in a single pass. Combines inference + evaluation for convenience.

Usage:
    python scripts/evaluation/iiw_predict_and_score.py --iiw-root ./datasets/IIW/
"""

import argparse
import json
import os

import numpy as np
import torch

from evaluations.whdr import compute_whdr, load_image
from lighting.inference import (
    get_device, load_model, load_mapper, get_transform,
    load_and_transform, pad_to_min_size, unpad,
)
from lighting.relighting_modules import img_mean, img_std


def main():
    parser = argparse.ArgumentParser(description="Run albedo inference on IIW and compute WHDR")
    parser.add_argument("--iiw-root", default="./datasets/IIW/", help="Path to IIW data directory")
    parser.add_argument("--model", default="pretrained_models/CroCoDiLight.pth", help="Model checkpoint path")
    parser.add_argument("--mapper", default="pretrained_models/CroCoDiLight_albedo_mapper.pth", help="Albedo mapper weights path")
    parser.add_argument("--output-json", default="whdr_results_albedo_mapper_all.json", help="Output JSON for results")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, cpu). Auto-detects if not set.")
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.model, device)
    mapper = load_mapper(model, args.mapper, device)
    transform = get_transform()

    root_dir = args.iiw_root
    filenames = [f for f in os.listdir(root_dir) if f.endswith(".png") and "_ref" not in f]

    whdr_base = {}
    whdr_shifted = {}
    with torch.no_grad():
        for filename in filenames:
            path = os.path.join(root_dir, filename)
            img = load_and_transform(path, transform, device)
            img, pad_info = pad_to_min_size(img)

            delit_img = model.apply_mapper(img, mapper, use_consistency=False)
            delit_img = unpad(delit_img, pad_info)

            # Denormalize and save reflectance image
            out_img = delit_img[0].detach().cpu().permute(1, 2, 0).numpy()
            out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
            from matplotlib import pyplot as plt
            plt.imsave(path.replace(".png", "_ref.png"), np.clip(out_img, 0, 1))

            # Compute WHDR
            reflectance = load_image(filename=path.replace(".png", "_ref.png"), is_srgb=True)
            judgements = json.load(open(path.replace(".png", ".json")))
            score_base = compute_whdr(reflectance, judgements, 0.1)
            score_shifted = compute_whdr(reflectance + 0.5, judgements, 0.1)
            print(f"{filename}  WHDR: {score_base:.6f}  WHDR+0.5: {score_shifted:.6f}")
            whdr_base[filename] = score_base
            whdr_shifted[filename] = score_shifted

    avg_base = np.mean(np.array(list(whdr_base.values())))
    avg_shifted = np.mean(np.array(list(whdr_shifted.values())))
    print(f"\nAverage WHDR:      {avg_base:.6f}")
    print(f"Average WHDR+0.5:  {avg_shifted:.6f}")

    results = {fn: {"whdr": whdr_base[fn], "whdr_shifted": whdr_shifted[fn]} for fn in filenames}
    results["average"] = {"whdr": float(avg_base), "whdr_shifted": float(avg_shifted)}
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
