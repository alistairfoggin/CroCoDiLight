"""Albedo WHDR evaluation script.

Computes WHDR on pre-saved albedo reflectance images against IIW human judgements.
Uses the test split (every 5th image) as specified in the IIW evaluation protocol.

Usage:
    python scripts/evaluation/albedo_whdr.py --iiw-root ./datasets/IIW/
"""

import argparse
import json
import os

import numpy as np

from crocodilight.evaluation.whdr import compute_whdr, load_image


def main():
    parser = argparse.ArgumentParser(description="Evaluate albedo estimation using WHDR on IIW dataset")
    parser.add_argument("--iiw-root", default="./datasets/IIW/", help="Path to IIW data directory")
    parser.add_argument("--output", default=None, help="Output JSON file for per-image WHDR scores")
    parser.add_argument("--every-n", type=int, default=5, help="Evaluate every Nth image (test split)")
    args = parser.parse_args()

    root_dir = args.iiw_root
    filenames = [f for f in os.listdir(root_dir) if f.endswith(".png") and "_ref" not in f]
    # Sort numerically and take test split
    filenames = sorted(filenames, key=lambda x: int(x.split(".")[0]))
    filenames = filenames[::args.every_n]

    whdr_base = {}
    whdr_shifted = {}
    for filename in filenames:
        path = os.path.join(root_dir, filename)
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

    if args.output:
        results = {fn: {"whdr": whdr_base[fn], "whdr_shifted": whdr_shifted[fn]} for fn in filenames}
        results["average"] = {"whdr": float(avg_base), "whdr_shifted": float(avg_shifted)}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
