"""Shadow removal evaluation script.

Computes MAE, RMSE, LPIPS, PSNR, and SSIM metrics on shadow removal predictions
across multiple datasets (SRD, ISTD+, WSRD+, INS).

Usage:
    python scripts/evaluation/shadow_metrics.py --predictions <folder>
"""

import argparse
import os

import cv2
import numpy as np

from evaluations.metrics import (
    list_images, read_img, mae_rmse_lab, psnr_score, ssim_score,
    create_lpips_fn, calc_lpips,
)


def evaluate_folder(gt_folder, pred_folder, lpips_fn, suffix="", ext="png"):
    gt_files = list_images(gt_folder)
    pred_files = list_images(pred_folder)
    assert len(gt_files) == len(pred_files), "Number of ground truth and prediction images must be the same."

    maes, rmses, lpips_values, psnrs, ssims = [], [], [], [], []
    for fname in pred_files:
        if suffix and suffix not in fname:
            gt_name = fname.split('.')[0] + suffix + '.' + ext
        else:
            gt_name = fname
        gt_path = os.path.join(gt_folder, gt_name)
        pred_path = os.path.join(pred_folder, fname)
        gt_img = read_img(gt_path)
        pred_img = read_img(pred_path)

        # Resize to 256x256 if needed
        if gt_img.shape[:2] != (256, 256):
            gt_img = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_AREA)
        if pred_img.shape[:2] != (256, 256):
            pred_img = cv2.resize(pred_img, (256, 256), interpolation=cv2.INTER_AREA)
        if gt_img.shape != pred_img.shape:
            raise ValueError(f"Shape mismatch for {fname}")

        mae, rmse = mae_rmse_lab(gt_img, pred_img)
        lpips_value = calc_lpips(gt_img, pred_img, lpips_fn)
        maes.append(mae)
        rmses.append(rmse)
        lpips_values.append(lpips_value)
        psnrs.append(psnr_score(gt_img, pred_img))
        ssims.append(ssim_score(gt_img, pred_img))

    print(f"MAE (Lab): {np.mean(maes):.2f}")
    print(f"RMSE (Lab): {np.mean(rmses):.2f}")
    print(f"LPIPS: {np.mean(lpips_values):.3f}")
    print(f"PSNR: {np.mean(psnrs):.2f}")
    print(f"SSIM: {np.mean(ssims):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate shadow removal predictions")
    parser.add_argument("--predictions", required=True, help="Base folder containing prediction subfolders (SRD/, ISTD+/, WSRD+/, etc.)")
    parser.add_argument("--datasets-root", default="./datasets", help="Root directory containing ground truth datasets")
    parser.add_argument("--device", default=None, help="Device for LPIPS computation")
    args = parser.parse_args()

    from lighting.inference import get_device
    device = get_device(args.device)
    lpips_fn = create_lpips_fn(device)

    folder = args.predictions

    datasets = [
        ("SRD",   f"{args.datasets_root}/SRD/test/shadow_free",             f"{folder}/SRD",   "_free", "jpg"),
        ("ISTD+", f"{args.datasets_root}/ISTD+/test/test_C_fixed_official", f"{folder}/ISTD+", "",      "png"),
        ("WSRD+", f"{args.datasets_root}/WSRD+/val/gt",                     f"{folder}/WSRD+", "",      "png"),
        ("INS",   f"{args.datasets_root}/INS/test/shadow_free",             f"{folder}/INS",   "",      "png"),
    ]

    print(f"Evaluating predictions from: {folder}")
    for name, gt_dir, pred_dir, suffix, ext in datasets:
        if not os.path.isdir(pred_dir):
            print(f"\nSkipping {name}: prediction folder not found ({pred_dir})")
            continue
        if not os.path.isdir(gt_dir):
            print(f"\nSkipping {name}: ground truth folder not found ({gt_dir})")
            continue
        print(f"\nResults for {name}:")
        evaluate_folder(gt_dir, pred_dir, lpips_fn, suffix, ext)


if __name__ == "__main__":
    main()
