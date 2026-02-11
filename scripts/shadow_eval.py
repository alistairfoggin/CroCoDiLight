import os
import lpips
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

import torch
from skimage.metrics import structural_similarity as ssim


def list_images(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
lpips_fn = lpips.LPIPS(net='alex').to(device)


def calc_lpips(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img1 = img1 * 2 - 1  # Scale to [-1, 1]
    img2 = img2 * 2 - 1
    with torch.no_grad():
        lpips_value = lpips_fn(
            torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device),
            torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
        ).item()
    return lpips_value


def mae_rmse_lab(img1, img2):
    # Convert to LAB (expects uint8, BGR)
    lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = lab1 - lab2
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    return mae, rmse


def psnr_cv2(img1, img2):
    return cv2.PSNR(img1, img2)


def ssim_skimage(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb).astype(np.float32)[:, :, 0]
    return ssim(img1, img2, data_range=255.0)


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def evaluate_folder(gt_folder, pred_folder, suffix="", ext="png"):
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
        lpips_value = calc_lpips(gt_img, pred_img)
        maes.append(mae)
        rmses.append(rmse)
        lpips_values.append(lpips_value)
        psnrs.append(psnr_cv2(gt_img, pred_img))
        ssims.append(ssim_skimage(gt_img, pred_img))

    print(f"MAE (Lab): {np.mean(maes):.2f}")
    print(f"RMSE (Lab): {np.mean(rmses):.2f}")
    print(f"LPIPS: {np.mean(lpips_values):.3f}")
    print(f"PSNR: {np.mean(psnrs):.2f}")
    print(f"SSIM: {np.mean(ssims):.3f}")


def main_diffs():
    for folder in ["CroCoDiLight", "HomoFormer", "OmniSR", "StableShadowRemoval"]:
        for file in os.listdir(os.path.join("Comparisons", folder)):
            if "_diff" in file:
                continue
            if file.endswith("0.png"):
                gt_filename = file.replace(".png", "_gt.jpg")
            elif file.endswith(".png"):
                gt_filename = file.replace(".png", "_gt.png")
            else:
                gt_filename = file.replace(".jpg", "_gt.jpg")
            gt_path = os.path.join("Comparisons/Inputs", gt_filename)
            pred_path = os.path.join("Comparisons", folder, file)
            gt_img = read_img(gt_path)
            pred_img = read_img(pred_path)
            if gt_img.shape != pred_img.shape: # Predictions are lower resolution than GT
                gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_AREA)
            # gt_img = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_AREA)
            # pred_img = cv2.resize(pred_img, (256, 256), interpolation=cv2.INTER_AREA)

            save_diff(gt_img, pred_img, f"Comparisons/{folder}/{file.replace('.', '_diff.')}")

def save_diff(gt_img, pred_img, save_path):
    diff_img = np.abs(gt_img.astype(np.float32) - pred_img.astype(np.float32))
    diff_img = np.clip(diff_img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))


def main_eval():
    if len(sys.argv) != 2:
        print("Usage: python eval.py <model_folder_name>")
        exit(1)
    folder = sys.argv[1]

    datasets_root = "./Datasets"  # replace with your datasets directory path

    print(f"Evaluating on model: {folder}")
    print("Results for SRD:")
    gt_folder = f"{datasets_root}/SRD/test/shadow_free"
    pred_folder = f"{folder}/SRD"
    suffix = "_free"
    evaluate_folder(gt_folder, pred_folder, suffix, "jpg")

    print("Results for ISTD+:")
    gt_folder = f"{datasets_root}/ISTD/test/test_C_fixed_official"
    pred_folder = f"{folder}/ISTD+"
    suffix = ""
    evaluate_folder(gt_folder, pred_folder, suffix, "png")

    print("Results for WSRD+:")
    gt_folder = f"{datasets_root}/WSRD+/val/gt"
    pred_folder = f"{folder}/WSRD+"
    suffix = ""
    evaluate_folder(gt_folder, pred_folder, suffix, "png")

    print("Results for INS:")
    gt_folder = f"{datasets_root}/INS/test/shadow_free"
    pred_folder = f"{folder}/INS"
    suffix = ""
    evaluate_folder(gt_folder, pred_folder, suffix, "png")



if __name__ == "__main__":
    main_eval()
    # main_diffs()
