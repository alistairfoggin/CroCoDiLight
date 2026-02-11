import torch
from pathlib import Path
from PIL import Image
import cv2
from torchvision.transforms import v2 as transforms

from blended_tiling import TilingModule

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lighting.relighting_modules import rescale_image
from lighting.relighting_model import load_relight_model

transform = transforms.Compose([
    transforms.ToImage(),
    # transforms.Resize(448),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

img_info = {"height": 448, "width": 448}
tile_size = 448

def extract_dyn(img_path, resize=None):
    img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
    if resize is not None:
        img = transforms.Resize(resize)(img)
    tiling_module = TilingModule(tile_size=tile_size, tile_overlap=0.2, base_size=img.shape[2:])
    img = tiling_module.split_into_tiles(img).to(device)
    with torch.no_grad():
        feat, pos, _ = croco_relight.croco._encode_image(img, do_mask=False, return_all_blocks=False)
        static, dyn, dyn_pos = croco_relight.lighting_extractor(feat, pos)
    return static, dyn, pos, tiling_module

def main_interpolate_two_frames(img_dir: Path, out_dir: Path):
    frame_a = img_dir / "frame_000300.png"
    frame_b = img_dir / "frame_000600.png"
    static_a, dyn_a, pos_a, tiling_module_a = extract_dyn(frame_a, resize=1024)
    static_b, dyn_b, pos_b, tiling_module_b = extract_dyn(frame_b, resize=1024)

    steps = 5

    img_a = transform(Image.open(frame_a).convert('RGB')).unsqueeze(0).to(device)
    img_b = transform(Image.open(frame_b).convert('RGB')).unsqueeze(0).to(device)

    for idx, alpha in enumerate(torch.linspace(0, 1, steps)):
        dyn_interp = dyn_a * (1 - alpha) + dyn_b * alpha
        img_interp = img_a * (1 - alpha) + img_b * alpha

        with torch.no_grad():
            feat_interp = croco_relight.lighting_entangler(static_a, pos_a, dyn_interp)
            out_img = croco_relight.croco.decode(feat_interp, pos_a, img_info)
            out_img = tiling_module_a.rebuild_with_masks(out_img)
            out_img = rescale_image(out_img)
            out_img = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            cv2.imwrite(str(out_dir / f"dyn_interp_{idx:02d}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

            out_img = rescale_image(img_interp)
            out_img = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            cv2.imwrite(str(out_dir / f"img_interp_{idx:02d}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))



    print(f"Saved {steps} images using lerp interpolation")

def freeze_lighting(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(in_dir.glob("frame_*.png"))
    static_ref, dyn_ref, pos_ref, tiling_module_ref = extract_dyn(frame_files[25])

    for frame_file in frame_files[5::5]:
        static, dyn, pos, tiling_module = extract_dyn(frame_file)
        with torch.no_grad():
            feat = croco_relight.lighting_entangler(static, pos, dyn_ref)
            out_img = croco_relight.croco.decode(feat, pos, img_info)
            out_img = tiling_module.rebuild_with_masks(out_img)
            out_img = rescale_image(out_img)
            out_img = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            cv2.imwrite(str(out_dir / str(frame_file.name).replace(".png", "_relit.png")), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print(f"Processed {frame_file.name}")


def transfer_lighting(in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(in_dir.glob("frame_*.png"))
    static_ref, dyn_ref, pos_ref, tiling_module_ref = extract_dyn(frame_files[10])

    for frame_file in frame_files[5::5]:
        static, dyn, pos, tiling_module = extract_dyn(frame_file)
        with torch.no_grad():
            feat = croco_relight.lighting_entangler(static_ref, pos_ref, dyn)
            out_img = croco_relight.croco.decode(feat, pos, img_info)
            out_img = tiling_module.rebuild_with_masks(out_img)
            out_img = rescale_image(out_img)
            out_img = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            cv2.imwrite(str(out_dir / frame_file.name), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        print(f"Processed {frame_file.name}")

def main_interpolate_alternating_frames(out_dyn_dir, out_img_dir):
    # Interpolate between multiple frames in an alternating fashion
    frame_files = sorted(img_dir.glob("frame_*.jpg"))
    steps = 5
    static, dyn, pos, tiling_module = extract_dyn(frame_files[0])

    for i in range(0, len(frame_files) - steps, steps):
        static_a, dyn_a, pos_a, tiling_module_a = extract_dyn(frame_files[i])
        static_b, dyn_b, pos_b, tiling_module_b = extract_dyn(frame_files[i + steps])
        frame_a = transform(Image.open(frame_files[i]).convert('RGB')).unsqueeze(0).to(device)
        frame_b = transform(Image.open(frame_files[i + steps]).convert('RGB')).unsqueeze(0).to(device)
        for idx, alpha in enumerate(torch.linspace(0, 1, steps+1)):
            if alpha == 1.0:
                break
            print(f"Interpolating between frames {i} and {i+steps}, step {idx+1}/{steps}")
            dyn_interp = dyn_a * (1 - alpha) + dyn_b * alpha
            frame_interp = frame_a * (1 - alpha) + frame_b * alpha

            with torch.no_grad():
                feat_interp = croco_relight.lighting_entangler(static_a, pos_a, dyn_interp)
                out_img = croco_relight.croco.decode(feat_interp, pos, img_info)
                out_img = tiling_module.rebuild_with_masks(out_img)

                out_img = rescale_image(out_img)
                out_img = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                cv2.imwrite(str(out_dyn_dir / f"interp_{i}_{idx:02d}.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

                out_gt_img = rescale_image(frame_interp)
                out_gt_img = (out_gt_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
                cv2.imwrite(str(out_img_dir / f"interp_{i}_{idx:02d}.png"), cv2.cvtColor(out_gt_img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    croco_relight = load_relight_model('pretrained_models/croco_relight_full.pth', device)
    croco_relight.eval()

    freeze_lighting("./data/timelapse/src", "./data/freeze_lighting/timelapse/")



