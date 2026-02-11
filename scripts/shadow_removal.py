import os

import cv2
import torch
from PIL import Image

import torchvision.transforms.v2 as transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lighting.relighting_modules import img_mean, img_std, rescale_image
from lighting.relighting_model import load_relight_model, LightingMapper


def folder_remove_shadows(input_folder, output_folder, model: RelightModule, mapper: LightingMapper, device):
    files = [f for f in os.listdir(input_folder) if
             os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for file in files:
            # gt_file = file.replace('.jpg', '_free.jpg') if file.endswith('.jpg') else file
            # gt_path = os.path.join(gt_folder, gt_file)
            img_path = os.path.join(input_folder, file)
            img = Image.open(img_path).convert("RGB")
            # gt_img = Image.open(gt_path).convert("RGB")
            transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=img_mean, std=img_std),
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
            # gt_img_tensor = transform(gt_img).unsqueeze(0).to(device)

            deshadowed_img = model.apply_mapper(img_tensor, mapper, use_consistency=False)
            # deshadowed_img = model.apply_gt_mapper(img_tensor, gt_img_tensor)
            deshadowed_img_np = (rescale_image(deshadowed_img)[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                'uint8')
            deshadowed_img_np = cv2.cvtColor(deshadowed_img_np, cv2.COLOR_RGB2BGR)

            out_file = file.replace('.jpg', '.png') if file.endswith('.jpg') else file
            out_path = os.path.join(output_folder, out_file)
            cv2.imwrite(out_path, deshadowed_img_np)
            print(f"Processed {file}, saved to {out_path}")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    croco_relight = load_relight_model('pretrained_models/croco_relight_full.pth', device)
    croco_relight.eval()
    shadow_mapper = LightingMapper(patch_size=croco_relight.croco.enc_embed_dim, extractor_depth=8,
                                   rope=croco_relight.croco.rope).to(device)
    # Load the entangler weights
    shadow_mapper.load_mapper(torch.load('pretrained_models/shadow_mapper.pth', 'cpu'))
    shadow_mapper.eval()

    base_root_dir = "./datasets/relighting/" # Replace with your dataset base directory
    root_dir_srd = base_root_dir + "srd/SRD/"
    root_dir_wsrd = base_root_dir + "wsrd/"
    root_dir_istd = base_root_dir + "ISTD_Dataset/"

    # SRD
    folder_remove_shadows("./datasets/relighting/SRD/test/shadow/",
                          "./data/sr_outputs/SRD/",
                          croco_relight, shadow_mapper, device)
    # ISTD+
    folder_remove_shadows("./datasets/relighting/ISTD/test/test_A/",
                          "./data/sr_outputs/ISTD+/",
                          croco_relight, shadow_mapper, device)

    # WSRD+
    folder_remove_shadows(("./datasets/relighting/WSRD+/val/input/"),
                          "./data/sr_outputs/WSRD+/",
                          croco_relight, shadow_mapper, device)

    # INS
    folder_remove_shadows("./datasets/relighting/INS/test/origin/",
                          "./data/sr_outputs/INS/",
                          croco_relight, shadow_mapper, device)
