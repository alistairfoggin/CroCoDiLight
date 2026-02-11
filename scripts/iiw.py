import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluations import whdr
from lighting.relighting_model import load_relight_model, LightingMapper

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')

    croco_relight = load_relight_model('pretrained_models/croco_relight_full.pth', device)
    croco_relight.eval()
    albedo_mapper = LightingMapper(patch_size=croco_relight.croco.enc_embed_dim, extractor_depth=8, rope=croco_relight.croco.rope).to(device)
    albedo_mapper.load_mapper(torch.load('pretrained_models/albedo_mapper.pth', 'cpu'))
    albedo_mapper.eval()
    root_dir = "./data/iiw/data/"  # replace with your directory path
    filenames = [file for file in os.listdir(root_dir) if file.endswith(".png") and not "_ref" in file]
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    whdr_measurements = {}
    with torch.no_grad():
        for filename in filenames:
            path = os.path.join(root_dir, filename)
            print(path)
            pil_img = Image.open(path).convert('RGB')
            img = transform(pil_img).unsqueeze(0).to(device)
            H = W = 448
            _, _, H1, W1 = img.shape
            img_info = {'height': H, 'width': W}
            resized = False
            if H1 < H or W1 < W:
                new_img = torch.zeros((1, 3, max(H1, H), max(W1, W)), device=device)
                new_img[:, :, :H1, :W1] = img
                img = new_img
                resized = True

            delit_img = croco_relight.apply_mapper(img, albedo_mapper, use_consistency=False)

            if resized:
                delit_img = delit_img[:, :, :H1, :W1]
            out_img = delit_img[0].detach().cpu().permute(1, 2, 0).numpy()
            out_img = out_img * np.array(img_std).reshape((1, 1, -1)) + np.array(img_mean).reshape((1, 1, -1))
            plt.imsave(path.replace(".png", "_ref.png"), np.clip(out_img, 0, 1))

            reflectance = whdr.load_image(filename=path.replace(".png", "_ref.png"), is_srgb=True)
            judgements = json.load(open(path.replace(".png", ".json")))
            img_whdr = whdr.compute_whdr(reflectance, judgements, 0.1)
            print("saved. WHDR:", img_whdr)
            whdr_measurements[filename] = img_whdr

    with open("whdr_results_albedo_mapper_all.json", "w") as f:
        json.dump(whdr_measurements, f, indent=4)
    average_whdr = np.mean(np.array(list(whdr_measurements.values())))
    print("Average WHDR:", average_whdr)
