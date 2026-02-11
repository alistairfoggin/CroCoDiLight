import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from matplotlib import pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lighting.relighting_modules import rescale_image
from lighting.relighting_model import load_relight_model, LightingMapper, RelightModule


def save_image(tensor, path):
    tensor = rescale_image(tensor).squeeze(0).cpu().detach()
    cv2.imwrite(path, cv2.cvtColor((tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def swap_lighting(croco_relight: RelightModule):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Replace with your own images
    img1 = transform(Image.open('./data/Input1.png').convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open('./data/Input2.png').convert('RGB')).unsqueeze(0).to(device)
    img_info = {'height': 448, 'width': 448}

    with torch.no_grad():
        img1_relit, img2_relit, static, static_pos, dyn, _ = croco_relight(img1, img2, do_tiling=False)

    save_image(img1_relit, "./data/out/compression/lighting3.png")
    save_image(img2_relit, "./data/out/compression/intrinsics3.png")


def apply_and_show(croco_relight, mapper_model, image_path, title='Mapped'):
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(448),
        # transforms.CenterCrop(448),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        mapped_img = croco_relight.apply_mapper(img, mapper_model)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(rescale_image(img).squeeze(0).cpu().permute(1, 2, 0))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(rescale_image(mapped_img).squeeze(0).cpu().permute(1, 2, 0))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    croco_relight = load_relight_model('pretrained_models/croco_relight_full.pth', device)
    croco_relight.eval()

    albedo_mapper = LightingMapper(
        patch_size=croco_relight.croco.enc_embed_dim, extractor_depth=8,
        rope=croco_relight.croco.rope).to(device)
    albedo_mapper.load_mapper(torch.load('pretrained_models/shadow_mapper.pth', 'cpu'))
    albedo_mapper.eval()

    # swap_lighting(croco_relight)
    # apply_and_show(croco_relight, albedo_mapper, os.path.expanduser('~/Downloads/test_kitchen_image.jpg'), title='Albedo')
    apply_and_show(croco_relight, albedo_mapper, os.path.expanduser('data/Input_SR_cropped.png'), title='Shadow Removal')