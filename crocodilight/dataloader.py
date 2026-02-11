import json
import os
import random
from pathlib import Path
import pickle

import polars as pl
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


class DualDirectoryDataset(Dataset):
    def __init__(self, root_dir: str | Path, dir_a: str | Path, dir_b: str | Path, transform=None, split="train",
                 suffix=""):
        root_dir = Path(root_dir)
        if os.path.exists(root_dir / "train_test_split.json"):
            with open(root_dir / "train_test_split.json", "r") as f:
                train_test_split = json.load(f)
            self.img_names = train_test_split[split]
            if self.img_names is None:
                raise ValueError("Invalid Split '{}'".format(split))
            self.dir_a = root_dir / dir_a
            self.dir_b = root_dir / dir_b
        else:
            root_dir = root_dir / split
            self.dir_a = root_dir / dir_a
            self.dir_b = root_dir / dir_b
            self.img_names = [name for name in os.listdir(self.dir_a) if name.endswith((".png", ".jpg"))]

        self.transform = transform
        self.suffix = suffix

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_a_path = self.dir_a / img_name
        if self.suffix != "":
            if img_name.endswith(".png"):
                img_b_name = img_name.replace(".png", self.suffix + ".png")
            elif img_name.endswith(".jpg"):
                img_b_name = img_name.replace(".jpg", self.suffix + ".jpg")
            else:
                img_b_name = img_name
        else:
            img_b_name = img_name
        img_b_path = self.dir_b / img_b_name
        img_a = transforms.functional.to_image(Image.open(img_a_path).convert('RGB'))
        img_b = transforms.functional.to_image(Image.open(img_b_path).convert('RGB'))
        imgs = torch.stack([img_a, img_b])
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs


class CGIntrinsicDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.train_list = pickle.load(open(self.root_dir / "train_list/img_batch.p", "rb"))
        self.render_list = pickle.load(open(self.root_dir / "render_list/img_batch.p", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.train_list) + len(self.render_list)

    def __getitem__(self, idx):
        if idx >= len(self.train_list):
            idx -= len(self.train_list)
            is_train_list = False
        else:
            is_train_list = True

        if is_train_list:
            img_name = self.train_list[idx]
            img_path = self.root_dir / "images" / img_name
            albedo_name = img_name.replace(".png", "_albedo.png")
            albedo_path = self.root_dir / "images" / albedo_name
        else:
            img_name = self.render_list[idx]
            if "_" in img_name:
                ending = img_name.split("_")[1]
                albedo_name = img_name.replace(ending, "albedo.png")
            else:
                if ".png" in img_name:
                    albedo_name = img_name.replace(".png", "_albedo.png")
                else:
                    albedo_name = img_name.replace(".jpg", "_albedo.png")
            img_path = self.root_dir / "rendered/images" / img_name
            albedo_path = self.root_dir / "rendered/albedo" / albedo_name

        img = transforms.functional.to_image(Image.open(img_path).convert('RGB'))
        albedo = transforms.functional.to_image(Image.open(albedo_path).convert('RGB'))
        img[albedo == 0] = 0  # mask out invalid albedo regions
        imgs = torch.stack([img, albedo])
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs


class HypersimDataset(Dataset):
    def __init__(self, root_dir: str | Path, split="train", transform=None):
        self.root_dir = Path(root_dir)
        split_df = pl.read_csv(self.root_dir / "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv")
        split_df = split_df.filter(pl.col("split_partition_name").eq(split) &
                                   pl.col("included_in_public_release").eq(True) &
                                   pl.col("frame_id").eq(0))
        self.split_df = split_df
        self.transform = transform

    def __len__(self):
        return len(self.split_df)

    def __getitem__(self, idx):
        row = self.split_df.row(idx, named=True)
        path = self.root_dir / f"data/{row['scene_name']}/images/scene_{row['camera_name']}_final_preview"
        color_img = transforms.functional.to_image(
            Image.open(path / f"frame.{row['frame_id']:04d}.color.jpg").convert('RGB'))
        albedo_img = transforms.functional.to_image(
            Image.open(path / f"frame.{row['frame_id']:04d}.diffuse_reflectance.jpg").convert('RGB'))
        color_img[albedo_img == 0] = 0  # mask out invalid albedo regions
        imgs = torch.stack([color_img, albedo_img])
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs


class ScenePairDataset(Dataset):
    def __init__(self, root_dir: str | Path, internal_folder=None, prefix=None, transform=None):
        self.root_dir = Path(root_dir)
        self.subdirectories = [folder for folder in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / folder)]
        self.internal_folder = internal_folder
        self.prefix = prefix
        self.transform = transform

        self.images = []
        for subdirectory in self.subdirectories:
            if self.internal_folder is not None:
                imgs_path = self.root_dir / subdirectory / self.internal_folder
            else:
                imgs_path = self.root_dir / subdirectory

            for img in os.listdir(imgs_path):
                if not self.check_filename(img):
                    continue

                self.images.append((imgs_path, img))

    def check_filename(self, filename):
        if self.prefix is not None and not filename.startswith(self.prefix):
            return False
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, img_name = self.images[idx]
        other_imgs = [name for name in os.listdir(img_path) if self.check_filename(name) and name != img_name]
        if len(other_imgs) == 0:
            raise FileNotFoundError(f"No image pair found for {img_path / img_name}")

        other_img_name = random.choice(other_imgs)

        img1 = transforms.functional.to_image(Image.open(img_path / img_name).convert('RGB'))
        img2 = transforms.functional.to_image(Image.open(img_path / other_img_name).convert('RGB'))
        out = torch.stack([img1, img2])  # stack the pair of images
        if self.transform:
            out = self.transform(out)
        return out


class BigTimeDataset(Dataset):
    def __init__(self, root_dir, internal_folder=None, prefix=None, post_process=None,
                 filetypes=('.png', '.jpg', '.jpeg', '.tif'), transform=None, split="train"):
        """
        Args:
            root_dir (string): Directory with all the subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        if os.path.exists((self.root_dir / "train_test_split.json")) and split is not None:
            with open(self.root_dir / "train_test_split.json", "r") as f:
                train_test_split = json.load(f)
            self.subdirectories = train_test_split[split]
            if self.subdirectories is None:
                raise ValueError("Invalid Split '{}'".format(split))
        else:
            self.subdirectories = [folder for folder in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / folder)]
        self.transform = transform
        self.internal_folder = internal_folder
        self.prefix = prefix
        self.filetypes = filetypes
        self.post_process = post_process

    def get_idx(self, dir_name):
        return self.subdirectories.index(dir_name)

    def __len__(self):
        return len(self.subdirectories)

    def check_filename(self, filename):
        if self.prefix is not None and not filename.startswith(self.prefix):
            return False
        return filename.lower().endswith(self.filetypes)

    def __getitem__(self, idx):
        subdir_name = self.subdirectories[idx]

        if self.internal_folder is not None:
            subdir_path = os.path.join(self.root_dir, subdir_name, self.internal_folder)
        else:
            subdir_path = os.path.join(self.root_dir, subdir_name)
        image_files = [f for f in os.listdir(subdir_path) if self.check_filename(f)]

        if len(image_files) < 2:
            raise ValueError(f"Subdirectory {subdir_name} does not contain at least two images.")

        selected_images = random.sample(image_files, 2)
        pair_images = []
        for img_file in selected_images:
            img_path = os.path.join(subdir_path, img_file)
            image = transforms.functional.to_image(Image.open(img_path).convert("RGB"))
            pair_images.append(image)
        out = torch.stack(pair_images)  # stack the pair of images
        if self.transform:
            out = self.transform(out)
        if self.post_process is not None:
            out = self.post_process(out)
        return out

