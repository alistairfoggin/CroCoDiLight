import os
import lpips
import torch
import torchvision.transforms.v2 as transforms
from torch import nn

import wandb
from torch.utils.data import ConcatDataset, DataLoader

from lighting.dataloader import HypersimDataset, CGIntrinsicDataset, DualDirectoryDataset, BigTimeDataset
from lighting.relighting_modules import img_mean, img_std
from lighting.relighting_model import load_relight_model, LightingMapper

base_root_dir = "./datasets/" # Replace with your datasets directory path


def train_mapper(mapper_type: str):
    epochs = 120 if mapper_type == "shadow" else 40
    lr = 5e-5
    batch_size = 32
    res = 448
    transform = transforms.Compose([
        transforms.RandomCrop((res, res), pad_if_needed=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    val_dataset = None
    if mapper_type == "albedo":
        root_dir_hypersim = base_root_dir + "ML-HyperSim/"
        root_dir_cgi = base_root_dir + "CGIntrinsics/"
        root_dir_mit_illumination_test = base_root_dir + "Multi_Illumination/test/"

        dataset_hypersim = HypersimDataset(root_dir_hypersim, split="train", transform=transform)
        dataset_cgi = CGIntrinsicDataset(root_dir_cgi, transform=transform)
        dataset = ConcatDataset([dataset_hypersim, dataset_cgi])

        val_hypersim = HypersimDataset(root_dir_hypersim, split="val", transform=transform)
        val_mit = BigTimeDataset(root_dir_mit_illumination_test, prefix="dir_", transform=transform, split="val")
        val_dataset = ConcatDataset([val_hypersim])

    elif mapper_type == "shadow":
        root_dir_srd = base_root_dir + "SRD/"
        root_dir_wsrd = base_root_dir + "WSRD+/"
        root_dir_istd = base_root_dir + "ISTD+/"
        dataset_srd = DualDirectoryDataset(root_dir_srd, "shadow", "shadow_free", transform=transform,
                                           split="train", suffix="_no_shadow")
        dataset_wsrd = DualDirectoryDataset(root_dir_wsrd, "input", "gt", transform=transform, split="train")
        dataset_istd_plus = DualDirectoryDataset(root_dir_istd, "train_A", "train_C_fixed_ours", transform=transform,
                                                 split="train")
        dataset = ConcatDataset([dataset_srd, dataset_istd_plus, dataset_wsrd])
        val_srd = DualDirectoryDataset(root_dir_srd, "shadow", "shadow_free", transform=transform, split="test",
                                       suffix="_free")
        val_istd = DualDirectoryDataset(root_dir_istd, "test_A", "test_C_fixed_official", transform=transform,
                                        split="test")
        val_dataset = ConcatDataset([val_srd, val_istd])

    else:
        raise ValueError("Unknown mapper types: " + mapper_type)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    croco_relight = load_relight_model('pretrained_models/CroCoDiLight.pth', device)
    # Freeze all components - only the mapper will be trained
    croco_relight.freeze_components(encoder=True, decoder=True, extractor=True, entangler=True)

    mapper_model = LightingMapper(patch_size=croco_relight.croco.enc_embed_dim, extractor_depth=8, rope=croco_relight.croco.rope).to(device)
    # Load the entangler weights as initialization
    if not os.path.exists('pretrained_models/CroCoDiLight_entangler_init.pth'):
        torch.save(croco_relight.lighting_entangler.state_dict(), 'pretrained_models/CroCoDiLight_entangler_init.pth')
    entangler_ckpt = torch.load('pretrained_models/CroCoDiLight_entangler_init.pth', 'cpu')
    mapper_model.load_mapper(entangler_ckpt)
    mapper_optim = torch.optim.Adam(mapper_model.latent_mapper.parameters(), lr=lr)

    run = wandb.init(
        entity="your-wandb-entity",  # replace with your wandb entity
        project=f"CroCoDiLight-{mapper_type}-mapper-train",
        config={"epochs": epochs, "learning_rate": lr, "batch_size": batch_size},
        notes="Train albedo estimation" if mapper_type == "albedo" else "Train shadow removal with INS",
    )
    img_info = {'height': res, 'width': res}

    mse_loss = nn.MSELoss()
    img_loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn = lambda pred, gt: 0.5 * img_loss_fn(pred, gt).mean() + 0.5 * mse_loss(pred, gt)

    step = -1
    for epoch in range(epochs):
        for (i, batch) in enumerate(dataloader):
            step += 1
            batch = batch.to(device)

            input_img, gt_img = batch[:, 0], batch[:, 1]

            mapper_optim.zero_grad()

            with torch.no_grad():
                feat, pos = croco_relight.croco.encode_image_pairs(input_img, gt_img, False)
                static, dyn, dyn_pos = croco_relight.lighting_extractor(feat, pos)
                static, _ = torch.chunk(static, 2, dim=0)
                dyn, dyn_gt = torch.chunk(dyn, 2, dim=0)
                pos, _ = torch.chunk(pos, 2, dim=0)

            pred_dyn = mapper_model(static, pos, dyn, None)

            dyn_loss = mse_loss(pred_dyn, dyn_gt)
            with torch.no_grad():
                pred_latents = croco_relight.lighting_entangler(static, pos, pred_dyn)
                pred_img = croco_relight.croco.decode(pred_latents, pos, img_info)
                img_loss = loss_fn(pred_img, gt_img)
            loss = dyn_loss #+ img_loss

            loss.backward()
            mapper_optim.step()

            if i % 20 == 0:
                run.log({"dyn_loss": dyn_loss.item(), "img_loss": img_loss.item(), "loss": loss.item()}, step=step)
                print(f"Epoch {epoch}, iteration {i}: Loss: {loss.item()}")

            if i % 50 == 0:
                out_img = torch.zeros((3, res, res * 3))
                # 0 0: img input
                out_img[:, :, :res] = input_img[0].detach()
                # 0 1: img pred
                out_img[:, :, res:res * 2] = pred_img[0].detach()
                # 0 2: img gt
                out_img[:, :, res * 2:] = gt_img[0].detach()
                out_img = out_img * torch.tensor(img_std).reshape((-1, 1, 1)) + torch.tensor(img_mean).reshape(
                    (-1, 1, 1))
                wandb_img = wandb.Image(out_img.clamp(0.0, 1.0),
                                        caption="Left: input, Middle: prediction, Right: ground truth")
                run.log({"train_images": wandb_img}, step=step)
        # Validation loop
        if mapper_type == "albedo":
            for i, imgs in enumerate(val_mit):
                if i % 10 != 0:
                    continue
                with torch.no_grad():
                    imgs = imgs.to(device).unsqueeze(0)
                    input_img = imgs[:, 0]
                    feat, pos, _ = croco_relight.croco._encode_image(input_img, False, False)
                    static, dyn, dyn_pos = croco_relight.lighting_extractor(feat, pos)
                    pred_dyn = mapper_model(static, pos, dyn, None)

                    pred_latents = croco_relight.lighting_entangler(static, pos, pred_dyn)
                    pred_img = croco_relight.croco.decode(pred_latents, pos, img_info)

                out_img = torch.zeros((3, res, res * 2))
                # 0 0: img input
                out_img[:, :, :res] = input_img[0].detach()
                # 0 1: img pred
                out_img[:, :, res:res * 2] = pred_img[0].detach()
                out_img = out_img * torch.tensor(img_std).reshape((-1, 1, 1)) + torch.tensor(img_mean).reshape(
                    (-1, 1, 1))
                wandb_img = wandb.Image(out_img.clamp(0.0, 1.0),
                                        caption="Left: input, Middle: prediction, Right: ground truth")
                run.log({"val_images": wandb_img})

        if val_dataset is not None:
            val_img_losses = []
            val_dyn_losses = []
            for i, imgs in enumerate(val_dataset):
                with torch.no_grad():
                    imgs = imgs.to(device).unsqueeze(0)
                    input_img, gt_img = imgs[:, 0], imgs[:, 1]

                    feat, pos = croco_relight.croco.encode_image_pairs(input_img, gt_img, False)
                    static, dyn, dyn_pos = croco_relight.lighting_extractor(feat, pos)

                    static, _ = torch.chunk(static, 2, dim=0)
                    dyn, dyn_gt = torch.chunk(dyn, 2, dim=0)
                    pos, _ = torch.chunk(pos, 2, dim=0)

                    pred_dyn = mapper_model(static, pos, dyn, None)

                    pred_latents = croco_relight.lighting_entangler(static, pos, pred_dyn)
                    pred_img = croco_relight.croco.decode(pred_latents, pos, img_info)
                    val_img_loss = loss_fn(pred_img, gt_img)
                    val_dyn_loss = mse_loss(pred_dyn, dyn_gt)
                    val_img_losses.append(val_img_loss.item())
                    val_dyn_losses.append(val_dyn_loss.item())

                if i % 10 == 0:
                    out_img = torch.zeros((3, res, res * 3))
                    # 0 0: img input
                    out_img[:, :, :res] = input_img[0].detach()
                    # 0 1: img pred
                    out_img[:, :, res:res * 2] = pred_img[0].detach()
                    # 0 2: img gt
                    out_img[:, :, res * 2:] = gt_img[0].detach()
                    out_img = out_img * torch.tensor(img_std).reshape((-1, 1, 1)) + torch.tensor(img_mean).reshape(
                        (-1, 1, 1))
                    wandb_img = wandb.Image(out_img.clamp(0.0, 1.0),
                                            caption="Left: input, Middle: prediction, Right: ground truth")
                    run.log({"val_images": wandb_img})
                if i == 50:
                    break
            mean_val_img_loss = sum(val_img_losses) / len(val_img_losses)
            mean_val_dyn_loss = sum(val_dyn_losses) / len(val_dyn_losses)
            run.log({"mean_val_img_loss": mean_val_img_loss, "mean_val_dyn_loss": mean_val_dyn_loss})

        if epoch % 10 == 0:
            torch.save(mapper_model.state_dict(), f'pretrained_models/CroCoDiLight_{mapper_type}_mapper.pth')

    torch.save(mapper_model.state_dict(), f'pretrained_models/CroCoDiLight_{mapper_type}_mapper.pth')
    run.finish()


if __name__ == '__main__':
    train_mapper("shadow") # Either "albedo" or "shadow" depending on which mapper you are training
