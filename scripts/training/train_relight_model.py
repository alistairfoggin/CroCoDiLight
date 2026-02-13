import lpips
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

import torchvision.transforms.v2 as transforms

from crocodilight.dataloader import BigTimeDataset, DualDirectoryDataset, HypersimDataset, CGIntrinsicDataset, \
    ScenePairDataset
from crocodilight.relighting_modules import img_mean, img_std
from crocodilight.relighting_model import CroCoDecode, RelightModule

if __name__ == "__main__":
    epochs = 10
    lr = 5e-5
    batch_size = 8

    run = wandb.init(
        entity="your-wandb-entity",  # replace with your wandb entity
        project="crocodilight-train",
        config={"epochs": epochs, "learning_rate": lr, "batch_size": batch_size},
        notes="bigtime+mit_illumination+srd+wsrd+istd+hypersim+cgi 448x448 LPIPS+MSE",
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    ckpt = torch.load('pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth', 'cpu')
    croco_kwargs = ckpt.get('croco_kwargs', {})
    croco_kwargs['img_size'] = 448
    croco_decode = CroCoDecode(**croco_kwargs).to(device)
    croco_decode.load_state_dict(ckpt['model'])
    croco_decode.setup()
    decode_ckpt = torch.load('pretrained_models/CroCoDiLight_decoder.pth', 'cpu')
    croco_decode.load_state_dict(decode_ckpt)
    croco_relight = RelightModule(croco_decode).to(device)
    # Freeze encoder and pretrained decoder, train only lighting extractor + entangler
    croco_relight.freeze_components(encoder=True, decoder=True, extractor=False, entangler=False)
    croco_optim = torch.optim.Adam(croco_relight.parameters(), lr=lr)

    base_root_dir = "./datasets/" # replace with your dataset directory path
    root_dir_bigtime = base_root_dir + "BigTime/phoenix/S6/zl548/AMOS/BigTime_v1/"
    root_dir_mit_illumination = base_root_dir + "Multi_Illumination/train/"
    root_dir_mit_illumination_test = base_root_dir + "Multi_Illumination/test/"

    root_dir_srd = base_root_dir + "SRD/"
    root_dir_wsrd = base_root_dir + "WSRD+/"
    root_dir_istd = base_root_dir + "ISTD+/"

    root_dir_hypersim = base_root_dir + "ML-HyperSim/"
    root_dir_cgi = base_root_dir + "CGIntrinsics/"

    res = 448
    transform = transforms.Compose([
        transforms.RandomCrop((res, res), pad_if_needed=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    train_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.05),
        transform,
    ])

    # Timelapse/multiple illumination
    dataset_bigtime = ScenePairDataset(root_dir_bigtime, internal_folder="00", transform=train_transform)
    dataset_mit = ScenePairDataset(root_dir_mit_illumination, prefix="dir_", transform=train_transform)

    # Shadow Removal
    dataset_srd = DualDirectoryDataset(root_dir_srd, "shadow", "shadow_free", transform=train_transform,
                                       split="train", suffix="_no_shadow")
    dataset_wsrd = DualDirectoryDataset(root_dir_wsrd, "input", "gt", transform=train_transform, split="train")
    dataset_istd_plus = DualDirectoryDataset(root_dir_istd, "train_A", "train_C_fixed_ours", transform=train_transform,
                                             split="train")
    # Albedo
    dataset_hypersim = HypersimDataset(root_dir_hypersim, split="train", transform=train_transform)
    dataset_cgi = CGIntrinsicDataset(root_dir_cgi, transform=train_transform)

    dataset = ConcatDataset([dataset_bigtime, dataset_mit, dataset_srd, dataset_wsrd, dataset_istd_plus, dataset_hypersim, dataset_cgi])
    print("Number of image pairs:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_mit = BigTimeDataset(root_dir_mit_illumination_test, prefix="dir_", transform=transform, split=None)

    val_wsrd = DualDirectoryDataset(root_dir_wsrd, "input", "gt", transform=transform, split="val")
    val_hypersim = HypersimDataset(root_dir_hypersim, split="val", transform=transform)

    val_dataset = ConcatDataset([val_mit, val_wsrd, val_hypersim])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    mse_loss = nn.MSELoss()
    img_loss_fn = lpips.LPIPS(net='alex')
    img_loss_fn.to(device)
    loss_fn = lambda pred, gt: 0.5 * img_loss_fn(pred, gt).mean() + 0.5 * mse_loss(pred, gt)

    step = -1
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            step += 1
            batch = batch.to(device)
            img1, img2 = batch[:, 0], batch[:, 1]

            croco_optim.zero_grad()

            img1_relit, img2_relit, static, static_pos, dyn, _ = croco_relight(img1, img2, do_tiling=False)
            static1, static2 = static.chunk(2, dim=0)
            dyn1, dyn2 = dyn.chunk(2, dim=0)

            with torch.no_grad():
                zero_dyn = torch.zeros_like(dyn)
                latents = croco_relight.lighting_entangler(static, static_pos, zero_dyn)
                _, _, H1, W1 = img1.size()
                img_info = {'height': H1, 'width': W1}
                img_delit = croco_decode.decode(latents, static_pos, img_info)
                img1_delit, img2_delit = img_delit.chunk(2, dim=0)
                loss_delight = loss_fn(img1_delit, img2_delit)
                loss_static_latents = mse_loss(static1, static2)

            loss_relight = loss_fn(img1_relit, img2) + loss_fn(img2_relit, img1)

            loss = loss_relight + 0.3 * loss_static_latents
            loss.backward()
            croco_optim.step()

            if step % 20 == 0:
                run.log(data={"loss": loss.item(), "relight_loss": loss_relight.item() / 2,
                              "latent_loss": loss_static_latents.item(), "delight_loss": loss_delight.item()},
                        step=step)
                print(
                    f"Epoch {epoch}, iteration {i}, Total loss: {loss.item()}, Relighting: {loss_relight.item() / 2}, Intrinsic: {loss_static_latents.item()}")

            if step % 60 == 0:
                out_img = torch.zeros((3, res * 2, res * 3))
                # 0 0: img1 gt
                out_img[:, :res, :res] = img1[0].detach()
                # 0 1: img2 relit to match img1
                out_img[:, :res, res:res * 2] = img2_relit[0].detach()
                # 0 2: img1 delit
                out_img[:, :res, res * 2:] = img1_delit[0].detach()
                # 1 0: img2 gt
                out_img[:, res:, :res] = img2[0].detach()
                # 1 1: img1 relit to match img2
                out_img[:, res:, res:res * 2] = img1_relit[0].detach()
                # 1 2: img2 delit
                out_img[:, res:, res * 2:] = img2_delit[0].detach()
                out_img = out_img * torch.tensor(img_std).reshape((-1, 1, 1)) + torch.tensor(img_mean).reshape(
                    (-1, 1, 1))
                wandb_img = wandb.Image(out_img.clamp(0.0, 1.0),
                                        caption="Left: Ground Truth, Middle left: Reconstructed images, Middle right: Relit images, Right: Delit images")
                run.log({"train_images": wandb_img}, step=step)
            if i == 0:
                torch.save({'croco_kwargs': croco_kwargs, 'model': croco_relight.state_dict()},
                           "pretrained_models/CroCoDiLight.pth")

        # Validation
        val_relight_losses = []
        val_recon_losses = []
        val_delight_losses = []
        val_intrinsic_losses = []
        n_val_batches = len(val_dataset) // batch_size
        n_step = n_val_batches // 6
        val_step = -1
        for i, batch in enumerate(val_dataloader):
            batch = batch.to(device)

            img1, img2 = batch[:, 0], batch[:, 1]

            with torch.no_grad():
                img1_relit, img2_relit, static, static_pos, dyn, _ = croco_relight(img1, img2)
                static1, static2 = static.chunk(2, dim=0)

                zero_dyn = torch.zeros_like(dyn)
                latents = croco_relight.lighting_entangler(static, static_pos, zero_dyn)
                recon_latents = croco_relight.lighting_entangler(static, static_pos, dyn)
                _, _, H1, W1 = img1.size()
                img_info = {'height': H1, 'width': W1}
                img_delit = croco_decode.decode(latents, static_pos, img_info)
                img1_delit, img2_delit = img_delit.chunk(2, dim=0)
                img_recon = croco_decode.decode(recon_latents, static_pos, img_info)
                img1_recon, img2_recon = img_recon.chunk(2, dim=0)

                loss_val_relight = loss_fn(img1_relit, img2) + loss_fn(img2_relit, img1)
                loss_val_recon = loss_fn(img1_recon, img1) + loss_fn(img2_recon, img2)
                loss_delight = loss_fn(img1_delit, img2_delit)
                loss_static_latents = mse_loss(static1, static2)
                val_relight_losses.append(loss_val_relight.item() / 2)
                val_recon_losses.append(loss_val_recon.item() / 2)
                val_delight_losses.append(loss_delight.item())
                val_intrinsic_losses.append(loss_static_latents.item())

            if i % n_step == 0:
                val_step += 1
                out_img = torch.zeros((3, res * 2, res * 4))
                # 0 0: img1 gt
                out_img[:, :res, :res] = img1[0].detach()
                # 0 1: img1 recon
                out_img[:, :res, res:res * 2] = img1_recon[0].detach()
                # 0 2: img2 relit to match img1
                out_img[:, :res, res * 2:res * 3] = img2_relit[0].detach()
                # 0 3: img1 delit
                out_img[:, :res, res * 3:] = img1_delit[0].detach()
                # 1 0: img2 gt
                out_img[:, res:, :res] = img2[0].detach()
                # 1 1: img2 recon
                out_img[:, res:, res:res * 2] = img2_recon[0].detach()
                # 1 2: img1 relit to match img2
                out_img[:, res:, res * 2:res * 3] = img1_relit[0].detach()
                # 1 3: img2 delit
                out_img[:, res:, res * 3:] = img2_delit[0].detach()
                out_img = out_img * torch.tensor(img_std).reshape((-1, 1, 1)) + torch.tensor(img_mean).reshape(
                    (-1, 1, 1))
                wandb_img = wandb.Image(out_img.clamp(0.0, 1.0),
                                        caption="Left: Ground Truth, Middle left: Reconstructed images, Middle right: Relit images, Right: Delit images")
                run.log({f"val_images_{val_step}": wandb_img}, step=step)

        val_relight_loss = torch.mean(torch.tensor(val_relight_losses)).item()
        val_recon_loss = torch.mean(torch.tensor(val_recon_losses)).item()
        val_delight_loss = torch.mean(torch.tensor(val_delight_losses)).item()
        val_intrinsic_loss = torch.mean(torch.tensor(val_intrinsic_losses)).item()
        run.log(data={"val_relight_loss": val_relight_loss, "val_delight_loss": val_delight_loss,
                      "val_recon_loss": val_recon_loss, "val_intrinsic_loss": val_intrinsic_loss}, step=step)
        print(
            f"Epoch {epoch}, Validation relighting loss: {val_relight_loss}, Validation delight loss: {val_delight_loss}")

    torch.save({'croco_kwargs': croco_kwargs, 'model': croco_relight.state_dict()},
               "pretrained_models/CroCoDiLight.pth")
    run.finish()
