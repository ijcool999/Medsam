# finetune_drusen.py
# -*- coding: utf-8 -*-
"""
Fine-tune MedSAM on OCT drusen data.
Train image encoder + mask decoder; freeze prompt encoder.
Uses full-image box for “segment all drusen” supervision.
"""

import os
import shutil
import argparse
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from segment_anything import sam_model_registry


# ─── Dataset ──────────────────────────────────────────────────────────────────
class NpyDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_paths = sorted(
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith("_bscan.npy")
        )
        self.mask_paths = sorted(
            os.path.join(mask_folder, f)
            for f in os.listdir(mask_folder)
            if f.endswith("_mask.npy")
        )
        assert len(self.image_paths) == len(self.mask_paths), \
            "Mismatch between images and masks count"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32) / 255.0
        msk = np.load(self.mask_paths[idx]).astype(np.float32) / 255.0
        # add channel
        img = torch.from_numpy(img[None]).float()  # (1,H,W)
        msk = torch.from_numpy(msk[None]).float()  # (1,H,W)
        return img, msk


# ─── MedSAM Wrapper ───────────────────────────────────────────────────────────
class MedSAM(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder  = sam_model.image_encoder
        self.mask_decoder   = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        # freeze prompt encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, boxes_np):
        # image_embeddings
        img_emb = self.image_encoder(image)
        # prompt encoding
        with torch.no_grad():
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32, device=image.device)
            if boxes.ndim == 2:
                boxes = boxes[:, None, :]  # (B,1,4)
            sparse, dense = self.prompt_encoder(
                points=None, boxes=boxes, masks=None
            )
        # mask decoding
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        # upsample
        B,_,H,W = image.shape
        masks = F.interpolate(
            low_res_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        return masks  # (B,1,H,W)


# ─── Training Script ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_masks",  type=str, required=True)
    parser.add_argument("--val_images",   type=str, required=True)
    parser.add_argument("--val_masks",    type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to original SAM checkpoint (.pth)")
    parser.add_argument("--model_type",   type=str, default="vit_b")
    parser.add_argument("--device",       type=str, default="cpu")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--work_dir",     type=str, default="drusen_finetune")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # prepare output folder
    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.work_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(save_dir, os.path.basename(__file__)))

    # device
    device = torch.device(args.device)

    # load SAM and wrap
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = MedSAM(sam).to(device)
    model.train()

    # count params
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,} | Trainable: {trainable:,}")

    # optimizer on encoder + decoder
    params = list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # loss functions
    dice = DiceLoss(sigmoid=True)
    bce  = nn.BCEWithLogitsLoss()

    # data loaders
    train_ds = NpyDataset(args.train_images, args.train_masks)
    val_ds   = NpyDataset(args.val_images,   args.val_masks)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # derive full-image box
    sample_img, _ = train_ds[0]
    _,H,W = sample_img.shape  # (1,H,W)
    full_box = np.array([[0,0,W,H]], dtype=np.float32)

    best_val = float("inf")
    train_losses, val_losses = [], []

    # training loop
    for epoch in range(args.epochs):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            B = imgs.shape[0]
            boxes = np.tile(full_box, (B,1))  # (B,4)

            optimizer.zero_grad()
            preds = model(imgs, boxes)
            loss  = dice(preds, masks) + bce(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---------- VALIDATE ----------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                B = imgs.shape[0]
                boxes = np.tile(full_box, (B,1))
                preds = model(imgs, boxes)
                val_loss += (dice(preds, masks) + bce(preds, masks)).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # logging & checkpoints
        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # always save latest
        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        # plot losses
        plt.figure(figsize=(6,4))
        plt.plot(train_losses, label="train")
        plt.plot(val_losses,   label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

    print(f"Training complete — best val loss: {best_val:.4f}")
    print(f"Checkpoints & logs saved in: {save_dir}")


if __name__ == "__main__":
    main()
