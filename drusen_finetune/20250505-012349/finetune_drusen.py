# train_one_gpu.py
# -*- coding: utf-8 -*-
"""
Fine-tune MedSAM on OCT drusen data.
Train image encoder + mask decoder; freeze prompt encoder.
"""

import os
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import transform
from monai.losses import DiceLoss
from segment_anything import sam_model_registry


# ─── Dataset ──────────────────────────────────────────────────────────────────
class NpyDataset(Dataset):
    """
    Dataset for OCT drusen B-scans and binary masks.
    Resizes all images and masks to 1024×1024 to match SAM's ViT patch embedding.
    """
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
        # Load grayscale image & mask
        gray = np.load(self.image_paths[idx]).astype(np.float32)
        gt   = np.load(self.mask_paths[idx]).astype(np.float32)

        # Resize to 1024×1024 (SAM ViT-B expects 1024×1024 inputs)
        gray_resized = transform.resize(
            gray, (1024, 1024), order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        mask_resized = transform.resize(
            gt,   (1024, 1024), order=0, preserve_range=True, anti_aliasing=False
        )
        # Binarize mask
        mask_bin = (mask_resized > 0.5).astype(np.float32)

        # Stack to 3-channel for image encoder
        img = np.stack([gray_resized]*3, axis=0)  # (3, H, W)
        msk = mask_bin[None]                      # (1, H, W)

        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

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
        img_emb = self.image_encoder(image)
        with torch.no_grad():
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32, device=image.device)
            if boxes.ndim == 2:
                boxes = boxes[:, None, :]
            sparse, dense = self.prompt_encoder(
                points=None, boxes=boxes, masks=None
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        B,_,H,W = image.shape
        masks = F.interpolate(
            low_res_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        return masks


# ─── Training Script ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # If using MPS backend, lower batch_size to avoid OOM

    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_masks",  type=str, required=True)
    parser.add_argument("--val_images",   type=str, required=True)
    parser.add_argument("--val_masks",    type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to original SAM checkpoint (.pth)")
    parser.add_argument("--model_type",   type=str, default="vit_b")
    parser.add_argument("--device",       type=str, default="cpu")
    parser.add_argument("--batch_size",   type=int, default=1,
                        help="Batch size (auto-reduced to 1 on MPS if needed)")
    parser.add_argument("--epochs",       type=int, default=10)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--work_dir",     type=str, default="drusen_finetune")
    args = parser.parse_args()
    # force batch_size=1 on MPS to reduce memory footprint
    if args.device.startswith("mps") and args.batch_size > 1:
        print("MPS detected: overriding batch_size to 1 to avoid OOM")
        args.batch_size = 1

    torch.manual_seed(42)
    np.random.seed(42)

    run_id   = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.work_dir, run_id)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(save_dir, os.path.basename(__file__)))

    device = torch.device(args.device)

    sam   = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = MedSAM(sam).to(device)
    model.train()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,} | Trainable: {trainable:,}")

    params = list(model.image_encoder.parameters()) + list(model.mask_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    dice = DiceLoss(sigmoid=True)
    bce  = nn.BCEWithLogitsLoss()

    train_ds = NpyDataset(args.train_images, args.train_masks)
    val_ds   = NpyDataset(args.val_images,   args.val_masks)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=False)

    sample_img,_ = train_ds[0]
    _,H,W = sample_img.shape
    full_box = np.array([[0,0,W,H]], dtype=np.float32)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            B = imgs.shape[0]
            boxes = np.tile(full_box, (B,1))

            optimizer.zero_grad()
            preds = model(imgs, boxes)
            loss  = dice(preds, masks) + bce(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

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

        print(f"[Epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, "latest.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

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
