import os, numpy as np, torch
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_paths = sorted(
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith('_bscan.npy')
        )
        self.mask_paths = sorted(
            os.path.join(mask_folder, f)
            for f in os.listdir(mask_folder)
            if f.endswith('_mask.npy')
        )
        assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32)/255.0
        msk = np.load(self.mask_paths[idx]).astype(np.float32)/255.0
        img = torch.from_numpy(img[None]).float()  # (1,H,W)
        msk = torch.from_numpy(msk[None]).float()
        return img, msk
