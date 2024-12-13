import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rawpy
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from isp_ops import apply_white_balance, demosaic, color_correction, tone_mapping, apply_gamma

class ZurichRawToRgbDataset(Dataset):
    def __init__(self, root_dir, transform=None, cc_matrix=None):
        self.root_dir = root_dir
        self.raw_dir = os.path.join(root_dir, 'raw')
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.transform = transform
        self.cc_matrix = cc_matrix if cc_matrix is not None else torch.eye(3)

        self.raw_files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.dng')])
        self.rgb_files = [f.replace('.dng', '.png') for f in self.raw_files]

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_files[idx])
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])

        with rawpy.imread(raw_path) as raw:
            raw_img = raw.raw_image.copy().astype(np.float32)
            black_level = np.reshape(raw.black_level_per_channel, (2,2))
            black_level = np.tile(black_level, (raw.raw_image.shape[0]//2, raw.raw_image.shape[1]//2))
            white_level = raw.white_level
            raw_norm = (raw_img - black_level) / (white_level - black_level)
            
            raw_tensor = torch.from_numpy(raw_norm).unsqueeze(0) # shape: (1,H,W), CPU tensor
            wb_gains = raw.camera_whitebalance[:3] if len(raw.camera_whitebalance) > 3 else raw.camera_whitebalance
            wb_gains = torch.tensor(wb_gains, dtype=torch.float32)

            # Apply white balance
            wb_applied = apply_white_balance(raw_tensor, wb_gains)

            # Multi-threading: 
            # Split wb_applied into two vertical tiles and demosaic each in parallel
            B, C, H, W = wb_applied.shape
            mid = H // 2
            top_tile = wb_applied[:,:,0:mid,:]
            bottom_tile = wb_applied[:,:,mid:,:]

            def demosaic_tile(t):
                return demosaic(t)  # returns B,3,partial_H,W

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_top = executor.submit(demosaic_tile, top_tile)
                future_bottom = executor.submit(demosaic_tile, bottom_tile)
                top_demosaiced = future_top.result()
                bottom_demosaiced = future_bottom.result()

            # Stitch the two halves back together
            demosaiced = torch.cat([top_demosaiced, bottom_demosaiced], dim=2) # B,3,H,W

            # Color correction
            ccm = torch.tensor(raw.color_matrix[:3,:3], dtype=torch.float32)
            cc_img = color_correction(demosaiced, ccm)

            # Tone mapping and gamma
            tm_img = tone_mapping(cc_img)
            final_img = apply_gamma(tm_img)

        rgb_gt = Image.open(rgb_path).convert('RGB')
        rgb_gt_tensor = transforms.ToTensor()(rgb_gt)

        # Model will learn to map raw_tensor -> rgb_gt_tensor
        if self.transform:
            raw_tensor, rgb_gt_tensor = self.transform((raw_tensor, rgb_gt_tensor))

        return raw_tensor, rgb_gt_tensor

def get_data_loaders(root_dir, batch_size=4, num_workers=4, shuffle=True):
    train_dataset = ZurichRawToRgbDataset(root_dir=os.path.join(root_dir, 'train'))
    val_dataset = ZurichRawToRgbDataset(root_dir=os.path.join(root_dir, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
