import os
import glob
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2

class LIDCDataset(Dataset):
    def __init__(self, root_dir, csv_file, transforms=None):
        import pandas as pd
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms or Compose([Resize(224,224), ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row['patientid'])
        label = int(row['label'])
        # load DICOM slices
        folder = os.path.join(self.root_dir, pid)
        files = glob.glob(os.path.join(folder, '*.dcm'))
        slices = []
        for fp in files:
            ds = pydicom.dcmread(fp)
            pixel = ds.pixel_array.astype(np.float32)
            slices.append((ds.InstanceNumber, pixel))
        slices = [s for _, s in sorted(slices, key=lambda x: x[0])]
        volume = np.stack(slices, axis=0)
        # select central slice
        img = volume[volume.shape[0] // 2]
        # normalize
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255).astype(np.uint8)
        # to 3-channel
        img = np.stack([img]*3, axis=-1)
        # apply transforms
        augmented = self.transforms(image=img)
        tensor = augmented['image']  # CxHxW float tensor
        return tensor, torch.tensor(label, dtype=torch.long)