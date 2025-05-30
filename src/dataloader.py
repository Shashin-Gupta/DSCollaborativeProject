from io import BytesIO
import os
import glob
import boto3
from dotenv import load_dotenv
import numpy as np
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

class DRDataset(Dataset):
    def __init__(self, df, s3_client, bucket_name, transforms=None, prefix='Diabetic Retinopathy Screening AI.v1i.multiclass/'):
        self.df = df.reset_index(drop=True)
        self.s3 = s3_client
        self.bucket = bucket_name
        self.prefix = prefix  # folder inside the bucket, if any
        self.transforms = transforms or Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),  # ImageNet normalization (standard for pretrained models)
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # load_dotenv()
        # s3 = boto3.client(
        #     's3',
        #     aws_access_key_id=os.getenv('ACCESS_KEY'),
        #     aws_secret_access_key=os.getenv('SECRET_KEY'),
        #     region_name='us-east-2'
        # )
    
        row = self.df.iloc[idx]
        pid = str(row['filename'])
        label = int(row['Diagnosis'])

        key = f"{self.prefix}{pid}"  # assumes file name = patientid.jpg
        print('KEYSTER', key)
        # Read image directly from S3
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = response['Body'].read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Apply transforms
        img = np.array(img)
        augmented = self.transforms(image=img)
        tensor = augmented['image']
        return tensor, torch.tensor(label, dtype=torch.long)
