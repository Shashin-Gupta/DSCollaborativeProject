import os
import boto3
import argparse
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataloader import DRDataset
from src.model import DRClassifier
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, HueSaturationValue,
    RandomBrightnessContrast, GaussNoise, CoarseDropout, Resize, Normalize, Compose)
from albumentations.pytorch import ToTensorV2

def get_transforms(train=True):
    if train:
        return Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            HueSaturationValue(p=0.5),
            RandomBrightnessContrast(p=0.5),
            GaussNoise(p=0.3),
            CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return Compose([
        Resize(224,224), 
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in tqdm(loader, desc='Train'):  
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Eval'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
    return total_loss/len(loader.dataset), correct/len(loader.dataset)
    
def main(args):
    load_dotenv()

    os.makedirs(args.output_dir, exist_ok=True)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('ACCESS_KEY'),
        aws_secret_access_key=os.getenv('SECRET_KEY'),
        region_name='us-east-2'
    )

    df = pd.read_csv(args.csv)

    # Comment out after hyperparameter tuning
    # df = df.sample(frac=0.2, random_state=9).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Diagnosis'], random_state=9)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)

    train_ds = DRDataset(train_df, s3_client, args.bucket, transforms=get_transforms(train=True))
    val_ds = DRDataset(val_df, s3_client, args.bucket, transforms=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRClassifier(num_classes=2, pretrained=True).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}: Train loss={train_loss:.4f}, acc={train_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_plot.png'))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()
    main(args)