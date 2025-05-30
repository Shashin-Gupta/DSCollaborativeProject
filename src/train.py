import os
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataloader import DRDataset
from src.model import NoduleClassifier
from albumentations import Compose, Resize, RandomBrightnessContrast, Rotate
from albumentations.pytorch import ToTensorV2

def get_transforms(train=True):
    if train:
        return Compose([
            Resize(224,224),
            RandomBrightnessContrast(),
            Rotate(limit=15),
            ToTensorV2()
        ])
    return Compose([Resize(224,224), ToTensorV2()])

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
    
def main(df, data_dir, batch_size=8, lr=1e-4, epochs=10, output_dir="outputs"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    df = df.rename(columns={"filename": "img", "Diagnosis": "label"})

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=9)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)

    train_ds = DRDataset(data_dir, 'data/train.csv', transforms=get_transforms(train=True))
    val_ds   = DRDataset(data_dir, 'data/val.csv', transforms=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = NoduleClassifier(num_classes=2, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}: Train loss={train_loss:.4f}, acc={train_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)

main(
    df,
    data_dir="s3://diabetic-retinopathy-project-2025/images",  # or local if testing
    batch_size=8,
    lr=1e-4,
    epochs=2,
    output_dir="outputs"
)
