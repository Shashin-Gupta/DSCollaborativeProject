# Lung CT Scan Segmentation with U-Net

This project performs semantic segmentation of lung nodules using the [LUNA16](https://luna16.grand-challenge.org/Download/) CT scan dataset. It uses a U-Net architecture to learn pixel-wise predictions from annotated 3D CT volumes.

## 🧠 Model

- U-Net (2D slices)
- Loss: Dice Loss + Binary Crossentropy
- Metric: Dice Coefficient, IoU

## 🗃 Dataset

- Dataset: [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Download/)
- Format: NIfTI / MHD
- Preprocessing: Rescale, Normalize, Convert to 2D slices

## 🏁 Getting Started

### Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/lung-ct-unet-segmentation.git
cd lung-ct-unet-segmentation
