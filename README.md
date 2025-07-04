# Detecting Diabetic Retinopathy with Computer Vision

By: Cyrus Navasca, Shashin Gupta and Owen Feng

This project implements a computer vision pipeline to detect diabetic retinopathy, an eye disease affecting diabetes patients, from retina scans. It includes data loading, preprocessing, model training, and evaluation.

## Project Structure

```
DSCOLLABORATIVE_CV_Project/
├── data/
│   ├── annotations_combined.csv
│   ├── train.csv
│   ├── val.csv
│   └── README.md           
├── notebooks/
│   └── EDA_and_Training.ipynb  # exploratory data analysis & training demo
├── outputs/                 # model checkpoints & logs
├── requirements.txt         # python dependencies
├── .gitignore
├── README.md
└── src/
    ├── dataloader.py       # dataset & transforms
    ├── model.py            # model definition
    └── train.py            # training & evaluation loop
```

## Model Training

   ```bash
   PYTHONPATH=. python3 src/train.py \
  --csv data/annotations_combined.csv \
  --bucket diabetic-retinopathy-project-2025 \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --output_dir outputs
   ```
