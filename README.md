# Lung Nodule Detection with LIDC-IDRI

This project implements a computer vision pipeline to detect lung nodules from CT scans using the LIDC-IDRI dataset. It includes data loading, preprocessing, model training, and evaluation.

## Dataset

- Download raw DICOM images (133 GB) from the Cancer Imaging Archive (LIDC-IDRI) using the NBIA Data Retriever: https://cancerimagingarchive.net/collection/lidc-idri/.
- Download `Nodule Counts by Patient.xlsx` and `Patient Diagnoses.xlsx` and place them in `data/`.
- Unzip and structure under `data/LIDC-IDRI/<patient_id>/...`.

## Project Structure

```
DSCOLLABORATIVE_CV_Project/
├── data/
│   ├── LIDC-IDRI/            # raw DICOM folders
│   ├── Nodule_Counts_by_Patient.xlsx
│   ├── Patient_Diagnoses.xlsx
│   └── README.md            # instructions for data
├── notebooks/
│   └── EDA_and_Training.ipynb  # exploratory data analysis & training demo
├── outputs/                 # model checkpoints & logs
├── requirements.txt         # python dependencies
├── .gitignore
├── README.md
└── src/
    ├── utils.py            # helper scripts (generate CSVs)
    ├── dataloader.py       # dataset & transforms
    ├── model.py            # model definition
    └── train.py            # training & evaluation loop
```

## Quickstart

1. Create annotations CSV:
   ```bash
   python src/utils.py --xlsx data/Nodule_Counts_by_Patient.xlsx --out_csv data/annotations.csv
   ```
2. Train the model:
   ```bash
   python src/train.py \
     --data_dir data/LIDC-IDRI \
     --csv data/annotations.csv \
     --epochs 20 \
     --batch_size 16 \
     --lr 1e-4 \
     --output_dir outputs
   ```