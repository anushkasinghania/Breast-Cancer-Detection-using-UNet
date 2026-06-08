# Computer-aided Breast Cancer Diagnosis using Deep Convolutional Neural Networks

Computer-aided breast cancer diagnosis using deep convolutional neural
networks. This project performs **automated tumor segmentation on breast
ultrasound images** using a U-Net architecture, identifying and outlining
suspicious regions to assist in early breast cancer detection.

> Developed as part of a 2021 research internship. A companion research paper
> (`ICCCIS_637_UNet.pdf`) and internship report are included in this repository.

## Overview

Manual analysis of breast ultrasound scans is time-consuming and subject to
inter-observer variability. This project trains a U-Net — a fully convolutional
encoder–decoder network designed for biomedical image segmentation — to
automatically generate pixel-level masks of tumor regions from ultrasound images.

## Dataset

- **Source:** [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- **Classes:** benign, malignant, normal
- **Contents:** ultrasound images paired with ground-truth segmentation masks
- **Preprocessing:** all images and masks resized to **256 × 256**
- **Split:** 517 training images / 130 validation images (80/20 split, `random_state=42`)

## Model Architecture

A standard **U-Net** with a contracting (encoder) path, a bottleneck, and an
expanding (decoder) path with skip connections.

<img width="6912" height="3456" alt="U-Net" src="https://github.com/user-attachments/assets/d89d367f-9ece-4523-9e97-791706613123" />

- **`conv2d_block`** — two 3×3 convolutions, each followed by batch normalization
  and ReLU activation
- **Encoder:** 4 downsampling stages (Conv block → MaxPooling → Dropout), filters
  doubling each stage (16 → 32 → 64 → 128)
- **Bottleneck:** Conv block with 256 filters
- **Decoder:** 4 upsampling stages (Conv2DTranspose → concatenate skip connection
  → Dropout → Conv block)
- **Output:** 1×1 convolution with sigmoid activation (binary mask)

**Hyperparameters:** `n_filters=16`, `dropout=0.05`, `batchnorm=True`

## Training

| Setting | Value |
|---------|-------|
| Loss | BCE + Dice loss (`bce_dice_loss`) |
| Metrics | Dice coefficient, accuracy |
| Optimizer | Adam |
| Epochs | 50 |
| Batch size | 16 |
| Augmentation | rotation (±60°), zoom, horizontal & vertical flips |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best `val_loss`) |

## Results

Evaluated on the validation set using the best saved checkpoint:

| Metric | Value |
|--------|-------|
| Accuracy | **0.9429** |
| Dice coefficient | **0.5785** |
| Dice loss | 0.4215 |
| Loss | 0.5998 |

The model produces predicted masks that closely match the ground-truth tumor
regions. Sample outputs (Image → Ground Truth → Predicted Mask → Binary
Prediction) are available in the `Results/` folder.

## Repository Structure
├── BrC_Detection_Image_Segmentation.ipynb # Main notebook (data, model, training, evaluation)
├── ICCCIS_637_UNet.pdf # Research paper
├── 071_Anushka Singhania_Internship Report 2021.pdf
├── Results/ # Sample prediction outputs
└── README.md


## How to Run

1. Open `BrC_Detection_Image_Segmentation.ipynb` in Google Colab.
2. Upload your `kaggle.json` API key when prompted.
3. Run the cells in order — the notebook downloads the dataset, builds and trains
   the U-Net, and visualizes predictions.

## Tech Stack

Python · TensorFlow / Keras · OpenCV · scikit-image · scikit-learn · NumPy ·
Matplotlib

## Author

**Anushka Singhania**
