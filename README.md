# Histopathology Deep Learning Image Segmentation

## About the Project

This is our academic group project based on deep learning and medical image analysis.

The main objective is to segment cellular structures from cervical cytology images using a deep learning model.

Our project focuses on two structures:

- Cytoplasm
- Nucleus

We implemented a dual-output U-Net architecture that predicts both segmentation masks from a single input image.

We also perform morphological analysis to extract different features from the segmented regions.

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Streamlit

## Dataset

We used the **CX22 Multi-Train dataset**, downloaded from Kaggle.

Dataset: [CX22 Multi-Train – Kaggle](https://www.kaggle.com/datasets/niladrishekharray/cx22-multi-train)

The dataset contains cervical cytology images with corresponding ground-truth masks for cytoplasm and nucleus segmentation.

The dataset is not included in this repository and must be downloaded separately.

## Model Architecture

We implemented a U-Net architecture with two output channels.

- Input image: 512 × 512
- Output channel 1: Cytoplasm
- Output channel 2: Nucleus

## Training Details

| Parameter | Value |
|---|---|
| Model | U-Net |
| Epochs | 30 |
| Batch size | 1 |
| Optimizer | AdamW |
| Loss Function | BCEWithLogits + Dice Loss |
| Training Images | 320 |
| Validation Images | 80 |

## Validation Results

| Metric | Cytoplasm | Nucleus |
|---|---:|---:|
| Dice Score | 95.41% | 84.20% |
| IoU | 91.28% | 73.28% |
| Precision | 94.78% | 83.72% |
| Recall | 96.19% | 85.37% |

**Mean Dice Score: 89.80%**

These are validation results from our experiment.

## Morphological Analysis

After segmentation, we extract different morphological features:

- Nuclear area
- Cytoplasmic area
- Nucleus-to-cytoplasm area ratio
- Nuclear perimeter
- Circularity
- Eccentricity
- Solidity

These features help us study the shape and size characteristics of the segmented structures.

## Streamlit Application

We developed a Streamlit application with two main features:

**1. Custom Image Segmentation**
- Upload a cervical cytology image.
- Generate cytoplasm and nucleus masks.
- View original images, predicted masks and overlays.
- View morphological measurements.

**2. Test Dataset Evaluation**
- Select an image from the test dataset.
- Automatically load its corresponding ground-truth masks.
- Compare predicted masks with ground-truth masks.
- Calculate segmentation metrics.

## Project Results

### Original Image vs Predicted Segmentation

Our prediction examples are available in:

`outputs/predictions/`

### Training and Evaluation Graphs

The `outputs/` folder also contains training and evaluation graphs and result files.

## Research Papers

The research papers referred to during this project are listed in the `references/README.md` file.

## How to Run

Clone the repository:

```bash
git clone https://github.com/ankush-agarwal209/Histopathology-deep-learning-image-segmentation
cd https://github.com/ankush-agarwal209/Histopathology-deep-learning-image-segmentation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Make sure the trained model checkpoint is available at:

`outputs/best_unet.pth`

Run the application:

```bash
streamlit run app.py
```
