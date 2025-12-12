# TEM Cell-Segmentation-EECS-453-Final-Project

This repository contains an Attention U-Net model for multi-class semantic segmentation of Transmission Electron Microscopy (TEM) images. It includes scripts for training the model and running inference on new data.
## Project Structure

- `run_model.py`: Main executable script for running inference on images.
- `best_model.pth`: The trained model weights file (highest validation Dice score).
- `test_data_tiff/`: Directory containing test images (`raw_*.tiff`).
- `predictions/`: Directory where inference results are saved.

## Requirements

To run this code, you need Python and the following libraries installed:

- torch
- torchvision
- numpy
- matplotlib
- tifffile
- tqdm
- albumentations
- opencv-python

You can install them via pip:

```bash
pip install torch torchvision numpy matplotlib tifffile tqdm albumentations opencv-python
```

## Usage

To generate segmentation masks for a set of images using the trained model:

1. Place your input TIFF images in a folder (e.g., `test_data_tiff`).
2. Run the inference script:

```bash
python run_model.py
```

By default, this script uses the `./test_data_tiff` folder. You can change the input folder by editing the `INPUT_FOLDER` variable at the top of `run_model.py`.

The script will:
- Load the `AttentionUNet` architecture and the weights from `best_model.pth`.
- Process each image using a sliding-window approach to handle large resolutions.
- Save the predicted segmentation masks (as `.tiff`) and visual comparisons (as `.png`) in the `predictions/` folder.

## Model Details

- **Architecture**: U-Net with Attention Gates
- **Input**: 1-channel grayscale images (processed as 256x256 patches)
- **Classes**:
  - 0: Background
  - 1: Blood Vessels
  - 2: Myelinated Axons
  - 3: Unmyelinated Axons
  - 4: Schwann Cells
- **Training Strategy**: AdamW optimizer, Cross-Entropy + Dice Loss, and geometric data augmentations
