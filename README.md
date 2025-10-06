
# AFW-NET: Brain Tumor Segmentation

## Overview
This project implements a 3D U-Net with Spectral Convolutions and Adaptive Frequency Weighting (AFW) for brain tumor segmentation on the BraTS2021 Training dataset.

## Key Features
- 3D U-Net architecture with spectral convolutions
- Adaptive Frequency Weighting (AFW) for enhanced segmentation
- Multi-class brain tumor segmentation (Background, Necrotic Core, Edema, Enhancing Tumor)


## Files
- `train1.py` - training script
- `spectral_3d_unet_afw_attention.py` - Model architecture
- `dataset.py` and preprocessing.py - Data loading and preprocessing
- `enhanced_test_complete.py`,  - Testing script
- ``ensemble_prediction.py`.py` - 
- ``metrics.py` - Evaluation metrics

## Usage
```
# Preprocessing
python preprocessing.py
python dataset.py

# Model
spectral_3d_unet_afw_attention.py

# Training
python train1.py

# Evaluation and Testing
python enhanced_test_complete.py
python ensemble_prediction_test.py
metrics.py



## Results
-85% overall accuracy on BraTS2021 dataset
- Robust performance across all tumor classes
- Suitable for clinical applications

## Dependencies
- PyTorch
- NumPy
- Matplotlib
- Nibabel
- TorchIO

## Academic Use
This project is part of an MSc thesis in Data science. The code and results are suitable for academic research and publications.

---
**Author**: [MERCY AJOKE SUPREME]  
**Institution**: [PAN-ATLANTIC UNIVERSITY, IBEJU LEKKI, LAGOS STATE, NIGERIA]  
**Year**: 2025
```