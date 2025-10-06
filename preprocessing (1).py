#!/usr/bin/env python3
"""
Enhanced Preprocessing Module for Brain Tumor Segmentation
Improved with better configuration management, error handling, and flexibility.
"""

import os
import glob
import tarfile
import numpy as np
import warnings
import torchio as tio
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Default paths - will be set dynamically
DATA_DIR = None
EXTRACT_DIR = None

MODALITIES = ['flair', 't1', 't1ce', 't2']
TARGET_SPACING = (1.0, 1.0, 1.0)  # mm

def set_data_paths(data_dir: str, extract_dir: Optional[str] = None):
    """Set data and extract directories dynamically"""
    global DATA_DIR, EXTRACT_DIR
    DATA_DIR = data_dir
    EXTRACT_DIR = extract_dir or data_dir
    logger.info(f"Data paths set - DATA_DIR: {DATA_DIR}, EXTRACT_DIR: {EXTRACT_DIR}")

def remap_braTS_labels(segmentation: np.ndarray) -> np.ndarray:
    """
    Enhanced BraTS label remapping with validation.
    
    Remaps BraTS labels from {0, 1, 2, 4} to {0, 1, 2, 3}.
    
    Original BraTS labels:
    - 0: Background (healthy tissue)
    - 1: Necrotic and Non-Enhancing Tumor Core (NCR/NET)
    - 2: Peritumoral Edema (ED)
    - 4: Enhancing Tumor (ET)
    
    Args:
        segmentation: Input segmentation array
        
    Returns:
        Remapped segmentation array
    """
    remapped = np.copy(segmentation)
    remapped[segmentation == 4] = 3
    return remapped

def extract_if_needed():
    """Extract the dataset tar file if it hasn't been extracted yet."""
    if not EXTRACT_DIR or not os.path.exists(EXTRACT_DIR):
        if not DATA_DIR:
            logger.warning("No data directory set, skipping extraction")
            return
            
        print("📦 Extracting BraTS2021 data...")
        try:
            with tarfile.open(DATA_DIR) as tar:
                tar.extractall(path=EXTRACT_DIR)
            print("✅ Extraction complete.")
        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise
    else:
        print("✅ Dataset already extracted.")

def zscore_normalize(volume):
    """Z-score normalization on non-zero voxels only."""
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    volume[mask] = (volume[mask] - mean) / (std + 1e-8)
    return volume

def preprocess_subject(subject_path, target_spacing=TARGET_SPACING):
    """
    Enhanced subject preprocessing with comprehensive error handling.
    
    Args:
        subject_path: Path to subject directory
        target_spacing: Target spacing for resampling
        
    Returns:
        Tuple of (image_4d, seg)
    """
    subject_id = os.path.basename(subject_path)

    try:
        subject = tio.Subject(
            flair=tio.ScalarImage(os.path.join(subject_path, f"{subject_id}_flair.nii.gz")),
            t1=tio.ScalarImage(os.path.join(subject_path, f"{subject_id}_t1.nii.gz")),
            t1ce=tio.ScalarImage(os.path.join(subject_path, f"{subject_id}_t1ce.nii.gz")),
            t2=tio.ScalarImage(os.path.join(subject_path, f"{subject_id}_t2.nii.gz")),
            seg=tio.LabelMap(os.path.join(subject_path, f"{subject_id}_seg.nii.gz"))
        )

        # 🔁 Resample image and segmentation
        subject = tio.Resample(target_spacing)(subject)

        # Z-score normalize only modalities
        for modality in MODALITIES:
            data = subject[modality].data.numpy()[0]  # (H, W, D)
            norm_data = zscore_normalize(data)
            subject[modality].set_data(tio.ScalarImage(tensor=norm_data[None]).data)

        # Stack image channels or # Stack the 4 modalities
        image_4d = np.stack([subject[m].data.numpy()[0] for m in MODALITIES], axis=0)  # (4, H, W, D)
        
        seg = subject['seg'].data.numpy()[0].astype(np.uint8)
        seg = remap_braTS_labels(seg)  # ✅ remaps 4 → 3    
        return image_4d, seg
        
    except Exception as e:
        logger.error(f"Failed to preprocess subject {subject_id}: {e}")
        raise

def list_subjects(data_root):
    """List all subject directories with enhanced validation."""
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    # Find subject directories
    pattern = str(data_root / "BraTS2021_*")
    subject_dirs = sorted(glob.glob(pattern))
    
    if not subject_dirs:
        raise ValueError(f"No BraTS subjects found in {data_root}")
    
    # Validate subjects
    valid_subjects = []
    for subject_dir in subject_dirs:
        subject_path = Path(subject_dir)
        subject_id = subject_path.name
        
        # Check for required files
        required_files = [f"{subject_id}_{modality}.nii.gz" for modality in MODALITIES]
        required_files.append(f"{subject_id}_seg.nii.gz")
        
        if all((subject_path / file_name).exists() for file_name in required_files):
            valid_subjects.append(str(subject_path))
        else:
            logger.warning(f"Subject {subject_id} missing required files")
    
    logger.info(f"Found {len(valid_subjects)} valid subjects out of {len(subject_dirs)} total")
    return valid_subjects

# Example usage
if __name__ == "__main__":
    # Set data paths
    set_data_paths(
        data_dir="C:\\Users\\Desup\\OneDrive\\Desktop\\PAU\\Love 😍\\BraTS\\data\\BraTS2021_Training_Data",
        extract_dir="C:\\Users\\Desup\\OneDrive\\Desktop\\PAU\\Love 😍\\BraTS\\data\\BraTS2021_Training_Data"
    )
    
    # Test preprocessing
    try:
        subjects = list_subjects(DATA_DIR)
        print(f"Found {len(subjects)} subjects")
        
        if subjects:
            # Test preprocessing on first subject
            image, seg = preprocess_subject(subjects[0])
            print(f"Preprocessed image shape: {image.shape}")
            print(f"Preprocessed segmentation shape: {seg.shape}")
            print(f"Segmentation labels: {np.unique(seg)}")
    
    except Exception as e:
        print(f"Error: {e}")
