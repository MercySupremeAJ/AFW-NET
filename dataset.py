#!/usr/bin/env python3
"""
Enhanced Dataset Module for Brain Tumor Segmentation
Improved with better error handling, configuration management, and optimization.
"""

import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
from collections import Counter
import os
import logging
from typing import Tuple, List, Optional, Dict, Any
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torchio as tio

class BraTSDataset(Dataset):
    """
    Enhanced BraTS Dataset with improved preprocessing, error handling, and optimization.
    
    Features:
    - Dynamic preprocessing with TorchIO
    - Comprehensive error handling
    - Configurable augmentation pipeline
    - Dataset statistics computation
    - Memory optimization
    - Label distribution analysis
    """
    
    def __init__(
        self, 
        data_dir: str, 
        augment: bool = False, 
        target_shape: Tuple[int, int, int] = (240, 240, 152),
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced BraTS dataset.
        
        Args:
            data_dir: Path to dataset directory
            augment: Whether to apply augmentations
            target_shape: Target shape for all volumes
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.target_shape = target_shape
        self.config = config or {}
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Import preprocessing functions
        try:
            from preprocessing import preprocess_subject, list_subjects, extract_if_needed, remap_braTS_labels
            self.preprocess_subject = preprocess_subject
            self.list_subjects = list_subjects
            self.extract_if_needed = extract_if_needed
            self.remap_braTS_labels = remap_braTS_labels
        except ImportError as e:
            logger.error(f"Failed to import preprocessing functions: {e}")
            raise
        
        # Extract data if needed
        self.extract_if_needed()
        
        # Get subject directories
        try:
            print("🔍 Scanning for BraTS subjects...")
            self.subject_dirs = self.list_subjects(data_root=str(self.data_dir))
            if not self.subject_dirs:
                raise ValueError(f"No subjects found in {data_dir}")
            print(f"✅ Found {len(self.subject_dirs)} subjects")
            logger.info(f"Found {len(self.subject_dirs)} subjects")
        except Exception as e:
            logger.error(f"Failed to list subjects: {e}")
            raise
        
        # Initialize augmentation pipeline
        self._setup_augmentation_pipeline()
        
        # Setup preprocessing
        self.crop_or_pad = tio.CropOrPad(self.target_shape)
        
        # Class names for brain tumor segmentation (define early for statistics)
        self.class_names = {
            0: "Background",
            1: "Necrotic Core", 
            2: "Edema",
            3: "Enhancing Tumor"
        }
        
        # Compute dataset statistics (optional)
        if self.config.get('compute_stats', True):
            print("📊 Computing dataset statistics...")
            self._compute_dataset_statistics()
            print("✅ Dataset statistics computed")
        
        # Initialize printed count for sanity checks
        if not hasattr(BraTSDataset, 'printed_count'):
            BraTSDataset.printed_count = 0
    
    def _setup_augmentation_pipeline(self):
        """Setup configurable augmentation pipeline"""
        if not self.augment:
            self.transform = None
            return
        
        # Default augmentation configuration
        aug_config = self.config.get('augmentation', {})
        
        # Create augmentation pipeline
        transforms = []
        
        # Spatial augmentations
        if aug_config.get('flip', True):
            transforms.append(tio.RandomFlip(
                axes=aug_config.get('flip_axes', ('LR',)), 
                p=aug_config.get('flip_prob', 0.5)
            ))
        
        if aug_config.get('affine', True):
            transforms.append(tio.RandomAffine(
                scales=aug_config.get('affine_scales', 0.1),
                degrees=aug_config.get('affine_degrees', 10),
                translation=aug_config.get('affine_translation', 5),
                p=aug_config.get('affine_prob', 0.75)
            ))
        
        # Intensity augmentations
        if aug_config.get('noise', True):
            transforms.append(tio.RandomNoise(
                std=aug_config.get('noise_std', 0.1),
                p=aug_config.get('noise_prob', 0.5)
            ))
        
        if aug_config.get('bias_field', True):
            transforms.append(tio.RandomBiasField(
                p=aug_config.get('bias_field_prob', 0.3)
            ))
        
        if aug_config.get('blur', False):
            transforms.append(tio.RandomBlur(
                std=aug_config.get('blur_std', (0.5, 1.0)),
                p=aug_config.get('blur_prob', 0.3)
            ))
        
        if aug_config.get('gamma', False):
            transforms.append(tio.RandomGamma(
                log_gamma=aug_config.get('gamma_range', (-0.3, 0.3)),
                p=aug_config.get('gamma_prob', 0.3)
            ))
        
        self.transform = tio.Compose(transforms) if transforms else None
        logger.info(f"Augmentation pipeline: {len(transforms) if transforms else 0} transforms")
    
    def _compute_dataset_statistics(self):
        """Compute dataset-level statistics for analysis with caching support"""
        # Check for cached statistics first
        cache_path = self._get_stats_cache_path()
        if self._load_cached_statistics(cache_path):
            print("✅ Loaded cached dataset statistics")
            return
        
        print("📊 Computing dataset statistics...")
        logger.info("Computing dataset statistics...")
        
        self.label_distribution = Counter()
        self.shape_statistics = []
        self.intensity_statistics = []
        
        # Sample subset for statistics (to avoid memory issues)
        sample_size = min(len(self.subject_dirs), self.config.get('stats_sample_size', 200))  # Increased from 50 to 200
        sample_indices = np.random.choice(len(self.subject_dirs), sample_size, replace=False)
        print(f"📊 Sampling {sample_size} subjects for statistics computation...")
        
        processed_count = 0
        for idx in sample_indices:
            try:
                subject_path = self.subject_dirs[idx]
                image_np, seg_np = self.preprocess_subject(subject_path)
                seg_np = self.remap_braTS_labels(seg_np)
                
                # Check if segmentation has any non-zero values
                if seg_np.max() == 0:
                    print(f"⚠️ Subject {idx}: No segmentation labels found")
                    continue
                
                # Label distribution
                labels, counts = np.unique(seg_np, return_counts=True)
                self.label_distribution.update(dict(zip(labels, counts)))
                processed_count += 1
                
                if processed_count % 50 == 0:
                    print(f"📊 Processed {processed_count}/{sample_size} subjects...")
                
                # Shape statistics
                self.shape_statistics.append(image_np.shape)
                
                # Intensity statistics (for non-zero voxels)
                for modality in range(image_np.shape[0]):
                    modality_data = image_np[modality]
                    non_zero_mask = modality_data > 0
                    if non_zero_mask.sum() > 0:
                        self.intensity_statistics.append({
                            'modality': modality,
                            'mean': modality_data[non_zero_mask].mean(),
                            'std': modality_data[non_zero_mask].std(),
                            'min': modality_data[non_zero_mask].min(),
                            'max': modality_data[non_zero_mask].max()
                        })
                
            except Exception as e:
                logger.warning(f"Failed to process subject {idx} for statistics: {e}")
                continue
        
        # Log statistics
        logger.info(f"Label distribution: {dict(self.label_distribution)}")
        logger.info(f"Shape statistics computed for {len(self.shape_statistics)} samples")
        logger.info(f"Intensity statistics computed for {len(self.intensity_statistics)} modalities")
        
        # Save statistics to cache
        self._save_statistics_cache(cache_path)
        
        # Save voxel distribution to CSV for class weight computation
        self._save_voxel_distribution_csv()
    
    def _save_voxel_distribution_csv(self):
        """Save voxel distribution to CSV file for class weight computation"""
        import csv
        import os
        
        # Create reports directory if it doesn't exist
        reports_dir = "/content/drive/MyDrive/BraTS_Project/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        csv_path = os.path.join(reports_dir, "brats_mapped_voxel_distribution.csv")
        
        # Check if CSV already exists and is valid
        if os.path.exists(csv_path):
            try:
                # Verify the CSV has the expected format
                with open(csv_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    row = next(reader, None)
                    if row and all(f'class_{i}' in row for i in range(4)):
                        print(f"✅ Voxel distribution CSV already exists: {csv_path}")
                        print(f"📊 Class distribution: {dict(row)}")
                        return
            except Exception as e:
                logger.warning(f"Invalid existing CSV, will recreate: {e}")
        
        # Prepare data in the format expected by train1.py
        # Create a single row with class_0, class_1, class_2, class_3 columns
        csv_data = {
            'class_0': 0,
            'class_1': 0, 
            'class_2': 0,
            'class_3': 0
        }
        
        # Fill in the actual counts
        for label, count in self.label_distribution.items():
            if label in csv_data:
                csv_data[f'class_{label}'] = int(count)
        
        # Check if we have any data
        total_voxels = sum(csv_data.values())
        if total_voxels == 0:
            print("⚠️ No voxel data found! Using default class weights...")
            # Use default balanced weights
            csv_data = {
                'class_0': 1000000,  # Background
                'class_1': 10000,    # Necrotic Core  
                'class_2': 50000,    # Edema
                'class_3': 20000     # Enhancing Tumor
            }
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['class_0', 'class_1', 'class_2', 'class_3']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(csv_data)
        
        print(f"✅ Voxel distribution saved to: {csv_path}")
        print(f"📊 Class distribution: {csv_data}")
        logger.info(f"Voxel distribution CSV saved to {csv_path}")
    
    def _get_stats_cache_path(self) -> str:
        """Get the path for cached statistics file"""
        import hashlib
        
        # Create a hash based on data directory and configuration
        cache_key = f"{self.data_dir}_{self.config.get('stats_sample_size', 200)}_{len(self.subject_dirs)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        reports_dir = "/content/drive/MyDrive/BraTS_Project/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        return os.path.join(reports_dir, f"dataset_statistics_cache_{cache_hash}.json")
    
    def _load_cached_statistics(self, cache_path: str) -> bool:
        """Load cached statistics if available"""
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Restore statistics from cache
            self.label_distribution = Counter(cached_data.get('label_distribution', {}))
            self.shape_statistics = cached_data.get('shape_statistics', [])
            self.intensity_statistics = cached_data.get('intensity_statistics', [])
            
            # Display cached statistics
            print(f"📊 Cached statistics loaded:")
            print(f"   📈 Label distribution: {dict(self.label_distribution)}")
            print(f"   📏 Shape statistics: {len(self.shape_statistics)} samples")
            print(f"   🔢 Intensity statistics: {len(self.intensity_statistics)} modalities")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load cached statistics: {e}")
            return False
    
    def _save_statistics_cache(self, cache_path: str):
        """Save computed statistics to cache"""
        try:
            cache_data = {
                'label_distribution': dict(self.label_distribution),
                'shape_statistics': self.shape_statistics,
                'intensity_statistics': self.intensity_statistics,
                'num_subjects': len(self.subject_dirs),
                'target_shape': self.target_shape,
                'timestamp': str(Path().cwd())  # Simple timestamp placeholder
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"💾 Statistics cached to: {cache_path}")
            logger.info(f"Statistics cached to {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save statistics cache: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        return {
            'num_subjects': len(self.subject_dirs),
            'target_shape': self.target_shape,
            'augment': self.augment,
            'class_names': self.class_names,
            'label_distribution': dict(self.label_distribution) if hasattr(self, 'label_distribution') else {},
            'shape_statistics': self.shape_statistics if hasattr(self, 'shape_statistics') else [],
            'intensity_statistics': self.intensity_statistics if hasattr(self, 'intensity_statistics') else []
        }
    
    def save_dataset_info(self, save_path: str):
        """Save dataset information to file"""
        info = self.get_dataset_info()
        with open(save_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Dataset info saved to {save_path}")
    
    def __len__(self) -> int:
        return len(self.subject_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        if idx >= len(self.subject_dirs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.subject_dirs)}")
        
        try:
            subject_path = self.subject_dirs[idx]
            
            # Preprocess subject
            image_np, seg_np = self.preprocess_subject(subject_path)
            seg_np = self.remap_braTS_labels(seg_np)
            
            # Sanity check for first few samples
            if BraTSDataset.printed_count < 4:
                labels, counts = np.unique(seg_np, return_counts=True)
                # Convert to regular Python types for readable output
                readable_dict = {int(label): int(count) for label, count in zip(labels, counts)}
                logger.info(f"[Sanity Check {BraTSDataset.printed_count+1}] Sample {idx} Label Voxel Counts: {readable_dict}")
                BraTSDataset.printed_count += 1
            
            # Wrap in TorchIO Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.from_numpy(image_np).float()),
                label=tio.LabelMap(tensor=torch.from_numpy(seg_np).unsqueeze(0).long())
            )
            
            # Apply preprocessing
            subject = self.crop_or_pad(subject)
            
            # Apply augmentation if enabled
            if self.transform:
                subject = self.transform(subject)
            
            # Extract final tensors
            image_tensor = subject['image'].data  # shape: (4, H, W, D)
            label_tensor = subject['label'].data.squeeze(0)  # shape: (H, W, D)
            
            # Ensure consistent data types
            image_tensor = image_tensor.float()
            label_tensor = label_tensor.long()
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {e}")
            raise


def get_dataloaders(data_dir, batch_size, train_ratio=0.7, val_ratio=0.15, augment=False, num_workers=4, pin_memory=True, config=None):
    """
    Enhanced dataloaders with improved configuration and error handling.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        augment: Whether to apply augmentations
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate ratios
    test_ratio = 1.0 - train_ratio - val_ratio
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Create dataset
    try:
        full_dataset = BraTSDataset(data_dir, augment=augment, config=config)
        n_total = len(full_dataset)
        logger.info(f"Total dataset size: {n_total}")
        
        # Calculate split sizes
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val
        
        logger.info(f"Split sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        # Create splits
        train_set, val_set, test_set = random_split(
            full_dataset, 
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
            drop_last=True  # For consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        
        logger.info("Dataloaders created successfully")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    config = {
        'compute_stats': True,
        'stats_sample_size': 30,
        'augmentation': {
            'flip': True,
            'flip_axes': ('LR',),
            'flip_prob': 0.5,
            'affine': True,
            'affine_scales': 0.1,
            'affine_degrees': 10,
            'affine_translation': 5,
            'affine_prob': 0.75,
            'noise': True,
            'noise_std': 0.1,
            'noise_prob': 0.5,
            'bias_field': True,
            'bias_field_prob': 0.3,
            'blur': False,
            'gamma': False
        }
    }
    
    # Test dataset creation
    try:
        dataset = BraTSDataset(
            data_dir="/path/to/BraTS2021_Training_Data",
            augment=True,
            config=config
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        print(f"Dataset info: {dataset.get_dataset_info()}")
        
        # Test dataloader creation
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir="/path/to/BraTS2021_Training_Data",
            batch_size=2,
            config=config
        )
        
        print("Dataloaders created successfully")
        
    except Exception as e:
        print(f"Error: {e}")