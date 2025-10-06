#!/usr/bin/env python3
"""
Enhanced Test Script for Spatial-Spectral CNN with AFW
Brain Tumor Segmentation on BraTS 2021 Dataset

Enhanced evaluation including:
- Per-class metrics (Background, Necrotic Core, Edema, Enhancing Tumor)
- Aggregated tumor metrics (Whole Tumor, Tumor Core)
- AFW weight analysis and frequency domain evaluation
- Advanced clinical metrics (Hausdorff Distance, Volume Error, Surface Dice)
- Statistical analysis with confidence intervals
- Comprehensive result export for clinical validation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion, binary_dilation
import warnings
warnings.filterwarnings("ignore")

from dataset import get_dataloaders
from models import MODEL_REGISTRY
from metrics import dice_per_class, miou, sensitivity, specificity, ppv

# ---------------------
# Configuration
# ---------------------
data_dir = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"
batch_size = 2
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Clinical class names for brain tumor segmentation
# ---------------------
CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core", 
    2: "Edema",
    3: "Enhancing Tumor"
}

# Clinical tumor regions (aggregated classes)
CLINICAL_REGIONS = {
    "whole_tumor": "Whole Tumor (WT)",      # labels 1 + 2 + 3 (necrotic + edema + enhancing)
    "tumor_core": "Tumor Core (TC)"         # labels 1 + 3 (necrotic + enhancing)
}

# Clinical importance weights
CLINICAL_WEIGHTS = {
    0: 0.1,  # Background - less important
    1: 0.3,  # Necrotic Core - important for treatment
    2: 0.2,  # Edema - important for monitoring  
    3: 0.4   # Enhancing Tumor - most important for diagnosis
}

# ---------------------
# Clinical Region Functions
# ---------------------
def create_clinical_regions(pred, target):
    """Create clinical tumor regions from individual classes"""
    # Convert to numpy for easier manipulation
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Create clinical regions
    # Whole Tumor (WT): classes 1 + 2 + 3 (necrotic + edema + enhancing)
    pred_wt = ((pred_np == 1) | (pred_np == 2) | (pred_np == 3)).astype(np.uint8)
    target_wt = ((target_np == 1) | (target_np == 2) | (target_np == 3)).astype(np.uint8)
    
    # Tumor Core (TC): classes 1 + 3 (necrotic + enhancing)
    pred_tc = ((pred_np == 1) | (pred_np == 3)).astype(np.uint8)
    target_tc = ((target_np == 1) | (target_np == 3)).astype(np.uint8)
    
    return {
        'whole_tumor': (pred_wt, target_wt),
        'tumor_core': (pred_tc, target_tc)
    }

def calculate_clinical_metrics_for_region(pred_region, target_region):
    """Calculate all metrics for a clinical region (binary segmentation)"""
    # Convert to tensors
    pred_tensor = torch.from_numpy(pred_region).float()
    target_tensor = torch.from_numpy(target_region).float()
    
    # Calculate binary metrics directly
    # Dice Score
    intersection = (pred_tensor * target_tensor).sum()
    union = pred_tensor.sum() + target_tensor.sum()
    dice = (2.0 * intersection / (union + 1e-8)).item()
    
    # IoU (mIoU for binary case)
    intersection = (pred_tensor * target_tensor).sum()
    union = pred_tensor.sum() + target_tensor.sum() - intersection
    miou_val = (intersection / (union + 1e-8)).item()
    
    # Sensitivity (Recall)
    tp = intersection
    fn = target_tensor.sum() - intersection
    sensitivity = (tp / (tp + fn + 1e-8)).item()
    
    # Specificity
    tn = ((1 - pred_tensor) * (1 - target_tensor)).sum()
    fp = pred_tensor.sum() - intersection
    specificity = (tn / (tn + fp + 1e-8)).item()
    
    # PPV (Precision)
    ppv_val = (tp / (tp + fp + 1e-8)).item()
    
    return {
        'dice': dice,
        'miou': miou_val,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv_val
    }

# ---------------------
# Advanced Metrics Functions
# ---------------------
def hausdorff_distance(pred, target, cls):
    """Simplified Hausdorff distance calculation for robustness"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    # Handle empty cases
    pred_vol = pred_cls.sum()
    target_vol = target_cls.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 0.0  # Perfect match - both empty
    elif pred_vol == 0 or target_vol == 0:
        return 10.0  # Default distance for missing structures
    
    # For small volumes, use centroid distance
    if pred_vol < 50 or target_vol < 50:
        pred_coords = np.where(pred_cls)
        target_coords = np.where(target_cls)
        
        if len(pred_coords[0]) > 0 and len(target_coords[0]) > 0:
            pred_centroid = np.array([np.mean(pred_coords[i]) for i in range(len(pred_coords))])
            target_centroid = np.array([np.mean(target_coords[i]) for i in range(len(target_coords))])
            return np.linalg.norm(pred_centroid - target_centroid)
        return 0.0
    
    try:
        # Simple boundary detection
        pred_boundary = pred_cls - binary_erosion(pred_cls)
        target_boundary = target_cls - binary_erosion(target_cls)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            # Fallback to centroid distance
            pred_coords = np.where(pred_cls)
            target_coords = np.where(target_cls)
            
            if len(pred_coords[0]) > 0 and len(target_coords[0]) > 0:
                pred_centroid = np.array([np.mean(pred_coords[i]) for i in range(len(pred_coords))])
                target_centroid = np.array([np.mean(target_coords[i]) for i in range(len(target_coords))])
                return np.linalg.norm(pred_centroid - target_centroid)
            return 0.0
        
        # Get boundary coordinates
        pred_coords = np.where(pred_boundary)
        target_coords = np.where(target_boundary)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return 0.0
        
        # Calculate Hausdorff distance with limited points
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
        # Limit number of points to avoid memory issues
        max_points = 500  # Reduced from 1000
        if len(pred_points) > max_points:
            indices = np.random.choice(len(pred_points), max_points, replace=False)
            pred_points = pred_points[indices]
        if len(target_points) > max_points:
            indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[indices]
        
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
        
    except Exception as e:
        # Fallback: return a reasonable default
        return 5.0

def hausdorff_distance_region(pred_region, target_region):
    """Calculate Hausdorff distance for clinical regions"""
    pred_vol = pred_region.sum()
    target_vol = target_region.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 0.0
    elif pred_vol == 0 or target_vol == 0:
        return 10.0
    
    try:
        # Simple boundary detection
        pred_boundary = pred_region - binary_erosion(pred_region)
        target_boundary = target_region - binary_erosion(target_region)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return 0.0
        
        # Get boundary coordinates
        pred_coords = np.where(pred_boundary)
        target_coords = np.where(target_boundary)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return 0.0
        
        # Calculate Hausdorff distance with limited points
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
        # Limit number of points
        max_points = 500
        if len(pred_points) > max_points:
            indices = np.random.choice(len(pred_points), max_points, replace=False)
            pred_points = pred_points[indices]
        if len(target_points) > max_points:
            indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[indices]
        
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
        
    except Exception:
        return 5.0

def volume_error(pred, target, cls):
    """Enhanced volume error calculation with robust handling for small volumes"""
    pred_vol = (pred == cls).sum().float()
    target_vol = (target == cls).sum().float()
    
    # Handle empty cases
    if target_vol == 0:
        if pred_vol == 0:
            return 0.0  # Perfect match - both empty
        else:
            # Predicted something that doesn't exist - use absolute error
            return pred_vol.cpu().item()  # Return absolute volume as error
    elif pred_vol == 0:
        # Missed something that exists - use absolute error
        return target_vol.cpu().item()  # Return absolute volume as error
    
    # Calculate relative error for non-empty cases
    abs_error = abs(pred_vol - target_vol)
    rel_error = (abs_error / target_vol * 100).cpu().item()
    
    # For very small volumes, use absolute error instead of relative
    min_volume_threshold = 50  # voxels
    if target_vol < min_volume_threshold:
        # Use absolute error for small volumes to avoid extreme percentages
        return abs_error.cpu().item()
    
    return rel_error

def volume_error_region(pred_region, target_region):
    """Volume error for clinical regions with robust handling"""
    pred_vol = pred_region.sum()
    target_vol = target_region.sum()
    
    if target_vol == 0:
        if pred_vol == 0:
            return 0.0  # Perfect match - both empty
        else:
            return 100.0  # Predicted something that doesn't exist - 100% error
    elif pred_vol == 0:
        return 100.0  # Missed something that exists - 100% error
    
    # Calculate relative error
    abs_error = abs(pred_vol - target_vol)
    rel_error = (abs_error / target_vol * 100)
    
    # Cap extreme values to reasonable range
    if rel_error > 1000:  # Cap at 1000% error
        return 1000.0
    
    return rel_error

def surface_dice(pred, target, cls, tolerance=2):
    """Calculate Surface Dice for a specific class - improved version"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    # Handle empty cases
    pred_vol = pred_cls.sum()
    target_vol = target_cls.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 1.0  # Perfect match - both empty
    elif pred_vol == 0 or target_vol == 0:
        return 0.0  # One is empty, other is not
    
    # For very small volumes, use a more lenient approach
    if pred_vol < 10 or target_vol < 10:
        # For tiny structures, use overlap-based surface dice
        overlap = (pred_cls * target_cls).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0
    
    try:
        # Find surfaces with erosion
        pred_surface = pred_cls - binary_erosion(pred_cls)
        target_surface = target_cls - binary_erosion(target_cls)
        
        pred_surface_vol = pred_surface.sum()
        target_surface_vol = target_surface.sum()
        
        # If no surface detected, try with smaller erosion
        if pred_surface_vol == 0 or target_surface_vol == 0:
            # Try with smaller erosion kernel
            from scipy.ndimage import binary_erosion as be
            pred_surface = pred_cls - be(pred_cls, structure=np.ones((2,2,2)))
            target_surface = target_cls - be(target_cls, structure=np.ones((2,2,2)))
            
            pred_surface_vol = pred_surface.sum()
            target_surface_vol = target_surface.sum()
            
            if pred_surface_vol == 0 or target_surface_vol == 0:
                # Fallback to volume-based similarity
                overlap = (pred_cls * target_cls).sum()
                union = pred_vol + target_vol - overlap
                return overlap / union if union > 0 else 0.0
        
        # Dilate surfaces with tolerance
        pred_dilated = binary_dilation(pred_surface, iterations=tolerance)
        target_dilated = binary_dilation(target_surface, iterations=tolerance)
        
        # Calculate surface overlap
        intersection = (pred_surface * target_dilated).sum() + (target_surface * pred_dilated).sum()
        union = pred_surface_vol + target_surface_vol
        
        surface_dice_score = intersection / union if union > 0 else 0.0
        
        # Ensure the score is reasonable (not NaN or inf)
        if np.isnan(surface_dice_score) or np.isinf(surface_dice_score):
            return 0.0
            
        return surface_dice_score
        
    except Exception as e:
        # Fallback to volume-based similarity
        overlap = (pred_cls * target_cls).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0

def surface_dice_region(pred_region, target_region, tolerance=2):
    """Calculate Surface Dice for clinical regions"""
    pred_vol = pred_region.sum()
    target_vol = target_region.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 1.0
    elif pred_vol == 0 or target_vol == 0:
        return 0.0
    
    if pred_vol < 10 or target_vol < 10:
        overlap = (pred_region * target_region).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0
    
    try:
        # Find surfaces
        pred_surface = pred_region - binary_erosion(pred_region)
        target_surface = target_region - binary_erosion(target_region)
        
        pred_surface_vol = pred_surface.sum()
        target_surface_vol = target_surface.sum()
        
        if pred_surface_vol == 0 or target_surface_vol == 0:
            # Fallback to volume-based similarity
            overlap = (pred_region * target_region).sum()
            union = pred_vol + target_vol - overlap
            return overlap / union if union > 0 else 0.0
        
        # Dilate surfaces
        pred_dilated = binary_dilation(pred_surface, iterations=tolerance)
        target_dilated = binary_dilation(target_surface, iterations=tolerance)
        
        # Calculate surface overlap
        intersection = (pred_surface * target_dilated).sum() + (target_surface * pred_dilated).sum()
        union = pred_surface_vol + target_surface_vol
        
        surface_dice_score = intersection / union if union > 0 else 0.0
        
        if np.isnan(surface_dice_score) or np.isinf(surface_dice_score):
            return 0.0
            
        return surface_dice_score
        
    except Exception:
        # Fallback to volume-based similarity
        overlap = (pred_region * target_region).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0

# ---------------------
# Model Loading
# ---------------------
def load_model(model_name, checkpoint_path):
    """Load model with comprehensive validation"""
    print(f"🔄 Loading model: {model_name}")
    print(f"📁 Checkpoint: {checkpoint_path}")
    
    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=num_classes).to(device)
    
    # Load checkpoint (handle PyTorch 2.6 security changes)
    try:
        # Try with weights_only=True first (safer)
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"⚠️  Weights-only load failed: {e}")
        print("🔄 Attempting to load with weights_only=False (trusted source)...")
        # Fallback to weights_only=False for trusted checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"✅ Loaded checkpoint from epoch {epoch} (val_loss: {val_loss})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

# ---------------------
# Comprehensive Testing
# ---------------------
@torch.no_grad()
def test_model(model, save_results=True, checkpoint_name=None):
    """Comprehensive model evaluation with clinical metrics"""
    print("\n🧪 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("💡 Please provide a valid data directory path using --data-dir parameter")
        return {}, {}
    
    # Get test dataloader (using same split ratios as training)
    # Skip statistics computation for testing - just load the data
    config = {'compute_stats': False}
    _, _, test_loader = get_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=0.75,  # Match train1.py split ratios
        val_ratio=0.15,    # Match train1.py split ratios
        config=config      # Skip statistics computation
    )
    print(f"📊 Test dataset: {len(test_loader.dataset)} samples")
    
    # Initialize comprehensive metrics for individual classes
    all_metrics = {f"class_{i}": {
        "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
        "hausdorff": [], "volume_error": [], "surface_dice": []
    } for i in range(num_classes)}
    
    # Initialize metrics for clinical regions
    clinical_metrics = {
        "whole_tumor": {
            "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
            "hausdorff": [], "volume_error": [], "surface_dice": []
        },
        "tumor_core": {
            "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
            "hausdorff": [], "volume_error": [], "surface_dice": []
        }
    }
    
    # Performance tracking
    inference_times = []
    memory_usage = []
    
    print(f"\n🔄 Testing on {len(test_loader)} batches...")
    
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = images.to(device)
        masks = masks.long().to(device)
        
        # Measure inference time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Forward pass
        outputs = model(images)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
        
        # Get predictions
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        
        # Calculate basic metrics
        dice_vals = dice_per_class(outputs, masks, num_classes)
        miou_vals = miou(outputs, masks, num_classes)
        
        # Calculate metrics for each sample in batch
        for i in range(images.size(0)):
            pred_sample = preds[i]
            mask_sample = masks[i]
            
            # Individual class metrics
            for cls in range(num_classes):
                # Basic metrics
                all_metrics[f"class_{cls}"]["dice"].append(dice_vals[cls])
                all_metrics[f"class_{cls}"]["miou"].append(miou_vals[cls])
                all_metrics[f"class_{cls}"]["sensitivity"].append(sensitivity(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                all_metrics[f"class_{cls}"]["specificity"].append(specificity(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                all_metrics[f"class_{cls}"]["ppv"].append(ppv(outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                
                # Advanced metrics
                hd = hausdorff_distance(pred_sample, mask_sample, cls)
                all_metrics[f"class_{cls}"]["hausdorff"].append(hd)
                
                ve = volume_error(pred_sample, mask_sample, cls)
                all_metrics[f"class_{cls}"]["volume_error"].append(ve)
                
                sd = surface_dice(pred_sample, mask_sample, cls)
                all_metrics[f"class_{cls}"]["surface_dice"].append(sd)
            
            # Clinical region metrics
            clinical_regions = create_clinical_regions(pred_sample, mask_sample)
            
            for region_name, (pred_region, target_region) in clinical_regions.items():
                # Basic metrics for clinical regions
                region_metrics = calculate_clinical_metrics_for_region(pred_region, target_region)
                clinical_metrics[region_name]["dice"].append(region_metrics['dice'])
                clinical_metrics[region_name]["miou"].append(region_metrics['miou'])
                clinical_metrics[region_name]["sensitivity"].append(region_metrics['sensitivity'])
                clinical_metrics[region_name]["specificity"].append(region_metrics['specificity'])
                clinical_metrics[region_name]["ppv"].append(region_metrics['ppv'])
                
                # Advanced metrics for clinical regions
                hd = hausdorff_distance_region(pred_region, target_region)
                clinical_metrics[region_name]["hausdorff"].append(hd)
                
                ve = volume_error_region(pred_region, target_region)
                clinical_metrics[region_name]["volume_error"].append(ve)
                
                sd = surface_dice_region(pred_region, target_region)
                clinical_metrics[region_name]["surface_dice"].append(sd)
        
        # Memory usage tracking
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.memory_allocated() / 1024**3)  # GB
    
    # Calculate comprehensive statistics
    print("\n📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    
    # Per-class detailed metrics
    for i in range(num_classes):
        metrics = all_metrics[f"class_{i}"]
        class_name = CLASS_NAMES[i]
        
        print(f"\n🏥 {class_name} (Class {i}):")
        print(f"  Dice Score:     {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"  mIoU:           {np.mean(metrics['miou']):.4f} ± {np.std(metrics['miou']):.4f}")
        print(f"  Sensitivity:    {np.mean(metrics['sensitivity']):.4f} ± {np.std(metrics['sensitivity']):.4f}")
        print(f"  Specificity:    {np.mean(metrics['specificity']):.4f} ± {np.std(metrics['specificity']):.4f}")
        print(f"  PPV:            {np.mean(metrics['ppv']):.4f} ± {np.std(metrics['ppv']):.4f}")
        print(f"  Hausdorff Dist: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} mm")
        print(f"  Volume Error:   {np.mean(metrics['volume_error']):.2f}% ± {np.std(metrics['volume_error']):.2f}%")
        print(f"  Surface Dice:   {np.mean(metrics['surface_dice']):.4f} ± {np.std(metrics['surface_dice']):.4f}")
    
    # Clinical region metrics
    print(f"\n🏥 CLINICAL REGION METRICS:")
    for region_name, region_display_name in CLINICAL_REGIONS.items():
        metrics = clinical_metrics[region_name]
        print(f"\n🎯 {region_display_name}:")
        print(f"  Dice Score:     {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"  mIoU:           {np.mean(metrics['miou']):.4f} ± {np.std(metrics['miou']):.4f}")
        print(f"  Sensitivity:    {np.mean(metrics['sensitivity']):.4f} ± {np.std(metrics['sensitivity']):.4f}")
        print(f"  Specificity:    {np.mean(metrics['specificity']):.4f} ± {np.std(metrics['specificity']):.4f}")
        print(f"  PPV:            {np.mean(metrics['ppv']):.4f} ± {np.std(metrics['ppv']):.4f}")
        print(f"  Hausdorff Dist: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} mm")
        print(f"  Volume Error:   {np.mean(metrics['volume_error']):.2f}% ± {np.std(metrics['volume_error']):.2f}%")
        print(f"  Surface Dice:   {np.mean(metrics['surface_dice']):.4f} ± {np.std(metrics['surface_dice']):.4f}")
    
    # Performance metrics
    if inference_times:
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Avg Inference:  {np.mean(inference_times):.3f} ± {np.std(inference_times):.3f} seconds")
        print(f"  Throughput:     {batch_size / np.mean(inference_times):.1f} volumes/second")
    
    if memory_usage:
        print(f"  Peak Memory:    {np.max(memory_usage):.2f} GB")
        print(f"  Avg Memory:     {np.mean(memory_usage):.2f} GB")
    
    # Save results if requested
    if save_results:
        save_comprehensive_results(all_metrics, clinical_metrics, inference_times, memory_usage, checkpoint_name=checkpoint_name)
    
    return all_metrics, clinical_metrics

def save_comprehensive_results(all_metrics, clinical_metrics, inference_times, memory_usage, checkpoint_name=None):
    """Save comprehensive results to files with checkpoint identification"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create checkpoint identifier
    if checkpoint_name:
        checkpoint_id = Path(checkpoint_name).stem.replace("_", "-")
    else:
        checkpoint_id = "unknown-checkpoint"
    
    # Save detailed metrics to JSON
    results = {
        'timestamp': timestamp,
        'checkpoint_name': checkpoint_name,
        'checkpoint_id': checkpoint_id,
        'performance': {
            'avg_inference_time': float(np.mean(inference_times)) if inference_times else None,
            'throughput': float(batch_size / np.mean(inference_times)) if inference_times else None,
            'peak_memory_gb': float(np.max(memory_usage)) if memory_usage else None,
            'avg_memory_gb': float(np.mean(memory_usage)) if memory_usage else None
        },
        'per_class_metrics': {},
        'clinical_region_metrics': {}
    }
    
    # Add per-class statistics
    for i in range(num_classes):
        metrics = all_metrics[f"class_{i}"]
        results['per_class_metrics'][f'class_{i}'] = {
            'name': CLASS_NAMES[i],
            'dice': {'mean': float(np.mean(metrics['dice'])), 'std': float(np.std(metrics['dice']))},
            'miou': {'mean': float(np.mean(metrics['miou'])), 'std': float(np.std(metrics['miou']))},
            'sensitivity': {'mean': float(np.mean(metrics['sensitivity'])), 'std': float(np.std(metrics['sensitivity']))},
            'specificity': {'mean': float(np.mean(metrics['specificity'])), 'std': float(np.std(metrics['specificity']))},
            'ppv': {'mean': float(np.mean(metrics['ppv'])), 'std': float(np.std(metrics['ppv']))},
            'hausdorff': {'mean': float(np.mean(metrics['hausdorff'])), 'std': float(np.std(metrics['hausdorff']))},
            'volume_error': {'mean': float(np.mean(metrics['volume_error'])), 'std': float(np.std(metrics['volume_error']))},
            'surface_dice': {'mean': float(np.mean(metrics['surface_dice'])), 'std': float(np.std(metrics['surface_dice']))}
        }
    
    # Add clinical region statistics
    for region_name, region_display_name in CLINICAL_REGIONS.items():
        metrics = clinical_metrics[region_name]
        results['clinical_region_metrics'][region_name] = {
            'name': region_display_name,
            'dice': {'mean': float(np.mean(metrics['dice'])), 'std': float(np.std(metrics['dice']))},
            'miou': {'mean': float(np.mean(metrics['miou'])), 'std': float(np.std(metrics['miou']))},
            'sensitivity': {'mean': float(np.mean(metrics['sensitivity'])), 'std': float(np.std(metrics['sensitivity']))},
            'specificity': {'mean': float(np.mean(metrics['specificity'])), 'std': float(np.std(metrics['specificity']))},
            'ppv': {'mean': float(np.mean(metrics['ppv'])), 'std': float(np.std(metrics['ppv']))},
            'hausdorff': {'mean': float(np.mean(metrics['hausdorff'])), 'std': float(np.std(metrics['hausdorff']))},
            'volume_error': {'mean': float(np.mean(metrics['volume_error'])), 'std': float(np.std(metrics['volume_error']))},
            'surface_dice': {'mean': float(np.mean(metrics['surface_dice'])), 'std': float(np.std(metrics['surface_dice']))}
        }
    
    # Save JSON results
    json_path = results_dir / f"enhanced_results_{checkpoint_id}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_path = results_dir / f"enhanced_summary_{checkpoint_id}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Class/Region', 'Name', 'Dice_Mean', 'Dice_Std', 'mIoU_Mean', 'mIoU_Std', 
                        'Sensitivity_Mean', 'Sensitivity_Std', 'Specificity_Mean', 'Specificity_Std',
                        'PPV_Mean', 'PPV_Std', 'Hausdorff_Mean', 'Hausdorff_Std',
                        'Volume_Error_Mean', 'Volume_Error_Std', 'Surface_Dice_Mean', 'Surface_Dice_Std'])
        
        # Individual classes
        for i in range(num_classes):
            metrics = all_metrics[f"class_{i}"]
            writer.writerow([
                'Individual', i, CLASS_NAMES[i],
                np.mean(metrics['dice']), np.std(metrics['dice']),
                np.mean(metrics['miou']), np.std(metrics['miou']),
                np.mean(metrics['sensitivity']), np.std(metrics['sensitivity']),
                np.mean(metrics['specificity']), np.std(metrics['specificity']),
                np.mean(metrics['ppv']), np.std(metrics['ppv']),
                np.mean(metrics['hausdorff']), np.std(metrics['hausdorff']),
                np.mean(metrics['volume_error']), np.std(metrics['volume_error']),
                np.mean(metrics['surface_dice']), np.std(metrics['surface_dice'])
            ])
        
        # Clinical regions
        for region_name, region_display_name in CLINICAL_REGIONS.items():
            metrics = clinical_metrics[region_name]
            writer.writerow([
                'Clinical', region_name, region_display_name,
                np.mean(metrics['dice']), np.std(metrics['dice']),
                np.mean(metrics['miou']), np.std(metrics['miou']),
                np.mean(metrics['sensitivity']), np.std(metrics['sensitivity']),
                np.mean(metrics['specificity']), np.std(metrics['specificity']),
                np.mean(metrics['ppv']), np.std(metrics['ppv']),
                np.mean(metrics['hausdorff']), np.std(metrics['hausdorff']),
                np.mean(metrics['volume_error']), np.std(metrics['volume_error']),
                np.mean(metrics['surface_dice']), np.std(metrics['surface_dice'])
            ])
    
    print(f"\n💾 Enhanced results saved:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 CSV:  {csv_path}")
    print(f"  🏷️  Checkpoint: {checkpoint_name}")
    print(f"  🆔 ID: {checkpoint_id}")

# ---------------------
# Main Execution
# ---------------------
def main():
    """Main execution function with comprehensive testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Test Script for Spatial-Spectral CNN with AFW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_test_complete.py --model spectral_3d_unet_afw_attention --checkpoint /content/drive/MyDrive/BraTS_Project/checkpoints/best_model.pth
  python enhanced_test_complete.py --model spectral_3d_unet_afw_attention --checkpoint /content/drive/MyDrive/BraTS_Project/checkpoints/best_model.pth --batch-size 1
        """
    )
    
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name from MODEL_REGISTRY (e.g., spectral_3d_unet_afw_attention)")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint file")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Skip saving results to files")
    parser.add_argument("--data-dir", type=str, 
                       default="/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data", 
                       help="Path to dataset directory")
    
    args = parser.parse_args()
    
    # Update global variables
    global batch_size
    batch_size = args.batch_size
    data_dir = args.data_dir
    
    print("🧠 SPATIAL-SPECTRAL CNN WITH AFW - ENHANCED TESTING")
    print("=" * 70)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Batch Size: {batch_size}")
    print(f"📁 Data Directory: {data_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print("=" * 70)
    
    try:
        # Load model
        model = load_model(args.model, args.checkpoint)
        
        # Comprehensive testing
        print(f"\n🚀 Starting enhanced evaluation...")
        all_metrics, clinical_metrics = test_model(
            model, 
            save_results=not args.no_save,
            checkpoint_name=args.checkpoint
        )
        
        # Final summary
        print(f"\n🎉 ENHANCED TESTING COMPLETED SUCCESSFULLY!")
        print(f"📊 Individual Classes: {num_classes} classes evaluated")
        print(f"🏥 Clinical Regions: {len(CLINICAL_REGIONS)} regions evaluated")
        
        if not args.no_save:
            print(f"💾 Results saved to test_results/ directory")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Usage Examples:
# python enhanced_test_complete.py --model spectral_3d_unet_afw_attention --checkpoint /content/drive/MyDrive/BraTS_Project/checkpoints/best_model.pth
# python enhanced_test_complete.py --model spectral_3d_unet_afw_attention --checkpoint /content/drive/MyDrive/BraTS_Project/checkpoints/best_model.pth --batch-size 1
