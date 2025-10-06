#!/usr/bin/env python3
"""
Ensemble Prediction Testing Script for Spatial-Spectral CNN with AFW
Brain Tumor Segmentation on BraTS 2021 Dataset

True ensemble method that combines predictions from multiple models with weighted voting:
- Loads multiple checkpoints
- Combines predictions with weighted voting
- Evaluates ensemble performance
- Provides comprehensive metrics for ensemble vs individual models
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
import argparse
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

# Clinical class names
CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core", 
    2: "Edema",
    3: "Enhancing Tumor"
}

# Clinical tumor regions
CLINICAL_REGIONS = {
    "whole_tumor": "Whole Tumor (WT)",      # labels 1 + 2 + 3
    "tumor_core": "Tumor Core (TC)"         # labels 1 + 3
}

# ---------------------
# Ensemble Model Class
# ---------------------
class EnsembleModel(nn.Module):
    """Ensemble model that combines multiple models with weighted voting"""
    
    def __init__(self, models, weights):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.weights = self.weights / self.weights.sum()  # Normalize weights
        
    def forward(self, x):
        """Forward pass with weighted ensemble voting"""
        ensemble_output = None
        
        for i, model in enumerate(self.models):
            with torch.no_grad():
                output = model(x)
                weighted_output = output * self.weights[i]
                
                if ensemble_output is None:
                    ensemble_output = weighted_output
                else:
                    ensemble_output += weighted_output
        
        return ensemble_output

# ---------------------
# Clinical Region Functions
# ---------------------
def create_clinical_regions(pred, target):
    """Create clinical tumor regions from individual classes"""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
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
    """Simplified Hausdorff distance calculation"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    pred_vol = pred_cls.sum()
    target_vol = target_cls.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 0.0
    elif pred_vol == 0 or target_vol == 0:
        return 10.0
    
    if pred_vol < 50 or target_vol < 50:
        pred_coords = np.where(pred_cls)
        target_coords = np.where(target_cls)
        
        if len(pred_coords[0]) > 0 and len(target_coords[0]) > 0:
            pred_centroid = np.array([np.mean(pred_coords[i]) for i in range(len(pred_coords))])
            target_centroid = np.array([np.mean(target_coords[i]) for i in range(len(target_coords))])
            return np.linalg.norm(pred_centroid - target_centroid)
        return 0.0
    
    try:
        pred_boundary = pred_cls - binary_erosion(pred_cls)
        target_boundary = target_cls - binary_erosion(target_cls)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            pred_coords = np.where(pred_cls)
            target_coords = np.where(target_cls)
            
            if len(pred_coords[0]) > 0 and len(target_coords[0]) > 0:
                pred_centroid = np.array([np.mean(pred_coords[i]) for i in range(len(pred_coords))])
                target_centroid = np.array([np.mean(target_coords[i]) for i in range(len(target_coords))])
                return np.linalg.norm(pred_centroid - target_centroid)
            return 0.0
        
        pred_coords = np.where(pred_boundary)
        target_coords = np.where(target_boundary)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return 0.0
        
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
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

def hausdorff_distance_region(pred_region, target_region):
    """Calculate Hausdorff distance for clinical regions"""
    pred_vol = pred_region.sum()
    target_vol = target_region.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 0.0
    elif pred_vol == 0 or target_vol == 0:
        return 10.0
    
    try:
        pred_boundary = pred_region - binary_erosion(pred_region)
        target_boundary = target_region - binary_erosion(target_region)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return 0.0
        
        pred_coords = np.where(pred_boundary)
        target_coords = np.where(target_boundary)
        
        if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
            return 0.0
        
        pred_points = np.column_stack(pred_coords)
        target_points = np.column_stack(target_coords)
        
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
    """Volume error calculation"""
    pred_vol = (pred == cls).sum().float()
    target_vol = (target == cls).sum().float()
    
    if target_vol == 0:
        if pred_vol == 0:
            return 0.0
        else:
            return pred_vol.cpu().item()
    elif pred_vol == 0:
        return target_vol.cpu().item()
    
    abs_error = abs(pred_vol - target_vol)
    rel_error = (abs_error / target_vol * 100).cpu().item()
    
    min_volume_threshold = 50
    if target_vol < min_volume_threshold:
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
    """Surface Dice calculation"""
    pred_cls = (pred == cls).cpu().numpy()
    target_cls = (target == cls).cpu().numpy()
    
    pred_vol = pred_cls.sum()
    target_vol = target_cls.sum()
    
    if pred_vol == 0 and target_vol == 0:
        return 1.0
    elif pred_vol == 0 or target_vol == 0:
        return 0.0
    
    if pred_vol < 10 or target_vol < 10:
        overlap = (pred_cls * target_cls).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0
    
    try:
        pred_surface = pred_cls - binary_erosion(pred_cls)
        target_surface = target_cls - binary_erosion(target_cls)
        
        pred_surface_vol = pred_surface.sum()
        target_surface_vol = target_surface.sum()
        
        if pred_surface_vol == 0 or target_surface_vol == 0:
            from scipy.ndimage import binary_erosion as be
            pred_surface = pred_cls - be(pred_cls, structure=np.ones((2,2,2)))
            target_surface = target_cls - be(target_cls, structure=np.ones((2,2,2)))
            
            pred_surface_vol = pred_surface.sum()
            target_surface_vol = target_surface.sum()
            
            if pred_surface_vol == 0 or target_surface_vol == 0:
                overlap = (pred_cls * target_cls).sum()
                union = pred_vol + target_vol - overlap
                return overlap / union if union > 0 else 0.0
        
        pred_dilated = binary_dilation(pred_surface, iterations=tolerance)
        target_dilated = binary_dilation(target_surface, iterations=tolerance)
        
        intersection = (pred_surface * target_dilated).sum() + (target_surface * pred_dilated).sum()
        union = pred_surface_vol + target_surface_vol
        
        surface_dice_score = intersection / union if union > 0 else 0.0
        
        if np.isnan(surface_dice_score) or np.isinf(surface_dice_score):
            return 0.0
            
        return surface_dice_score
        
    except Exception:
        overlap = (pred_cls * target_cls).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0

def surface_dice_region(pred_region, target_region, tolerance=2):
    """Surface Dice for clinical regions"""
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
        pred_surface = pred_region - binary_erosion(pred_region)
        target_surface = target_region - binary_erosion(target_region)
        
        pred_surface_vol = pred_surface.sum()
        target_surface_vol = target_surface.sum()
        
        if pred_surface_vol == 0 or target_surface_vol == 0:
            overlap = (pred_region * target_region).sum()
            union = pred_vol + target_vol - overlap
            return overlap / union if union > 0 else 0.0
        
        pred_dilated = binary_dilation(pred_surface, iterations=tolerance)
        target_dilated = binary_dilation(target_surface, iterations=tolerance)
        
        intersection = (pred_surface * target_dilated).sum() + (target_surface * pred_dilated).sum()
        union = pred_surface_vol + target_surface_vol
        
        surface_dice_score = intersection / union if union > 0 else 0.0
        
        if np.isnan(surface_dice_score) or np.isinf(surface_dice_score):
            return 0.0
            
        return surface_dice_score
        
    except Exception:
        overlap = (pred_region * target_region).sum()
        union = pred_vol + target_vol - overlap
        return overlap / union if union > 0 else 0.0

# ---------------------
# Model Loading
# ---------------------
def load_model(model_name, checkpoint_path):
    """Load a single model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(in_channels=4, out_channels=num_classes).to(device)
    
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"⚠️  Weights-only load failed for {checkpoint_path}: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"✅ Loaded {os.path.basename(checkpoint_path)} from epoch {epoch} (val_loss: {val_loss})")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded {os.path.basename(checkpoint_path)} weights")
    
    model.eval()
    return model

def load_ensemble_models(checkpoint_paths, model_name="spectral_3d_unet_afw_attention"):
    """Load multiple models for ensemble"""
    models = []
    
    print(f"🔄 Loading {len(checkpoint_paths)} models for ensemble...")
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n📁 Model {i+1}: {checkpoint_path}")
        model = load_model(model_name, checkpoint_path)
        models.append(model)
    
    return models

# ---------------------
# Ensemble Testing
# ---------------------
@torch.no_grad()
def test_ensemble(ensemble_model, individual_models, checkpoint_paths, weights, test_loader):
    """Test ensemble model and individual models"""
    print(f"\n🧪 ENSEMBLE MODEL EVALUATION")
    print("=" * 60)
    
    # Initialize metrics for ensemble
    ensemble_metrics = {f"class_{i}": {
        "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
        "hausdorff": [], "volume_error": [], "surface_dice": []
    } for i in range(num_classes)}
    
    ensemble_clinical_metrics = {
        "whole_tumor": {
            "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
            "hausdorff": [], "volume_error": [], "surface_dice": []
        },
        "tumor_core": {
            "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
            "hausdorff": [], "volume_error": [], "surface_dice": []
        }
    }
    
    # Initialize metrics for individual models
    individual_results = {}
    for i, (model, checkpoint_path) in enumerate(zip(individual_models, checkpoint_paths)):
        individual_results[f"model_{i+1}"] = {
            "checkpoint": os.path.basename(checkpoint_path),
            "weight": weights[i],
            "all_metrics": {f"class_{j}": {
                "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
                "hausdorff": [], "volume_error": [], "surface_dice": []
            } for j in range(num_classes)},
            "clinical_metrics": {
                "whole_tumor": {
                    "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
                    "hausdorff": [], "volume_error": [], "surface_dice": []
                },
                "tumor_core": {
                    "dice": [], "miou": [], "sensitivity": [], "specificity": [], "ppv": [],
                    "hausdorff": [], "volume_error": [], "surface_dice": []
                }
            }
        }
    
    # Performance tracking
    ensemble_inference_times = []
    
    print(f"\n🔄 Testing ensemble and individual models on {len(test_loader)} batches...")
    
    for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating Ensemble")):
        images = images.to(device)
        masks = masks.long().to(device)
        
        # Measure ensemble inference time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Ensemble forward pass
        ensemble_outputs = ensemble_model(images)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            ensemble_inference_times.append(start_time.elapsed_time(end_time) / 1000.0)
        
        # Get ensemble predictions
        ensemble_preds = torch.argmax(F.softmax(ensemble_outputs, dim=1), dim=1)
        
        # Calculate ensemble metrics
        ensemble_dice_vals = dice_per_class(ensemble_outputs, masks, num_classes)
        ensemble_miou_vals = miou(ensemble_outputs, masks, num_classes)
        
        # Individual model predictions and metrics
        individual_outputs = []
        for model in individual_models:
            with torch.no_grad():
                output = model(images)
                individual_outputs.append(output)
        
        # Calculate metrics for each sample in batch
        for i in range(images.size(0)):
            mask_sample = masks[i]
            
            # Ensemble metrics
            for cls in range(num_classes):
                ensemble_metrics[f"class_{cls}"]["dice"].append(ensemble_dice_vals[cls])
                ensemble_metrics[f"class_{cls}"]["miou"].append(ensemble_miou_vals[cls])
                ensemble_metrics[f"class_{cls}"]["sensitivity"].append(sensitivity(ensemble_outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                ensemble_metrics[f"class_{cls}"]["specificity"].append(specificity(ensemble_outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                ensemble_metrics[f"class_{cls}"]["ppv"].append(ppv(ensemble_outputs[i:i+1], mask_sample.unsqueeze(0), cls).item())
                
                # Advanced metrics for ensemble
                hd = hausdorff_distance(ensemble_preds[i], mask_sample, cls)
                ensemble_metrics[f"class_{cls}"]["hausdorff"].append(hd)
                
                ve = volume_error(ensemble_preds[i], mask_sample, cls)
                ensemble_metrics[f"class_{cls}"]["volume_error"].append(ve)
                
                sd = surface_dice(ensemble_preds[i], mask_sample, cls)
                ensemble_metrics[f"class_{cls}"]["surface_dice"].append(sd)
            
            # Ensemble clinical region metrics
            ensemble_clinical_regions = create_clinical_regions(ensemble_preds[i], mask_sample)
            
            for region_name, (pred_region, target_region) in ensemble_clinical_regions.items():
                region_metrics = calculate_clinical_metrics_for_region(pred_region, target_region)
                ensemble_clinical_metrics[region_name]["dice"].append(region_metrics['dice'])
                ensemble_clinical_metrics[region_name]["miou"].append(region_metrics['miou'])
                ensemble_clinical_metrics[region_name]["sensitivity"].append(region_metrics['sensitivity'])
                ensemble_clinical_metrics[region_name]["specificity"].append(region_metrics['specificity'])
                ensemble_clinical_metrics[region_name]["ppv"].append(region_metrics['ppv'])
                
                hd = hausdorff_distance_region(pred_region, target_region)
                ensemble_clinical_metrics[region_name]["hausdorff"].append(hd)
                
                ve = volume_error_region(pred_region, target_region)
                ensemble_clinical_metrics[region_name]["volume_error"].append(ve)
                
                sd = surface_dice_region(pred_region, target_region)
                ensemble_clinical_metrics[region_name]["surface_dice"].append(sd)
            
            # Individual model metrics
            for model_idx, (output, model_name) in enumerate(zip(individual_outputs, individual_results.keys())):
                pred_sample = torch.argmax(F.softmax(output[i:i+1], dim=1), dim=1)[0]
                
                dice_vals = dice_per_class(output[i:i+1], mask_sample.unsqueeze(0), num_classes)
                miou_vals = miou(output[i:i+1], mask_sample.unsqueeze(0), num_classes)
                
                for cls in range(num_classes):
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["dice"].append(dice_vals[cls])
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["miou"].append(miou_vals[cls])
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["sensitivity"].append(sensitivity(output[i:i+1], mask_sample.unsqueeze(0), cls).item())
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["specificity"].append(specificity(output[i:i+1], mask_sample.unsqueeze(0), cls).item())
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["ppv"].append(ppv(output[i:i+1], mask_sample.unsqueeze(0), cls).item())
                    
                    hd = hausdorff_distance(pred_sample, mask_sample, cls)
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["hausdorff"].append(hd)
                    
                    ve = volume_error(pred_sample, mask_sample, cls)
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["volume_error"].append(ve)
                    
                    sd = surface_dice(pred_sample, mask_sample, cls)
                    individual_results[model_name]["all_metrics"][f"class_{cls}"]["surface_dice"].append(sd)
                
                # Individual model clinical region metrics
                individual_clinical_regions = create_clinical_regions(pred_sample, mask_sample)
                
                for region_name, (pred_region, target_region) in individual_clinical_regions.items():
                    region_metrics = calculate_clinical_metrics_for_region(pred_region, target_region)
                    individual_results[model_name]["clinical_metrics"][region_name]["dice"].append(region_metrics['dice'])
                    individual_results[model_name]["clinical_metrics"][region_name]["miou"].append(region_metrics['miou'])
                    individual_results[model_name]["clinical_metrics"][region_name]["sensitivity"].append(region_metrics['sensitivity'])
                    individual_results[model_name]["clinical_metrics"][region_name]["specificity"].append(region_metrics['specificity'])
                    individual_results[model_name]["clinical_metrics"][region_name]["ppv"].append(region_metrics['ppv'])
                    
                    hd = hausdorff_distance_region(pred_region, target_region)
                    individual_results[model_name]["clinical_metrics"][region_name]["hausdorff"].append(hd)
                    
                    ve = volume_error_region(pred_region, target_region)
                    individual_results[model_name]["clinical_metrics"][region_name]["volume_error"].append(ve)
                    
                    sd = surface_dice_region(pred_region, target_region)
                    individual_results[model_name]["clinical_metrics"][region_name]["surface_dice"].append(sd)
    
    return ensemble_metrics, ensemble_clinical_metrics, individual_results, ensemble_inference_times

def display_results(ensemble_metrics, ensemble_clinical_metrics, individual_results, weights, checkpoint_paths):
    """Display comprehensive results"""
    print(f"\n📊 ENSEMBLE EVALUATION RESULTS")
    print("=" * 80)
    
    # Ensemble weights
    print(f"\n🎯 Ensemble Configuration:")
    for i, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
        print(f"   Model {i+1}: {os.path.basename(checkpoint_path)} (Weight: {weight:.1f})")
    
    # Ensemble results
    print(f"\n🏆 ENSEMBLE MODEL RESULTS:")
    print("=" * 60)
    
    # Individual class results
    for i in range(num_classes):
        metrics = ensemble_metrics[f"class_{i}"]
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
    
    # Clinical region results
    print(f"\n🏥 ENSEMBLE CLINICAL REGION METRICS:")
    for region_name, region_display_name in CLINICAL_REGIONS.items():
        metrics = ensemble_clinical_metrics[region_name]
        print(f"\n🎯 {region_display_name}:")
        print(f"  Dice Score:     {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"  mIoU:           {np.mean(metrics['miou']):.4f} ± {np.std(metrics['miou']):.4f}")
        print(f"  Sensitivity:    {np.mean(metrics['sensitivity']):.4f} ± {np.std(metrics['sensitivity']):.4f}")
        print(f"  Specificity:    {np.mean(metrics['specificity']):.4f} ± {np.std(metrics['specificity']):.4f}")
        print(f"  PPV:            {np.mean(metrics['ppv']):.4f} ± {np.std(metrics['ppv']):.4f}")
        print(f"  Hausdorff Dist: {np.mean(metrics['hausdorff']):.2f} ± {np.std(metrics['hausdorff']):.2f} mm")
        print(f"  Volume Error:   {np.mean(metrics['volume_error']):.2f}% ± {np.std(metrics['volume_error']):.2f}%")
        print(f"  Surface Dice:   {np.mean(metrics['surface_dice']):.4f} ± {np.std(metrics['surface_dice']):.4f}")
    
    # Individual model comparison
    print(f"\n📊 INDIVIDUAL MODEL COMPARISON:")
    print("=" * 60)
    
    for model_name, results in individual_results.items():
        checkpoint_name = results['checkpoint']
        weight = results['weight']
        
        print(f"\n🤖 {model_name} ({checkpoint_name}) - Weight: {weight:.1f}:")
        
        # Show Dice scores for each class
        for i in range(num_classes):
            metrics = results['all_metrics'][f"class_{i}"]
            class_name = CLASS_NAMES[i]
            dice_mean = np.mean(metrics['dice'])
            print(f"   {class_name}: Dice = {dice_mean:.4f}")
        
        # Show clinical region Dice scores
        for region_name, region_display_name in CLINICAL_REGIONS.items():
            metrics = results['clinical_metrics'][region_name]
            dice_mean = np.mean(metrics['dice'])
            print(f"   {region_display_name}: Dice = {dice_mean:.4f}")

def save_ensemble_results(ensemble_metrics, ensemble_clinical_metrics, individual_results, weights, checkpoint_paths, inference_times):
    """Save ensemble results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("ensemble_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed JSON results
    json_path = results_dir / f"ensemble_prediction_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'ensemble_config': {
                'checkpoints': checkpoint_paths,
                'weights': weights
            },
            'ensemble_metrics': ensemble_metrics,
            'ensemble_clinical_metrics': ensemble_clinical_metrics,
            'individual_results': individual_results,
            'performance': {
                'avg_inference_time': float(np.mean(inference_times)) if inference_times else None,
                'throughput': float(batch_size / np.mean(inference_times)) if inference_times else None
            }
        }, f, indent=2)
    
    # Save CSV comparison
    csv_path = results_dir / f"ensemble_prediction_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model_Type', 'Model_Name', 'Weight', 'Class', 'Region', 'Dice_Mean', 'Dice_Std', 'mIoU_Mean', 'mIoU_Std'])
        
        # Ensemble results
        for i in range(num_classes):
            metrics = ensemble_metrics[f"class_{i}"]
            writer.writerow([
                'Ensemble', 'Ensemble', 'N/A', f"Class_{i}", CLASS_NAMES[i],
                np.mean(metrics['dice']), np.std(metrics['dice']),
                np.mean(metrics['miou']), np.std(metrics['miou'])
            ])
        
        for region_name, region_display_name in CLINICAL_REGIONS.items():
            metrics = ensemble_clinical_metrics[region_name]
            writer.writerow([
                'Ensemble', 'Ensemble', 'N/A', 'Clinical', region_display_name,
                np.mean(metrics['dice']), np.std(metrics['dice']),
                np.mean(metrics['miou']), np.std(metrics['miou'])
            ])
        
        # Individual model results
        for model_name, results in individual_results.items():
            checkpoint_name = results['checkpoint']
            weight = results['weight']
            
            for i in range(num_classes):
                metrics = results['all_metrics'][f"class_{i}"]
                writer.writerow([
                    'Individual', checkpoint_name, weight, f"Class_{i}", CLASS_NAMES[i],
                    np.mean(metrics['dice']), np.std(metrics['dice']),
                    np.mean(metrics['miou']), np.std(metrics['miou'])
                ])
            
            for region_name, region_display_name in CLINICAL_REGIONS.items():
                metrics = results['clinical_metrics'][region_name]
                writer.writerow([
                    'Individual', checkpoint_name, weight, 'Clinical', region_display_name,
                    np.mean(metrics['dice']), np.std(metrics['dice']),
                    np.mean(metrics['miou']), np.std(metrics['miou'])
                ])
    
    print(f"\n💾 Ensemble results saved:")
    print(f"  📄 JSON: {json_path}")
    print(f"  📊 CSV:  {csv_path}")

# ---------------------
# Main Execution
# ---------------------
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Ensemble Prediction Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ensemble_prediction_test.py --model spectral_3d_unet_afw_attention --checkpoint-dir /content/drive/MyDrive/BraTS_Project/checkpoints --checkpoints best_model.pth best_model_epoch_47.pth best_model_epoch_43.pth best_model_epoch_54.pth --weights 0.4 0.2 0.3 0.1
  python ensemble_prediction_test.py --checkpoints model1.pth model2.pth model3.pth --weights 0.5 0.3 0.2 --batch-size 1
        """
    )
    
    parser.add_argument("--model", type=str, default="spectral_3d_unet_afw_attention",
                       help="Model name from MODEL_REGISTRY (default: spectral_3d_unet_afw_attention)")
    parser.add_argument("--checkpoint-dir", type=str, default="/content/drive/MyDrive/BraTS_Project/checkpoints",
                       help="Directory containing checkpoint files")
    parser.add_argument("--checkpoints", nargs='+', required=True,
                       help="List of checkpoint filenames (without path)")
    parser.add_argument("--weights", nargs='+', type=float, required=True,
                       help="List of weights for each checkpoint")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--data-dir", type=str,
                       default="/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data",
                       help="Path to dataset directory")
    parser.add_argument("--no-save", action="store_true",
                       help="Skip saving results to files")
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.checkpoints) != len(args.weights):
        print("❌ Error: Number of checkpoints must match number of weights")
        return 1
    
    if abs(sum(args.weights) - 1.0) > 1e-6:
        print("⚠️  Warning: Weights do not sum to 1.0, normalizing...")
        total_weight = sum(args.weights)
        args.weights = [w / total_weight for w in args.weights]
    
    # Construct full checkpoint paths
    checkpoint_paths = []
    for checkpoint in args.checkpoints:
        full_path = os.path.join(args.checkpoint_dir, checkpoint)
        checkpoint_paths.append(full_path)
    
    # Update global variables
    global batch_size, data_dir
    batch_size = args.batch_size
    data_dir = args.data_dir
    
    print("🧠 SPATIAL-SPECTRAL CNN WITH AFW - ENSEMBLE PREDICTION TESTING")
    print("=" * 80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📊 Batch Size: {batch_size}")
    print(f"📁 Data Directory: {data_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Checkpoint Directory: {args.checkpoint_dir}")
    print(f"🎯 Number of Models: {len(args.checkpoints)}")
    print("=" * 80)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    try:
        # Load individual models
        individual_models = load_ensemble_models(checkpoint_paths, args.model)
        
        # Create ensemble model
        ensemble_model = EnsembleModel(individual_models, args.weights)
        ensemble_model.eval()
        
        print(f"\n✅ Ensemble model created with {len(individual_models)} models")
        print(f"🎯 Weights: {args.weights}")
        
        # Get test dataloader
        config = {'compute_stats': False}
        _, _, test_loader = get_dataloaders(
            data_dir,
            batch_size=batch_size,
            train_ratio=0.75,
            val_ratio=0.15,
            config=config
        )
        print(f"📊 Test dataset: {len(test_loader.dataset)} samples")
        
        # Test ensemble
        ensemble_metrics, ensemble_clinical_metrics, individual_results, inference_times = test_ensemble(
            ensemble_model, individual_models, checkpoint_paths, args.weights, test_loader
        )
        
        # Display results
        display_results(ensemble_metrics, ensemble_clinical_metrics, individual_results, args.weights, checkpoint_paths)
        
        # Save results
        if not args.no_save:
            save_ensemble_results(ensemble_metrics, ensemble_clinical_metrics, individual_results, args.weights, checkpoint_paths, inference_times)
        
        print(f"\n🎉 ENSEMBLE PREDICTION TESTING COMPLETED SUCCESSFULLY!")
        print(f"📊 Ensemble model evaluated with {len(individual_models)} individual models")
        
        if not args.no_save:
            print(f"💾 Results saved to ensemble_results/ directory")
        
    except Exception as e:
        print(f"❌ Ensemble testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

# Usage Examples:
# python ensemble_prediction_test.py --checkpoints best_model.pth best_model_epoch_47.pth best_model_epoch_43.pth best_model_epoch_54.pth --weights 0.4 0.2 0.3 0.1
# python ensemble_prediction_test.py --checkpoints model1.pth model2.pth model3.pth --weights 0.5 0.3 0.2 --batch-size 1
