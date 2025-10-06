# train1.py - 3D U-Net with Spectral Convolutions and Adaptive Frequency Weighting (AFW)
# 
# AFW Fixes Applied:
# 1. ✅ Removed conflicting AFW loss computation in DiceCrossEntropyAFWLoss
# 2. ✅ Improved AFW entropy loss to preserve spatial frequency structure
# 3. ✅ Better AFW weight initialization (small random values instead of ones)
# 4. ✅ Added comprehensive AFW weight monitoring during training
# 5. ✅ Fixed AFW regularization to work properly with the spectral model
#
# The AFW system now properly learns frequency-specific attention weights
# that adapt during training to focus on the most important frequency components.

# Set memory optimization BEFORE importing PyTorch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
import csv, json
import pandas as pd
from collections import defaultdict
from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import glob

from dataset import get_dataloaders
from models.spectral_3d_unet_afw_attention import UNet3D_SpectralAFW
from metrics import dice_per_class, sensitivity, specificity, ppv, miou
from log_utils import init_log_files, log_to_csv, log_to_json
from afw_training_utils import (count_trainable_params, afw_entropy_loss, save_afw_weights, monitor_afw_evolution)
from preprocessing import set_data_paths

# -----------------------------
# Configuration
# -----------------------------
data_dir = "/content/drive/MyDrive/BraTS_Project/BraTS2021_Training_Data"

# Initialize preprocessing data paths
set_data_paths(data_dir, data_dir)

num_epochs = 60  # Moderate increase from 50 to allow for recovery and convergence
batch_size = 5   # Conservative batch size to avoid OOM
lr = 1.25e-4    # Scaled learning rate: 1e-4 * (5/4) = 1.25e-4
seed = 42
CLASS_IDS = (0, 1, 2, 3)
NUM_CLASSES = len(CLASS_IDS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory optimization settings
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"🔧 GPU Memory Optimization Enabled")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
use_amp = True  # Mixed precision training

#---------------------------
# Logging paths
#---------------------------
log_dir = "/content/drive/MyDrive/BraTS_Project/logs"
csv_log_path = os.path.join(log_dir, "metrics_log.csv")
json_log_path = os.path.join(log_dir, "metrics_log.json")
AFW_log_path = os.path.join(log_dir, "afw_loss_log.csv")
loss_log_path = os.path.join(log_dir, "full_loss_log.csv")

# ✅ Make sure the full log directory exists
os.makedirs(log_dir, exist_ok=True)

init_log_files(csv_log_path, json_log_path, num_classes=NUM_CLASSES)

# Create file and write header if it doesn't exist
if not os.path.exists(AFW_log_path):
    with open(AFW_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "AFW_Loss"])

#-----------------------------
# Checkpoints
#-----------------------------
checkpoint_dir = "/content/drive/MyDrive/BraTS_Project/checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
# ✅ Make sure the full log directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# 📂 Path to voxel distribution CSV (after mapping with BraTSBraTSDataset)
reports_dir = "/content/drive/MyDrive/BraTS_Project/reports"
os.makedirs(reports_dir, exist_ok=True)  # ✅ Ensure reports directory exists

csv_path = os.path.join(reports_dir, "brats_mapped_voxel_distribution.csv")
weights_json_path = os.path.join(reports_dir, "class_weights.json")
weights_pt_path = os.path.join(reports_dir, "class_weights.pt")
report_dir = os.path.join(reports_dir, "figures")
training_viz_dir = os.path.join(reports_dir, "training_monitoring")  # New directory for training visualizations
os.makedirs(report_dir, exist_ok=True)  # ✅ Ensure figures directory exists
os.makedirs(training_viz_dir, exist_ok=True)  # ✅ Ensure training monitoring directory exists


# Early stopping
early_stopping_patience = 10
best_val_loss = float("inf")
epochs_no_improve = 0


# --------------------------------------------------
# 🔢 Class weight estimation
# --------------------------------------------------

def compute_and_save_class_weights_from_csv(csv_path, weights_json_path, visualize=False, report_dir="/content/drive/MyDrive/BraTS_Project/reports/figures/"):
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV with voxel counts per class
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["class_0", "class_1", "class_2", "class_3"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV missing required columns: {required_columns}")
        voxel_sums = df[required_columns].sum()
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

    # Compute inverse frequency
    total_voxels = voxel_sums.sum()
    raw_weights = total_voxels / (4 * voxel_sums)  # Inverse frequency

    # Normalize
    normalized_weights = raw_weights / raw_weights.sum()  # Normalize

    # 📊 Enhanced Visualization for Imbalanced Classes
    if visualize:  
        os.makedirs(report_dir, exist_ok=True)
        save_path = os.path.join(report_dir, "class_weights_bar_chart.png")
        
        # Create figure with better dimensions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Raw voxel counts (log scale for imbalanced classes)
        class_names = ['Background', 'Necrotic', 'Edema', 'Enhancing']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        ax1.bar(class_names, voxel_sums.values, color=colors, alpha=0.8)
        ax1.set_yscale('log')  # Log scale for better visualization of imbalanced classes
        ax1.set_title("Raw Voxel Counts per Class (Log Scale)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Voxel Count (log scale)")
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(voxel_sums.values):
            ax1.text(i, v * 1.1, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Normalized class weights
        bars = ax2.bar(class_names, normalized_weights.values, color=colors, alpha=0.8)
        ax2.set_title("Normalized Class Weights for Training", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Weight Value")
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(normalized_weights.values):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add statistics text
        stats_text = f"Total Voxels: {total_voxels:,}\nClass Imbalance Ratio: {voxel_sums.max() / voxel_sums.min():.1f}:1"
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved enhanced class weights chart to: {save_path}")
        plt.show()
        
        # Print detailed statistics
        print(f"\n📊 Class Distribution Analysis:")
        print(f"{'Class':<12} {'Voxels':<15} {'Percentage':<12} {'Weight':<10}")
        print("-" * 50)
        for i, (name, count, weight) in enumerate(zip(class_names, voxel_sums.values, normalized_weights.values)):
            percentage = (count / total_voxels) * 100
            print(f"{name:<12} {count:<15,} {percentage:<12.2f}% {weight:<10.4f}")

    # Save to JSON
    with open(weights_json_path, "w") as f:
        json.dump({f"class_{i}": float(w) for i, w in enumerate(normalized_weights)}, f, indent=4)

    # Save as PyTorch tensor (for training use)
    weights_tensor = torch.tensor(normalized_weights.values, dtype=torch.float32)
    torch.save(weights_tensor, weights_pt_path)

    print(f"📁 Saved class weights to: {weights_json_path} and {weights_pt_path}")
    return weights_tensor


# --------------------------------------------------
# New Training Utility Classes
# --------------------------------------------------

class TrainingProgressTracker:
    """Track training progress and estimate completion time"""
    
    def __init__(self, total_epochs):
        self.start_time = datetime.now()
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.current_epoch = 0
        
    def start_epoch(self, epoch):
        self.current_epoch = epoch
        self.epoch_start = datetime.now()
        
    def end_epoch(self):
        epoch_duration = datetime.now() - self.epoch_start
        self.epoch_times.append(epoch_duration)
        
    def estimate_completion(self):
        if len(self.epoch_times) > 0:
            avg_time_per_epoch = sum(self.epoch_times, timedelta()) / len(self.epoch_times)
            remaining_epochs = self.total_epochs - self.current_epoch
            estimated_completion = datetime.now() + (avg_time_per_epoch * remaining_epochs)
            return estimated_completion.strftime("%Y-%m-%d %H:%M:%S")
        return "Calculating..."
    
    def get_progress_stats(self):
        elapsed = datetime.now() - self.start_time
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times, timedelta()) / len(self.epoch_times)
            return {
                'elapsed': str(elapsed).split('.')[0],
                'avg_epoch_time': str(avg_epoch_time).split('.')[0],
                'completion_estimate': self.estimate_completion()
            }
        return {'elapsed': str(elapsed).split('.')[0]}


class CheckpointManager:
    """Manage checkpoints for large models with size limits"""
    
    def __init__(self, checkpoint_dir, max_checkpoints=5, size_limit_gb=2.0, retain_best=3, retain_latest=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.size_limit_gb = size_limit_gb * (1024**3)  # Convert to bytes
        # Retention policies
        self.retain_best = retain_best
        self.retain_latest = retain_latest
        
    def _print_size_notice(self, file_path, label):
        size_mb = self.get_checkpoint_size_mb(file_path)
        print(f"💾 Saved checkpoint: {label} ({size_mb:.1f} MB)")
        if size_mb > (self.size_limit_gb / (1024**2)):
            print(f"⚠️  Checkpoint size ({size_mb:.1f} MB) exceeds limit")
        
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space.
        Policy:
          - Keep at least the last N best full checkpoints (best_model_epoch_*.pth)
          - Keep only a few latest full checkpoints (latest_checkpoint_epoch_*.pth)
          - Do not delete stable files like best_model.pth or latest_checkpoint.pth
        """
        # Group files
        best_files = glob.glob(os.path.join(self.checkpoint_dir, "best_model_epoch_*.pth"))
        latest_files = glob.glob(os.path.join(self.checkpoint_dir, "latest_checkpoint_epoch_*.pth"))

        # Sort by mtime (newest first)
        best_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Prune best files beyond retain_best
        if len(best_files) > self.retain_best:
            for file_path in best_files[self.retain_best:]:
                try:
                    os.remove(file_path)
                    print(f"🗑️  Removed old best checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️  Failed to remove {file_path}: {e}")

        # Prune latest files beyond retain_latest
        if self.retain_latest is not None and len(latest_files) > self.retain_latest:
            for file_path in latest_files[self.retain_latest:]:
                try:
                    os.remove(file_path)
                    print(f"🗑️  Removed old latest checkpoint: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️  Failed to remove {file_path}: {e}")
    
    def get_checkpoint_size_mb(self, file_path):
        """Get checkpoint file size in MB"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024**2)
        except:
            return 0
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename, additional_data=None):
        """Save checkpoint with size validation and optional additional data"""
        file_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add additional data if provided (e.g., scheduler state)
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save checkpoint
        torch.save(checkpoint_data, file_path)
        
        # Check size and cleanup if needed
        self._print_size_notice(file_path, filename)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()

        # (weights-only save removed per user request; always save full FP32 checkpoints)


class LiveTrainingVisualizer:
    """Real-time training visualization"""
    
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Live Training Progress', fontsize=16)
        
        # Initialize plots
        self.loss_ax = self.axes[0, 0]
        self.metrics_ax = self.axes[0, 1]
        self.afw_ax = self.axes[1, 0]
        self.lr_ax = self.axes[1, 1]
        
        self.loss_data = {'train': [], 'val': [], 'epochs': []}
        self.metrics_data = {'dice': [], 'miou': [], 'epochs': []}
        self.afw_data = {'min': [], 'max': [], 'mean': [], 'epochs': []}
        self.lr_data = {'lr': [], 'epochs': []}
        
        plt.tight_layout()
        
    def update_plots(self, epoch, train_loss, val_loss, metrics, afw_stats, lr):
        """Update all plots with new data"""
        if epoch % self.update_interval != 0:
            return
            
        # Update data
        self.loss_data['epochs'].append(epoch)
        self.loss_data['train'].append(train_loss)
        self.loss_data['val'].append(val_loss)
        
        self.metrics_data['epochs'].append(epoch)
        self.metrics_data['dice'].append(metrics.get('dice', 0))
        self.metrics_data['miou'].append(metrics.get('miou', 0))
        
        self.afw_data['epochs'].append(epoch)
        self.afw_data['min'].append(afw_stats.get('min', 0))
        self.afw_data['max'].append(afw_stats.get('max', 0))
        self.afw_data['mean'].append(afw_stats.get('mean', 0))
        
        self.lr_data['epochs'].append(epoch)
        self.lr_data['lr'].append(lr)
        
        # Clear and redraw plots
        self._redraw_plots()
        plt.pause(0.01)  # Small pause to update display
        
    def _redraw_plots(self):
        """Redraw all plots with current data"""
        # Loss plot
        self.loss_ax.clear()
        if self.loss_data['epochs']:
            self.loss_ax.plot(self.loss_data['epochs'], self.loss_data['train'], 'b-', label='Train Loss')
            self.loss_ax.plot(self.loss_data['epochs'], self.loss_data['val'], 'r-', label='Val Loss')
            self.loss_ax.set_title('Training & Validation Loss')
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.legend()
            self.loss_ax.grid(True)
        
        # Metrics plot
        self.metrics_ax.clear()
        if self.metrics_data['epochs']:
            self.metrics_ax.plot(self.metrics_data['epochs'], self.metrics_data['dice'], 'g-', label='Dice Score')
            self.metrics_ax.plot(self.metrics_data['epochs'], self.metrics_data['miou'], 'm-', label='mIoU')
            self.metrics_ax.set_title('Segmentation Metrics')
            self.metrics_ax.set_xlabel('Epoch')
            self.metrics_ax.set_ylabel('Score')
            self.metrics_ax.legend()
            self.metrics_ax.grid(True)
        
        # AFW weights plot
        self.afw_ax.clear()
        if self.afw_data['epochs']:
            self.afw_ax.plot(self.afw_data['epochs'], self.afw_data['min'], 'b-', label='Min')
            self.afw_ax.plot(self.afw_data['epochs'], self.afw_data['max'], 'r-', label='Max')
            self.afw_ax.plot(self.afw_data['epochs'], self.afw_data['mean'], 'g-', label='Mean')
            self.afw_ax.set_title('AFW Weights Evolution')
            self.afw_ax.set_xlabel('Epoch')
            self.afw_ax.set_ylabel('Weight Value')
            self.afw_ax.legend()
            self.afw_ax.grid(True)
        
        # Learning rate plot
        self.lr_ax.clear()
        if self.lr_data['epochs']:
            self.lr_ax.semilogy(self.lr_data['epochs'], self.lr_data['lr'], 'b-')
            self.lr_ax.set_title('Learning Rate')
            self.lr_ax.set_xlabel('Epoch')
            self.lr_ax.set_ylabel('Learning Rate')
            self.lr_ax.grid(True)
        
        plt.tight_layout()
    
    def save_final_plot(self, save_path):
        """Save the final training visualization"""
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training visualization saved to: {save_path}")
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)




# -----------------------------
# Dice Loss: Loss function with class weights
# -----------------------------
class DiceCrossEntropyAFWLoss(nn.Module):
    def __init__(self, class_weights=None, afw_reg=False, lambda_afw=0.01, alpha=0.5):
        """
        Combines CrossEntropy + Dice loss with optional AFW regularization.

        Args:
            class_weights (torch.Tensor or None): Class weights for CrossEntropyLoss.
            afw_reg (bool): Whether to apply AFW regularization.
            lambda_afw (float): Regularization weight for AFW.
            alpha (float): Weight between CE and Dice (e.g., 0.5 = equal).
        """
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        # Set to_onehot_y=False since we handle one-hot manually
        self.dice = DiceCELoss(
            to_onehot_y=False,
            softmax=True,               # Dice loss works on softmax probabilities
            include_background=False    # Optional: set to True if you want to include background class
        )

        self.afw_reg = afw_reg
        self.lambda_afw = lambda_afw
        self.alpha = alpha

    def forward(self, preds, target):
        """
        Args:
            preds: logits, shape [B, C, D, H, W]
            target: ground truth class indices, shape [B, D, H, W] or [B, 1, D, H, W]
        Returns:
            total_loss: Weighted sum of CE, Dice, and optional AFW losses
            ce_loss, dice_loss, afw_loss: Individual loss components (detached)
        """

        # Step 1: Squeeze target to shape [B, D, H, W] if needed
        if target.ndim == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        # Step 2: Cross Entropy Loss (expects class indices, no one-hot)
        ce = self.ce(preds, target)

        # Step 3: One-hot encode target manually → shape: [B, C, D, H, W]
        target_onehot = F.one_hot(target.long(), num_classes=preds.shape[1])  # [B, D, H, W, C]
        target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()          # → [B, C, D, H, W]

        # Step 4: Dice Loss (manual one-hot passed, to_onehot_y=False)
        dice = self.dice(preds, target_onehot)

        # Step 5: AFW regularization is handled separately in training loop
        # This prevents double-counting and conflicting loss computations
        # Note: AFW loss is computed and added in the training loop, not here
        afw_loss = torch.tensor(0.0, device=preds.device, requires_grad=False)
        total_loss = self.alpha * ce + (1 - self.alpha) * dice

        return total_loss, ce.detach(), dice.detach(), afw_loss.detach()

# -----------------------------
# Train Function
# -----------------------------
def train_model(train_loader, val_loader):
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize training utilities
    progress_tracker = TrainingProgressTracker(num_epochs)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    visualizer = LiveTrainingVisualizer(update_interval=2)  # Update every 2 epochs
    
    # ✅ Step 1: Sanity check goes right here
    images, masks = next(iter(train_loader))
    print(f"[2nd Sanity Check] Image shape: {images.shape}")  # Expect: [B, 4, 240, 240, 152] where B is the batch_size.
    print(f"[3rd Sanity Check] Mask shape:  {masks.shape}")   # Expect: [B, 240, 240, 152]
    print(f"[4th Sanity Check] Unique labels in batch: {torch.unique(masks)}")

    # Model
    model = UNet3D_SpectralAFW(in_channels=4, out_channels=NUM_CLASSES).to(device)
    # Print model parameter count
    count_trainable_params(model)

    # --------------------------------------
    # Load pre-computed class weights
    # --------------------------------------
    # Losses and optimizer
    
    
    # ✅ Load pre-computed weights (already computed in main block)
    if os.path.exists(weights_pt_path):
        weights_tensor = torch.load(weights_pt_path, map_location=device, weights_only=False)
        print("✅ Loaded pre-computed class weights:", weights_tensor)
    else:
        # Fallback: compute weights if PT file doesn't exist
        weights_tensor = compute_and_save_class_weights_from_csv(
            csv_path,
            weights_json_path,
            visualize=False  # Already visualized in main block
        ).to(device)
        print("⚠️  Fallback: computed class weights:", weights_tensor)
    
    print("\n✅ Class weights details:")
    for idx, w in enumerate(weights_tensor.cpu().tolist()):
        print(f"  Class {idx}: {w:.6f}")

    criterion = DiceCrossEntropyAFWLoss(
        class_weights=weights_tensor, #my precomputed tensors
        afw_reg=True,
        lambda_afw=0.01, 
        alpha=0.5
        )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6
    )  # Improved scheduler (verbose not supported in this PyTorch version)

    # Mixed precision support
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume training from best checkpoint (improved logic)
    start_epoch = 0
    resume_loaded = False
    best_val_loss = float('inf')
    
    # Try to load best checkpoint first
    if os.path.exists(best_checkpoint_path):
        try:
            print(f"🏆 Loading best checkpoint: {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if checkpoint.get('optimizer_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"✅ Loaded best model from epoch {start_epoch} with val_loss: {best_val_loss:.4f}")
                resume_loaded = True
        except Exception as e:
            print(f"⚠️  Failed to load best checkpoint: {e}")
    
    # Fallback to latest checkpoint if best checkpoint failed
    if not resume_loaded and os.path.exists(checkpoint_path):
        try:
            print(f"⏪ Fallback: Loading latest checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if checkpoint.get('optimizer_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if checkpoint.get('scheduler_state_dict'):
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"✅ Resumed from epoch {start_epoch}")
                resume_loaded = True
        except Exception as e:
            print(f"⚠️  Failed to load latest checkpoint: {e}")

    if not resume_loaded:
        # Prefer most recent best
        best_ckpts = sorted(
            glob.glob(os.path.join(checkpoint_dir, "best_model_epoch_*.pth")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for ckpt_file in best_ckpts:
            try:
                print(f"⏪ Attempting resume from best: {ckpt_file}")
                checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if checkpoint.get('optimizer_state_dict'):
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', -1) + 1
                    print(f"✅ Resumed from epoch {start_epoch}")
                    resume_loaded = True
                    break
            except Exception as e:
                print(f"⚠️  Failed to load {ckpt_file}: {e}")

    if not resume_loaded:
        epoch_ckpts = sorted(
            glob.glob(os.path.join(checkpoint_dir, "latest_checkpoint_epoch_*.pth")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        for ckpt_file in epoch_ckpts:
            try:
                print(f"⏪ Attempting resume from: {ckpt_file}")
                checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if checkpoint['optimizer_state_dict'] is not None:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', -1) + 1
                    print(f"✅ Resumed from epoch {start_epoch}")
                    resume_loaded = True
                    break
            except Exception as e:
                print(f"⚠️  Failed to load {ckpt_file}: {e}")


    # Training loop with improved early stopping
    epochs_no_improve = 0
    early_stop_patience = 8  # Increased patience for better convergence
    min_delta = 0.001  # Minimum change to qualify as improvement
    
    print(f"🚀 Starting training from epoch {start_epoch + 1} with early stopping patience: {early_stop_patience}")
    print(f"📊 Current best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"🔧 Gradient clipping: max_norm=1.0")
    print(f"⚖️  AFW regularization: lambda=0.005")
    print(f"📈 Learning rate scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    print(f"💾 Weight decay: 1e-5")
    print(f"🎯 Batch size: {batch_size} (max available)")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        # Start epoch tracking
        progress_tracker.start_epoch(epoch)
        model.train()
        train_loss = 0.0        
        ce_losses = []
        dice_losses = []
        afw_losses = []
        epoch_afw_losses = []  # Store per-epoch average AFW losses. This reset here at the start of each epoch.

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training"):
            images = images.to(device)   # [B, 4, H, W, D]
            masks = masks.long().to(device)  # [B, H, W, D]

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)     # [B, C, H, W, D]

                loss, ce_loss, dice_loss, afw_loss = criterion(outputs, masks.unsqueeze(1))
                ce_losses.append(ce_loss.item())
                dice_losses.append(dice_loss.item())
                afw_losses.append(afw_loss.item())

                # -------------------------------
                # Add AFW entropy regularization (reduced strength for stability)
                # -------------------------------
                lambda_afw = 0.005  # Reduced from 0.01 for better training stability
                if hasattr(model, 'spectral_bottleneck') and hasattr(model.spectral_bottleneck, 'spectral_convs'):
                    afw_weights = model.spectral_bottleneck.spectral_convs[0].afw
                    afw_loss = afw_entropy_loss(afw_weights)
                    loss += lambda_afw * afw_loss
                    epoch_afw_losses.append(afw_loss.item())  # Collect for averaging later
                    
                    # Monitor AFW evolution
                    if (epoch + 1) % 5 == 0:  # Every 5 epochs
                        afw_logs_dir = os.path.join(training_viz_dir, "afw_evolution")
                        os.makedirs(afw_logs_dir, exist_ok=True)
                        monitor_afw_evolution(afw_weights, epoch + 1, save_dir=afw_logs_dir)
                else:
                    print(f"[AFW] Epoch {epoch+1}: No spectral_bottleneck found — skipping AFW loss.")
                    epoch_afw_losses.append(0.0)  # Add 0 for consistency

            # Backpropagation with mixed precision and gradient clipping
            scaler.scale(loss).backward()
            
            # Gradient clipping for training stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        
        # ---------- After All Batches ----------
        #----------------------------------------
        # ✅ Compute averages for each loss component
        #----------------------------------------
        avg_train_loss = train_loss / len(train_loader)
        avg_ce_loss = sum(ce_losses) / len(ce_losses)
        avg_dice_loss = sum(dice_losses) / len(dice_losses)
        avg_afw_loss = sum(afw_losses) / len(afw_losses)

        # End epoch tracking
        progress_tracker.end_epoch()
        
        # Get progress stats
        progress_stats = progress_tracker.get_progress_stats()
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        print(f"[Epoch {epoch+1}]  → CE: {avg_ce_loss:.4f} | Dice: {avg_dice_loss:.4f} | AFW: {avg_afw_loss:.4f}")
        print(f"[Progress] Elapsed: {progress_stats['elapsed']}, ETA: {progress_stats.get('completion_estimate', 'Calculating...')}")
        print(f"[LR] Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Log per-epoch losses
        write_header = not os.path.exists(loss_log_path) or os.stat(loss_log_path).st_size == 0

        with open(loss_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Epoch", "Total_Loss", "CE_Loss", "Dice_Loss", "AFW_Loss"])
            writer.writerow([epoch + 1, avg_train_loss, avg_ce_loss, avg_dice_loss, avg_afw_loss])
        

        if epoch_afw_losses:
            avg_afw_loss = sum(epoch_afw_losses) / len(epoch_afw_losses)
            print(f"[AFW] Epoch {epoch+1}: Avg AFW loss = {avg_afw_loss:.4f}")

            # Append to log file
            with open(AFW_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, avg_afw_loss])



        # -------------------------------
        # Validation phase
        # -------------------------------
        model.eval()
        val_loss = 0.0
        all_metrics = defaultdict(list)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss, ce, dice, afw = criterion(outputs, masks)
                val_loss += loss.item()

                for cls in range(NUM_CLASSES):
                    dice_val = dice_per_class(outputs, masks, NUM_CLASSES)[cls]
                    miou_val = miou(outputs, masks, NUM_CLASSES)[cls]
                    sen = sensitivity(outputs, masks, cls).item()
                    spc = specificity(outputs, masks, cls).item()
                    ppv_val = ppv(outputs, masks, cls).item()

                    all_metrics[f'dice_{cls}'].append(dice_val)
                    all_metrics[f'miou_{cls}'].append(miou_val)
                    all_metrics[f'sens_{cls}'].append(sen)
                    all_metrics[f'spec_{cls}'].append(spc)
                    all_metrics[f'ppv_{cls}'].append(ppv_val)

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # Print & log metrics
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}")

        # -----------------------------------------------
        # Save AFW weights every 5 epochs for visualization
        # -----------------------------------------------
        afw_stats = {'min': 0, 'max': 0, 'mean': 0}  # Default values
        if (epoch + 1) % 5 == 0:
            if hasattr(model, 'spectral') and hasattr(model.spectral, 'afw'):
                afw_weights_dir = os.path.join(training_viz_dir, "afw_weights")
                os.makedirs(afw_weights_dir, exist_ok=True)
                save_afw_weights(epoch + 1, model.spectral.afw, save_dir=afw_weights_dir)
                print(f"[AFW] Saved AFW weights at epoch {epoch + 1}")
                
                # Monitor AFW weight evolution using the dedicated function
                afw_stats = monitor_afw_evolution(model.spectral.afw, epoch + 1, save_dir=afw_logs_dir)
            else:
                print(f"[AFW] Epoch {epoch + 1}: AFW weights not found — nothing saved.")
        
        epoch_metrics = []
        for cls in CLASS_IDS:
            dice_avg = np.mean(all_metrics[f'dice_{cls}'])
            miou_avg = np.mean(all_metrics[f'miou_{cls}'])
            sens_avg = np.mean(all_metrics[f'sens_{cls}'])
            spec_avg = np.mean(all_metrics[f'spec_{cls}'])
            ppv_avg = np.mean(all_metrics[f'ppv_{cls}'])

            print(f"  Class {cls}: Dice={dice_avg:.4f}, mIoU={miou_avg:.4f}, Sensitivity={sens_avg:.4f}, Specificity={spec_avg:.4f}, PPV={ppv_avg:.4f}")
            epoch_metrics.append({
                'dice': dice_avg,
                'miou': miou_avg,
                'sensitivity': sens_avg,
                'specificity': spec_avg,
                'ppv': ppv_avg
            })
        
        # Update live visualization
        current_lr = optimizer.param_groups[0]['lr']
        metrics_dict = {'dice': np.mean([m['dice'] for m in epoch_metrics]) if epoch_metrics else 0,
                       'miou': np.mean([m['miou'] for m in epoch_metrics]) if epoch_metrics else 0}
        visualizer.update_plots(epoch + 1, avg_train_loss, avg_val_loss, metrics_dict, afw_stats, current_lr)

        # Logging train & validation loss to file
        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': epoch_metrics
        }

        log_to_csv(csv_log_path, epoch_record)
        log_to_json(json_log_path, epoch_record)

        # Save checkpoints using CheckpointManager
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, epoch_record, 
            f"latest_checkpoint_epoch_{epoch+1}.pth"
        )

        # Save best checkpoint (also used for resume preference)
        improvement = best_val_loss - avg_val_loss
        if improvement > min_delta:
            best_val_loss = avg_val_loss
            
            # Save best checkpoint with scheduler state
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'metrics': epoch_metrics,
                'best_val_loss': best_val_loss
            }
            
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, epoch_record,
                f"best_model_epoch_{epoch+1}.pth",
                additional_data={'scheduler_state_dict': scheduler.state_dict()}
            )
            # Also update a stable best checkpoint path
            torch.save(checkpoint_data, best_checkpoint_path)
            print(f"📌 Best model saved! (Improvement: {improvement:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epochs. (Change: {improvement:.4f})")

        # Early stopping with improved logic
        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            print(f"📊 Best validation loss: {best_val_loss:.4f}")
            print(f"📈 Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            break
    
    # Save final training visualization
    final_plot_path = os.path.join(training_viz_dir, "final_training_visualization.png")
    visualizer.save_final_plot(final_plot_path)
    visualizer.close()
    
    # Print final training summary
    final_stats = progress_tracker.get_progress_stats()
    print(f"\n🎉 Training completed!")
    print(f"📊 Total training time: {final_stats['elapsed']}")
    print(f"📈 Final visualization saved to: {final_plot_path}")
    print(f"💾 Checkpoints saved in: {checkpoint_dir}")
    print(f"📝 Logs saved in: {log_dir}")
    print(f"📊 Reports saved in: {report_dir}")
    print(f"📈 Training monitoring visualizations saved in: {training_viz_dir}")


def print_training_summary():
    """
    Print a comprehensive summary of training configuration.
    """
    print("🚀 Training Configuration Summary")
    print("=" * 50)
    print(f"📊 Model: UNet3D_SpectralAFW with AFW")
    print(f"📁 Data Directory: {data_dir}")
    print(f"🔢 Batch Size: {batch_size}")
    print(f"📈 Learning Rate: {lr}")
    print(f"⏱️  Epochs: {num_epochs}")
    print(f"🎯 Classes: {NUM_CLASSES}")
    print(f"💾 Device: {device}")
    print(f"🔧 Mixed Precision: {use_amp}")
    print(f"📝 Log Directory: {log_dir}")
    print(f"💾 Checkpoint Directory: {checkpoint_dir}")
    print(f"📊 Report Directory: {report_dir}")
    print(f"📈 Training Monitoring Directory: {training_viz_dir}")
    print(f"🛑 Early Stopping Patience: 8")  # Will be set in training function
    print("=" * 50)
    print()


if __name__ == "__main__":
    # 🔹 Prepare data loaders first
    print("🔄 Initializing dataset and creating data loaders...")
    print("⏳ This may take a few minutes for large datasets...")
    print("💡 Statistics will be cached for future runs to speed up startup")
    
    # Check if statistical files exist and are valid
    csv_voxel_path = os.path.join(reports_dir, "brats_mapped_voxel_distribution.csv")
    weights_json_path = os.path.join(reports_dir, "class_weights.json")
    
    # Check for existing statistical files
    stats_files_exist = (
        os.path.exists(csv_voxel_path) and 
        os.path.exists(weights_json_path) and
        os.path.exists(weights_pt_path)
    )
    
    # Also check for dataset statistics cache files
    cache_files = glob.glob(os.path.join(reports_dir, "dataset_statistics_cache_*.json"))
    cache_exists = len(cache_files) > 0
    
    # Validate CSV file if it exists
    csv_valid = False
    if os.path.exists(csv_voxel_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_voxel_path)
            required_columns = ["class_0", "class_1", "class_2", "class_3"]
            csv_valid = all(col in df.columns for col in required_columns) and len(df) > 0
            if csv_valid:
                print(f"✅ Found valid voxel distribution CSV: {csv_voxel_path}")
            else:
                print(f"⚠️  Invalid CSV format, will recompute statistics")
        except Exception as e:
            print(f"⚠️  Error reading CSV: {e}, will recompute statistics")
    
    # Skip statistics computation if all required files exist and are valid
    if stats_files_exist and csv_valid and cache_exists:
        print("🔄 Found existing statistical files - skipping dataset statistics computation")
        print(f"   📊 CSV: {csv_voxel_path}")
        print(f"   📊 Weights: {weights_json_path}")
        print(f"   📊 Cache: {len(cache_files)} file(s)")
        dataset_config = {'compute_stats': False}
    else:
        print("🆕 Missing or invalid statistical files - computing dataset statistics")
        if not stats_files_exist:
            print("   ⚠️  Missing weight files")
        if not csv_valid:
            print("   ⚠️  Invalid CSV file")
        if not cache_exists:
            print("   ⚠️  Missing cache files")
        dataset_config = {'compute_stats': True}
    
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        train_ratio=0.75,
        val_ratio=0.15,
        augment=True,
        num_workers=4, # Enable parallel data loading for faster iterations
        pin_memory=True, # Enable memory pinning for faster GPU transfer
        config=dataset_config  # Pass config to control statistics computation
    )
    
    print("✅ Data loaders created successfully!")
    print(f"📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")

    # ✅ Class weight logic (files already checked above)

    # Check if weights already exist, compute only if needed
    if os.path.exists(weights_pt_path) and os.path.exists(weights_json_path):
        print("📁 Loading existing class weights...")
        weights_tensor = torch.load(weights_pt_path, map_location=device, weights_only=False)
        print(f"✅ Loaded existing weights: {weights_tensor}")
    else:
        print("🔄 Computing class weights from scratch...")
        weights_tensor = compute_and_save_class_weights_from_csv(
            csv_path=csv_voxel_path,
            weights_json_path=weights_json_path,
            visualize=True,  # Save and show bar chart
            report_dir=report_dir
        ).to(device)
        print(f"✅ Computed and saved new weights: {weights_tensor}")
        print(f"📁 Weights location: {weights_json_path}")
    
    print("🚀 All preprocessing complete - starting training!")

    # 🚀 Train model - moved outside the conditional block
    # 🔹 Start training
    print_training_summary()
    train_model(train_loader, val_loader)

