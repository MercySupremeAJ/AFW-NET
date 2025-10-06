#!/usr/bin/env python3
"""
Enhanced Model Summary Script for Spatial-Spectral CNN with AFW
Brain Tumor Segmentation Model Analysis

Features:
- Comprehensive model architecture analysis
- FLOPs and parameter counting
- Memory usage estimation
- Layer-by-layer breakdown
- Export to multiple formats (TXT, CSV, JSON)
- Cross-platform compatibility (CPU/GPU)
"""

import os
import torch
import json
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False
    print("⚠️  torchsummary not available - using basic model info")

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("⚠️  ptflops not available - skipping FLOPs calculation")

from models.spectral_3d_unet_afw_attention import UNet3D_SpectralAFW


def get_basic_model_info(model, input_size, device):
    """Get basic model information without external dependencies"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32 parameter, convert to MB
    
    info = {
        "device": str(device),
        "input_shape": input_size,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "estimated_param_memory_mb": param_memory,
        "model_size_mb": param_memory
    }
    
    return info

def save_model_summary_txt(model, input_size, device, path_txt):
    """Save comprehensive model summary to text file"""
    with open(path_txt, "w", encoding="utf-8") as f:       
        f.write("🧠 SPATIAL-SPECTRAL CNN WITH AFW - MODEL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🖥️  Device: {device}\n")
        f.write(f"📐 Input shape: {input_size}\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic model info
        info = get_basic_model_info(model, input_size, device)
        f.write("📊 BASIC MODEL INFORMATION:\n")
        f.write(f"  Total Parameters: {info['total_parameters']:,}\n")
        f.write(f"  Trainable Parameters: {info['trainable_parameters']:,}\n")
        f.write(f"  Non-trainable Parameters: {info['non_trainable_parameters']:,}\n")
        f.write(f"  Estimated Model Size: {info['model_size_mb']:.2f} MB\n")
        f.write("\n")
        
        # Detailed architecture if torchsummary is available
        if TORCHSUMMARY_AVAILABLE:
            f.write("🏗️  DETAILED ARCHITECTURE:\n")
            f.write("-" * 40 + "\n")
            try:
                with redirect_stdout(f):
                    summary(model, input_size=input_size, device=str(device))
            except Exception as e:
                f.write(f"❌ Error generating detailed summary: {e}\n")
        else:
            f.write("🏗️  LAYER INFORMATION:\n")
            f.write("-" * 40 + "\n")
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules only
                    f.write(f"{name}: {module}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("✅ Model summary completed\n")

def save_model_flops_csv(model, input_size, path_csv):
    """Save FLOPs and parameter information to CSV"""
    input_res = tuple(input_size)
    
    if PTFLOPS_AVAILABLE and torch.cuda.is_available():
        try:
            with torch.cuda.device(0):
                macs, params = get_model_complexity_info(
                    model, input_res, as_strings=False, print_per_layer_stat=True, verbose=True
                )
        except Exception as e:
            print(f"⚠️  FLOPs calculation failed: {e}")
            macs, params = None, None
    else:
        macs, params = None, None
    
    # Get basic info as fallback
    info = get_basic_model_info(model, input_size, torch.device("cpu"))
    
    df = pd.DataFrame([{
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Input Shape": str(input_res),
        "MACs (FLOPs)": f"{macs:,}" if macs else "N/A",
        "Parameters": f"{params:,}" if params else f"{info['total_parameters']:,}",
        "Trainable Parameters": f"{info['trainable_parameters']:,}",
        "Model Size (MB)": f"{info['model_size_mb']:.2f}",
        "Device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    }])
    
    df.to_csv(path_csv, index=False)

def save_model_info_json(model, input_size, device, path_json):
    """Save comprehensive model information to JSON"""
    info = get_basic_model_info(model, input_size, device)
    
    # Add model architecture info
    architecture_info = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            architecture_info[name] = {
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
    
    # Add spectral components info if available
    spectral_info = {}
    try:
        if hasattr(model, 'spectral_bottleneck'):
            spectral_info["spectral_bottleneck"] = {
                "type": type(model.spectral_bottleneck).__name__,
                "parameters": sum(p.numel() for p in model.spectral_bottleneck.parameters())
            }
    except:
        pass
    
    comprehensive_info = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "UNet3D_SpectralAFW",
        "task": "Brain Tumor Segmentation",
        "dataset": "BraTS 2021",
        "basic_info": info,
        "architecture": architecture_info,
        "spectral_components": spectral_info,
        "dependencies": {
            "torchsummary_available": TORCHSUMMARY_AVAILABLE,
            "ptflops_available": PTFLOPS_AVAILABLE,
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    with open(path_json, 'w') as f:
        json.dump(comprehensive_info, f, indent=2)


def main():
    """Main function for comprehensive model analysis"""
    print("🧠 SPATIAL-SPECTRAL CNN WITH AFW - MODEL ANALYSIS")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📊 CUDA Available: {torch.cuda.is_available()}")
    print(f"🔧 TorchSummary Available: {TORCHSUMMARY_AVAILABLE}")
    print(f"🔧 PTFLOPS Available: {PTFLOPS_AVAILABLE}")
    
    # Use reduced shape for safe profiling
    input_size = (4, 64, 64, 64)  # Channels, Depth, Height, Width
    print(f"📐 Input Shape: {input_size}")
    print("=" * 60)
    
    # Initialize model
    print("🔄 Initializing model...")
    model = UNet3D_SpectralAFW(in_channels=4, out_channels=4)
    model.to(device)
    print("✅ Model initialized successfully")
    
    # Create reports directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("./reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive reports
    print("\n📊 Generating comprehensive model analysis...")
    
    # Text summary
    txt_path = reports_dir / f"model_summary_{timestamp}.txt"
    save_model_summary_txt(model, input_size, device, txt_path)
    print(f"✅ Model summary saved to {txt_path}")
    
    # CSV with parameters and FLOPs
    csv_path = reports_dir / f"model_analysis_{timestamp}.csv"
    save_model_flops_csv(model, input_size, csv_path)
    print(f"✅ Parameter analysis saved to {csv_path}")
    
    # JSON with comprehensive info
    json_path = reports_dir / f"model_info_{timestamp}.json"
    save_model_info_json(model, input_size, device, json_path)
    print(f"✅ Comprehensive info saved to {json_path}")
    
    # Display basic info
    info = get_basic_model_info(model, input_size, device)
    print(f"\n📈 MODEL STATISTICS:")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Model Size: {info['model_size_mb']:.2f} MB")
    
    print(f"\n🎉 Model analysis completed successfully!")
    print(f"📁 All reports saved to: {reports_dir.absolute()}")

if __name__ == "__main__":
    main()
