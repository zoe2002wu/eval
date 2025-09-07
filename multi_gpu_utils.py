"""
Multi-GPU utilities for parallelizing COCO evaluation across 4 GPUs.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import multiprocessing as mp
from functools import partial


def split_data_for_gpus(csv_path: str, num_gpus: int = 4, output_dir: str = "data/coco/gpu_splits") -> List[str]:
    """
    Split the COCO dataset into chunks for each GPU.
    
    Args:
        csv_path: Path to the COCO CSV file
        num_gpus: Number of GPUs to split data across
        output_dir: Directory to save the split CSV files
        
    Returns:
        List of paths to the split CSV files
    """
    print(f"Splitting data for {num_gpus} GPUs...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the full dataset
    data = pd.read_csv(csv_path)
    total_samples = len(data)
    samples_per_gpu = total_samples // num_gpus
    
    split_files = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * samples_per_gpu
        if gpu_id == num_gpus - 1:
            # Last GPU gets any remaining samples
            end_idx = total_samples
        else:
            end_idx = (gpu_id + 1) * samples_per_gpu
            
        gpu_data = data.iloc[start_idx:end_idx].copy()
        
        # Add GPU ID column for tracking
        gpu_data['gpu_id'] = gpu_id
        
        # Save split
        split_file = os.path.join(output_dir, f"coco_30k_gpu_{gpu_id}.csv")
        gpu_data.to_csv(split_file, index=False)
        split_files.append(split_file)
        
        print(f"GPU {gpu_id}: {len(gpu_data)} samples ({start_idx}-{end_idx-1})")
    
    return split_files


def create_gpu_output_dirs(base_output_dir: str, num_gpus: int = 4) -> List[str]:
    """
    Create separate output directories for each GPU.
    
    Args:
        base_output_dir: Base output directory
        num_gpus: Number of GPUs
        
    Returns:
        List of GPU-specific output directories
    """
    gpu_dirs = []
    for gpu_id in range(num_gpus):
        gpu_dir = os.path.join(base_output_dir, f"gpu_{gpu_id}")
        os.makedirs(gpu_dir, exist_ok=True)
        gpu_dirs.append(gpu_dir)
    return gpu_dirs


def aggregate_results(result_files: List[str], output_file: str) -> Dict[str, Any]:
    """
    Aggregate results from multiple GPUs into a single result file.
    
    Args:
        result_files: List of paths to result JSON files from each GPU
        output_file: Path to save the aggregated results
        
    Returns:
        Aggregated results dictionary
    """
    print("Aggregating results from all GPUs...")
    
    aggregated = {}
    
    for result_file in result_files:
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                gpu_results = json.load(f)
                
            # Merge results (assuming each GPU has unique keys or we want to average)
            for key, value in gpu_results.items():
                if key not in aggregated:
                    aggregated[key] = {}
                
                # For metrics like FID, LPIPS, CLIP - we might want to average or keep separate
                if isinstance(value, dict):
                    for metric, metric_value in value.items():
                        if metric not in aggregated[key]:
                            aggregated[key][metric] = []
                        aggregated[key][metric].append(metric_value)
                else:
                    if key not in aggregated:
                        aggregated[key] = []
                    aggregated[key].append(value)
    
    # Average the metrics across GPUs
    for key, value in aggregated.items():
        if isinstance(value, dict):
            for metric, metric_values in value.items():
                if isinstance(metric_values, list) and len(metric_values) > 0:
                    if isinstance(metric_values[0], dict):
                        # Handle nested dictionaries (e.g., {'mean': 0.5, 'std': 0.1})
                        if 'mean' in metric_values[0]:
                            means = [v['mean'] for v in metric_values if 'mean' in v]
                            stds = [v['std'] for v in metric_values if 'std' in v]
                            aggregated[key][metric] = {
                                'mean': np.mean(means),
                                'std': np.sqrt(np.mean([s**2 for s in stds]))  # RMS of stds
                            }
                    else:
                        # Simple numeric values
                        aggregated[key][metric] = np.mean(metric_values)
    
    # Save aggregated results
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=4)
    
    print(f"Aggregated results saved to {output_file}")
    return aggregated


def check_gpu_availability(num_gpus: int = 4) -> List[int]:
    """
    Check which GPUs are available and return a list of available GPU IDs.
    
    Args:
        num_gpus: Number of GPUs requested
        
    Returns:
        List of available GPU IDs
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        if i < num_gpus:
            available_gpus.append(i)
    
    if len(available_gpus) < num_gpus:
        print(f"Warning: Only {len(available_gpus)} GPUs available, requested {num_gpus}")
    
    return available_gpus


def create_gpu_specific_config(gpu_id: int, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create GPU-specific configuration by modifying device settings.
    
    Args:
        gpu_id: GPU ID to use
        base_config: Base configuration dictionary
        
    Returns:
        GPU-specific configuration
    """
    config = base_config.copy()
    config['device'] = f"cuda:{gpu_id}"
    config['gpu_id'] = gpu_id
    return config


def run_gpu_evaluation(gpu_id: int, config: Dict[str, Any]) -> str:
    """
    Run evaluation on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use
        config: Configuration dictionary
        
    Returns:
        Path to the result file
    """
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from evaluate import generate_coco_30k, cal_lpips_coco, cal_clip_score_coco
    from diffusers import StableDiffusionPipeline
    import subprocess
    
    device = f"cuda:{gpu_id}"
    print(f"Starting evaluation on GPU {gpu_id}...")
    
    # Load the pipeline on the specific GPU
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    # Generate images for this GPU's data chunk
    generate_coco_30k(
        pipe=pipe,
        penalty_param=config.get('penalty_param', 0.0),
        riemann=config.get('riemann', False),
        sample_num=config.get('sample_num', 1),
        file_path=config['gpu_csv_path'],
        out_dir=config['gpu_output_dir'],
        random_order=config.get('random_order', True),
        batch_size=config.get('batch_size', 4)
    )
    
    # Calculate LPIPS scores
    cal_lpips_coco(
        hparam_name=config.get('hparam_name', 'sd_orig'),
        num_edit=config.get('num_edit'),
        mom2_weight=config.get('mom2_weight'),
        edit_weight=config.get('edit_weight'),
        sample_num=config.get('sample_num', 1),
        edited_path=config['gpu_output_dir'],
        output_folder=config['gpu_results_dir'],
        dataset=config.get('dataset', 'artists'),
        original_path=config.get('original_path'),
        csv_path=config['gpu_csv_path'],
        device=device
    )
    
    # Calculate CLIP scores
    cal_clip_score_coco(
        hparam_name=config.get('hparam_name', 'sd_orig'),
        num_edit=config.get('num_edit'),
        mom2_weight=config.get('mom2_weight'),
        edit_weight=config.get('edit_weight'),
        sample_num=config.get('sample_num', 1),
        edited_path=config['gpu_output_dir'],
        out_put_folder=config['gpu_results_dir'],
        dataset=config.get('dataset', 'artists'),
        csv_path=config['gpu_csv_path'],
        device=device,
        vit_type=config.get('vit_type', 'large')
    )
    
    # Calculate FID score
    fid_output_file = os.path.join(config['gpu_results_dir'], 'coco_summary.json')
    subprocess.run([
        "python", "test_fid_score.py",
        "--generated_images_folder", config['gpu_output_dir'],
        "--coco_30k_folder", config.get('coco_30k_folder', 'data/coco/coco-30k'),
        "--output_folder", config['gpu_results_dir'],
        "--batch_size", str(config.get('fid_batch_size', 32)),
        "--output_file", "coco_summary.json",
        "--save_npz_folder", f"data/stats/fid_gpu_{gpu_id}",
        "--device", device,
        "--dict_key", config.get('hparam_name', 'sd_orig')
    ])
    
    print(f"GPU {gpu_id} evaluation completed. Results saved to {fid_output_file}")
    return fid_output_file
