#!/usr/bin/env python3
"""
Parallel evaluation script for running COCO evaluation across multiple GPUs.
"""
import os
import sys
import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from multi_gpu_utils import (
    split_data_for_gpus, 
    create_gpu_output_dirs, 
    aggregate_results, 
    check_gpu_availability,
    create_gpu_specific_config,
    run_gpu_evaluation
)
from util_global import DATA_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run COCO evaluation in parallel across multiple GPUs")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--coco_csv", type=str, default=str(DATA_DIR / "coco" / "coco_30k.csv"), 
                       help="Path to COCO CSV file")
    parser.add_argument("--output_dir", type=str, default="data/coco/images/sd_orig_parallel", 
                       help="Base output directory for generated images")
    parser.add_argument("--results_dir", type=str, default="results/sd_orig_parallel", 
                       help="Base results directory")
    parser.add_argument("--coco_30k_folder", type=str, default="data/coco/coco-30k", 
                       help="Path to COCO-30K reference images")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--fid_batch_size", type=int, default=32, help="Batch size for FID calculation")
    parser.add_argument("--penalty_param", type=float, default=0.0, help="Penalty parameter for Riemannian guidance")
    parser.add_argument("--riemann", action="store_true", help="Use Riemannian guidance")
    parser.add_argument("--vit_type", type=str, default="large", choices=["base", "large"], 
                       help="CLIP ViT model type")
    parser.add_argument("--dataset", type=str, default="artists", help="Dataset name for results")
    parser.add_argument("--hparam_name", type=str, default="sd_orig", help="Hyperparameter name")
    parser.add_argument("--num_edit", type=int, default=None, help="Number of edits")
    parser.add_argument("--mom2_weight", type=float, default=None, help="Momentum weight")
    parser.add_argument("--edit_weight", type=float, default=None, help="Edit weight")
    parser.add_argument("--random_order", action="store_true", default=True, help="Randomize data order")
    parser.add_argument("--skip_generation", action="store_true", help="Skip image generation, only compute metrics")
    parser.add_argument("--skip_aggregation", action="store_true", help="Skip result aggregation")
    
    args = parser.parse_args()
    
    print(f"Starting parallel evaluation with {args.num_gpus} GPUs...")
    print(f"Configuration: {vars(args)}")
    
    # Check GPU availability
    available_gpus = check_gpu_availability(args.num_gpus)
    if len(available_gpus) < args.num_gpus:
        print(f"Using {len(available_gpus)} GPUs instead of {args.num_gpus}")
        args.num_gpus = len(available_gpus)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Split data for each GPU
    print("Splitting data across GPUs...")
    split_files = split_data_for_gpus(
        csv_path=args.coco_csv,
        num_gpus=args.num_gpus,
        output_dir="data/coco/gpu_splits"
    )
    
    # Create GPU-specific output directories
    gpu_output_dirs = create_gpu_output_dirs(args.output_dir, args.num_gpus)
    gpu_results_dirs = create_gpu_output_dirs(args.results_dir, args.num_gpus)
    
    # Create base configuration
    base_config = {
        'hparam_name': args.hparam_name,
        'num_edit': args.num_edit,
        'mom2_weight': args.mom2_weight,
        'edit_weight': args.edit_weight,
        'sample_num': args.sample_num,
        'batch_size': args.batch_size,
        'fid_batch_size': args.fid_batch_size,
        'penalty_param': args.penalty_param,
        'riemann': args.riemann,
        'vit_type': args.vit_type,
        'dataset': args.dataset,
        'random_order': args.random_order,
        'coco_30k_folder': args.coco_30k_folder,
        'original_path': DATA_DIR / "coco" / "images" / "sd_orig"
    }
    
    # Create GPU-specific configurations
    gpu_configs = []
    for gpu_id in range(args.num_gpus):
        config = create_gpu_specific_config(gpu_id, base_config)
        config['gpu_csv_path'] = split_files[gpu_id]
        config['gpu_output_dir'] = gpu_output_dirs[gpu_id]
        config['gpu_results_dir'] = gpu_results_dirs[gpu_id]
        gpu_configs.append(config)
    
    if not args.skip_generation:
        # Run evaluation on each GPU in parallel
        print("Starting parallel evaluation...")
        start_time = time.time()
        
        # Use multiprocessing to run on different GPUs
        with mp.Pool(processes=args.num_gpus) as pool:
            result_files = pool.map(run_gpu_evaluation, 
                                  [(gpu_id, config) for gpu_id, config in enumerate(gpu_configs)])
        
        end_time = time.time()
        print(f"Parallel evaluation completed in {end_time - start_time:.2f} seconds")
        
        # Print individual GPU results
        for i, result_file in enumerate(result_files):
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    gpu_results = json.load(f)
                print(f"GPU {i} results: {gpu_results}")
    
    if not args.skip_aggregation:
        # Aggregate results from all GPUs
        print("Aggregating results...")
        result_files = [os.path.join(gpu_results_dirs[i], 'coco_summary.json') 
                       for i in range(args.num_gpus)]
        
        aggregated_file = os.path.join(args.results_dir, 'coco_summary_aggregated.json')
        aggregated_results = aggregate_results(result_files, aggregated_file)
        
        print("Final aggregated results:")
        print(json.dumps(aggregated_results, indent=2))
        
        # Also save a summary
        summary_file = os.path.join(args.results_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Parallel COCO Evaluation Summary\n")
            f.write(f"Number of GPUs: {args.num_gpus}\n")
            f.write(f"Total samples: {sum(len(pd.read_csv(sf)) for sf in split_files)}\n")
            f.write(f"Sample per GPU: {len(pd.read_csv(split_files[0]))}\n")
            f.write(f"Configuration: {vars(args)}\n\n")
            f.write(f"Aggregated Results:\n")
            f.write(json.dumps(aggregated_results, indent=2))
        
        print(f"Summary saved to {summary_file}")
    
    print("Parallel evaluation completed successfully!")


if __name__ == "__main__":
    main()
