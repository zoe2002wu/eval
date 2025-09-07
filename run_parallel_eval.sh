#!/bin/bash

# Launch script for parallel COCO evaluation across 4 GPUs
# Usage: ./run_parallel_eval.sh [additional_args]

set -e

echo "Starting parallel COCO evaluation on 4 GPUs..."

# Default arguments
DEFAULT_ARGS="--num_gpus 4 --sample_num 1 --batch_size 4 --fid_batch_size 32"

# Check if CUDA is available
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"; then
    echo "Error: CUDA not available or PyTorch not installed"
    exit 1
fi

# Run the parallel evaluation
python parallel_evaluate.py $DEFAULT_ARGS "$@"

echo "Parallel evaluation completed!"
