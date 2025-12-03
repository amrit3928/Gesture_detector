#!/bin/bash

# Get the directory where this script is located (dataset/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go to the project root directory (one level up from dataset/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Define the paths
# Based on your unzip.sh, the data is in dataset/hagrid_light/hagrid_light
DATASET_PATH="dataset/hagrid_light/hagrid_light"
OUTPUT_PATH="data/processed"
PYTHON_SCRIPT="src/process_dataset.py"

echo "=================================================="
echo "Running Data Processing"
echo "Project Root: $PROJECT_ROOT"
echo "Dataset Path: $DATASET_PATH"
echo "Output Path:  $OUTPUT_PATH"
echo "=================================================="

# Check if the dataset directory exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found at $DATASET_PATH"
    echo "Please make sure you have unzipped the dataset using unzip.sh first."
    exit 1
fi

# Run the python script
# We use "python" here. If you use "python3", please change it.
python "$PYTHON_SCRIPT" --dataset_path "$DATASET_PATH" --output_path "$OUTPUT_PATH"
