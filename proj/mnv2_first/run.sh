#!/bin/bash

# Get the CFU-Playground root directory
CFU_PLAYGROUND="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$CFU_PLAYGROUND/proj/mnv2_first"

echo "=== CFU Playground ==="
echo "Root: $CFU_PLAYGROUND"
echo "Project: $PROJECT"
echo

echo "=== Activating CFU environment ==="
source "$CFU_PLAYGROUND/env/conda/bin/activate"

conda activate cfu-common || exit 1

echo "=== Entering MobileNetV2 project ==="
cd "$PROJECT" || exit 1

echo "=== Cleaning ==="
make clean

echo "=== Starting Renode ==="
make renode
