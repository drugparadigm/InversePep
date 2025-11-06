#!/bin/bash
set -e

echo "Installing libgfortran..."
conda install -y -c conda-forge libgfortran

echo "Installing Flask..."
pip install flask

echo "Installing PyTorch (CUDA 11.6)..."
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu116

echo "Installing PyTorch Geometric dependencies..."
pip install torch_cluster==1.6.1 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu116.html
pip install torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1%2Bcu116.html
pip install torch_geometric==2.3.1

echo "Installing Transformers and ESM..."
pip install transformers
pip install fair-esm

echo "All dependencies installed successfully!"
