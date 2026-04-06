#!/bin/bash
# TRIBE v2 Experiment Toolkit - One-shot Setup
# Run with: bash setup.sh

set -e

echo "============================================"
echo "  TRIBE v2 Experiment Toolkit Setup"
echo "============================================"

# 1. Create conda environment
echo "[1/6] Creating conda environment..."
conda create -n tribe python=3.10 -y 2>/dev/null || echo "Conda env may already exist"
eval "$(conda shell.bash hook)"
conda activate tribe

# 2. Install PyTorch (CUDA 12.4)
echo "[2/6] Installing PyTorch..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install TRIBE v2 dependencies
echo "[3/6] Installing TRIBE v2 dependencies..."
pip install \
    transformers>=4.40.0 \
    accelerate \
    nilearn \
    nibabel \
    pyvista \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    pandas \
    pyyaml \
    tqdm \
    wandb \
    jupyter \
    ipywidgets

# 4. Clone TRIBE v2
echo "[4/6] Cloning TRIBE v2 repo..."
if [ ! -d "tribev2" ]; then
    git clone https://github.com/facebookresearch/tribev2.git
else
    echo "tribev2 already cloned"
fi

# Install tribev2 in editable mode if it has a setup.py/pyproject.toml
cd tribev2
pip install -e . 2>/dev/null || echo "No setup.py found, will use sys.path instead"
cd ..

# 5. Download model weights
echo "[5/6] Downloading TRIBE v2 weights from HuggingFace..."
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
import os

# Download TRIBE v2 model
print('Downloading TRIBE v2 model weights...')
print('(This may take 10-30 minutes depending on your connection)')
try:
    path = snapshot_download(
        'facebook/tribev2',
        local_dir='./weights/tribev2',
        ignore_patterns=['*.md', '*.txt']
    )
    print(f'Downloaded to: {path}')
except Exception as e:
    print(f'Auto-download failed: {e}')
    print('You may need to accept the license at https://huggingface.co/facebook/tribev2')
    print('Then run: huggingface-cli login')
"

# 6. Download Schaefer atlas for ROI extraction
echo "[6/6] Downloading Schaefer atlas..."
python -c "
from nilearn import datasets
print('Downloading Schaefer 1000-parcel atlas...')
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7)
print(f'Atlas downloaded to: {atlas[\"maps\"]}')
print('Network labels available:', len(atlas['labels']), 'parcels')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  conda activate tribe"
echo ""
echo "Run Experiment 0 (smoke test):"
echo "  python scripts/tribe_wrapper.py --test"
echo ""
echo "Or tell Claude Code:"
echo "  'Run Experiment 0 from EXPERIMENTS.md'"
echo ""
