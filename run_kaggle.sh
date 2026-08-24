#!/bin/bash
# VietAIDetector — Kaggle T4x2 Deployment Script
# Usage: !bash /kaggle/working/VietAIDetector/run_kaggle.sh

set -e  # Exit on error

echo "============================================="
echo "  VietAIDetector — Kaggle Setup Script"
echo "============================================="

# Step 1: Environment Variables
echo "[1/5] Configuring environment variables..."
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Step 2: Install Dependencies
echo "[2/5] Installing dependencies..."
pip install -q -r requirements.txt

# Step 3: Download Vietnamese Font
echo "[3/5] Downloading NotoSans font for PDF reports..."
FONT_PATH="/tmp/NotoSans-Regular.ttf"
# If font does not exist or is smaller than 50KB (e.g. corrupt HTML), download it
if [ ! -f "$FONT_PATH" ] || [ $(wc -c < "$FONT_PATH" 2>/dev/null || echo 0) -lt 50000 ]; then
    rm -f "$FONT_PATH"
    wget -q -O "$FONT_PATH" \
        "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf" \
        || wget -q -O "$FONT_PATH" \
        "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf" \
        || echo "WARNING: Could not download font. PDF reports will attempt auto-download in Python."
fi
export FONT_PATH="$FONT_PATH"

# Step 4: Verify GPU
echo "[4/5] Verifying GPU..."
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPU count: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
if n < 2:
    print('  WARNING: Only 1 GPU detected. Both models will share the same GPU.')
"

# Step 5: Launch Application
echo "[5/5] Launching VietAIDetector..."
echo "============================================="
echo "  Loading PhoGPT-4B models..."
echo "  (First run will take ~3-5 minutes to download)"
echo "============================================="

cd /kaggle/working/VietAIDetector
python3 app.py