#!/bin/bash
echo "=== Flame Dev Container (Flame) ==="

# --- Environment verification ---
echo "Verifying environment..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || true
python -c "import triton; print(f'  Triton: {triton.__version__}')"
python -c "import torchtitan; print('  TorchTitan: OK')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"

# --- Install flame ---
FLAME_DIR="/home/r00914194/flame"

if [ -f "$FLAME_DIR/setup.py" ] || [ -f "$FLAME_DIR/pyproject.toml" ]; then
    echo "Installing flame in editable mode..."
    pip install -e "$FLAME_DIR" -q --no-build-isolation
    echo "  Flame: installed (editable)"
else
    echo "  Warning: Flame source not found at $FLAME_DIR"
    echo "  Make sure to mount: -v /r00914194/flame:/home/r00914194/flame"
fi

exec "$@"