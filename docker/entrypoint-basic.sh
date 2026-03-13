#!/bin/bash
echo "=== Flame Dev Container (Base) ==="

echo "Verifying environment..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || true
python -c "import triton; print(f'  Triton: {triton.__version__}')"
python -c "import fla; print('  FLA: OK')" 2>/dev/null || echo "  FLA: not installed or needs GPU"
python -c "import torchtitan; print('  TorchTitan: OK')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"

exec "$@"