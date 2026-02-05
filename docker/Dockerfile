# ============================================================================
# Dockerfile for Flame (fla-org/flame) - Flash Linear Attention Training
# ============================================================================
# 开发模式: Flame 源码通过 volume 挂载，支持实时修改
#
# 基础镜像: nvcr.io/nvidia/pytorch:25.06-py3
# - PyTorch 2.8.x (预发布版本)
# - CUDA 12.9.1
# - Python 3.12
# - Ubuntu 24.04
#
# 宿主机要求 (Ubuntu 22.04.5 LTS, kernel 5.15.0-141-generic):
# - NVIDIA Driver >= 570 (推荐) 或数据中心GPU可用 R535+/R545+
# - Docker with NVIDIA Container Toolkit
#
# 构建命令:
#   docker build -t flame:latest .
#
# 运行命令:
#   docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#       -v /home/a84400789/flame:/home/a84400789/flame \
#       -it flame:latest
# ============================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}

LABEL maintainer="a84400789"
LABEL description="Flame: Flash Language Modeling Made Easy - Dev Mode"
LABEL version="0.0"

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ============================================================================
# 系统依赖安装
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    vim \
    htop \
    tmux \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN git lfs install

# ============================================================================
# 处理 NGC 容器的 pip constraints 文件
# ============================================================================
RUN if [ -f /etc/pip/constraint.txt ]; then \
        cp /etc/pip/constraint.txt /etc/pip/constraint.txt.bak; \
    fi

# ============================================================================
# 安装 Python 依赖
# ============================================================================
RUN pip install --upgrade pip setuptools wheel

# Hugging Face 生态系统
RUN pip install --upgrade \
    transformers \
    datasets \
    accelerate \
    tokenizers \
    huggingface_hub \
    safetensors

# 训练监控工具
RUN pip install --upgrade \
    wandb \
    tensorboard

# 其他依赖
RUN pip install --upgrade \
    einops \
    ninja \
    packaging \
    pyyaml \
    tomli \
    toml \
    tyro

# ============================================================================
# 安装 flash-linear-attention (fla)
# ============================================================================
RUN pip uninstall -y flash-linear-attention fla-core 2>/dev/null || true && \
    pip install -U --no-build-isolation git+https://github.com/fla-org/flash-linear-attention

# ============================================================================
# [重要] 安装指定版本的 torchtitan (commit 0b44d4c)
# ============================================================================
RUN pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c

# ============================================================================
# 创建目录结构 (与宿主机路径一致)
# ============================================================================
RUN mkdir -p /home/a84400789/flame /home/a84400789/.cache/huggingface

# ============================================================================
# 设置环境变量
# ============================================================================
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=0
ENV NCCL_P2P_DISABLE=0
ENV TRITON_CACHE_DIR=/tmp/.triton
ENV HF_HOME=/home/a84400789/.cache/huggingface
ENV NGPU=8

# ============================================================================
# 创建 entrypoint 脚本 - 启动时执行 editable install
# ============================================================================
COPY <<'EOF' /entrypoint.sh
#!/bin/bash
echo "=== Flame Dev Container ==="

# 验证环境
echo "Verifying environment..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null || true
python -c "import triton; print(f'  Triton: {triton.__version__}')"
python -c "import fla; print('  FLA: OK')" 2>/dev/null || echo "  FLA: Failed (may need GPU)"
python -c "import torchtitan; print('  TorchTitan: OK')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"

# install flame in editable mode 
FLAME_DIR="/home/a84400789/flame"

if [ -f "$FLAME_DIR/setup.py" ] || [ -f "$FLAME_DIR/pyproject.toml" ]; then
    echo "Installing flame in editable mode (pip install -e) ..."
    pip install -e "$FLAME_DIR" -q --no-build-isolation
    echo "Done. Source code changes will take effect immediately."
else
    echo "Warning: Flame source not found at $FLAME_DIR"
    echo "Make sure to mount: -v /home/a84400789/flame:/home/a84400789/flame"
fi

exec "$@"
EOF
RUN chmod +x /entrypoint.sh

# ============================================================================
# 设置工作目录
# ============================================================================
WORKDIR /home/a84400789/flame

# ============================================================================
# 验证基础安装
# ============================================================================

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
