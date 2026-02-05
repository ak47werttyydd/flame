# Flame Docker 环境说明 (开发模式)

## 核心特性

- **源码挂载**: `/home/a84400789/flame` 直接映射到容器，修改代码即时生效
- **Editable Install**: 容器启动时自动执行 `pip install -e .`
- **缓存共享**: HuggingFace 缓存与宿主机共享，避免重复下载

## 宿主机兼容性

### 你的系统环境
- **OS**: Ubuntu 22.04.5 LTS
- **Kernel**: 5.15.0-141-generic (x86_64)

### 基础镜像
- **镜像**: `nvcr.io/nvidia/pytorch:25.06-py3`
- **CUDA**: 12.9.1
- **PyTorch**: 2.8.x
- **Python**: 3.12

### 驱动要求

| GPU 类型 | 最低驱动版本 |
|---------|------------|
| 消费级 GPU (RTX 系列) | **570+** |
| 数据中心 GPU (A100/H100/T4) | 535.86+ |

```bash
# 检查驱动版本
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

---

## 快速开始

### 1. 构建镜像

```bash
docker build -t flame:latest .
```

### 2. 运行容器

```bash
docker run --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/a84400789/flame:/home/a84400789/flame \
    -v /home/a84400789/.cache/huggingface:/home/a84400789/.cache/huggingface \
    -it flame:latest
```

或使用 docker-compose:

```bash
docker-compose up -d
docker-compose exec flame bash
```

### 3. 训练

容器启动后会自动执行 editable install，直接运行训练即可：

```bash
# 单 GPU 调试
NGPU=1 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/debug \
  --model.config configs/transformer_340M.json \
  --training.steps 100

# 8 GPU 训练
NGPU=8 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-340M \
  --model.config configs/transformer_340M.json \
  --optimizer.lr 1e-3 \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.steps 20480 \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.compile
```

---

## 开发工作流

1. 在宿主机编辑 `/home/a84400789/flame` 下的代码
2. 容器内直接运行，改动立即生效（无需重启容器）
3. 如果修改了 `setup.py` 或 `pyproject.toml`，重新执行：
   ```bash
   pip install -e . --no-deps
   ```

---

## 目录结构

```
/home/a84400789/flame/     # 工作目录 (挂载自宿主机)
├── configs/               # 模型配置
├── flame/                 # 核心代码
├── custom_models/         # 自定义模型
├── utils/                 # 工具脚本
├── exp/                   # 实验输出
└── train.sh               # 训练脚本

/home/a84400789/.cache/huggingface/  # HF 缓存 (挂载)
```

---

## 注意事项

1. **Volta (V100) 不支持**: NGC 25.01+ 不再支持 V100，如需使用请改用 `24.12-py3` 镜像
2. **源码必须存在**: 确保 `/home/a84400789/flame` 下有完整的 flame 源码
