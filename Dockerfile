# =============================================================================
# UM (Uveal Melanoma) - Pathology Foundation Model Benchmark
#
# 基础镜像: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
#   (自带 CUDA 12.1 + cuDNN 8 + PyTorch 2.1，国内可直接拉取，解决原镜像拉取失败)
#
# 使用方法:
#   构建镜像:
#     docker build -t um-benchmark:latest .
#
#   运行容器（挂载数据卷 + 使用 GPU）:
#     docker run --gpus all -it --rm ^
#       -v D:/Datas:/data ^
#       -v %cd%/output:/workspace/UM/output ^
#       um-benchmark:latest
#
#   容器内运行横评:
#     cd /workspace/UM/code/train
#     python run_benchmark.py
#
# 注意事项:
#   1. 特征文件和临床表通过 -v 挂载到容器中
#   2. 需在 config/benchmark.yml 中修改路径为容器内路径（如 /data/UVM/...）
#   3. openslide 需要宿主机的 GPU 驱动支持
# =============================================================================

# 修改1：替换为国内可拉取的基础镜像（自带CUDA12.1+torch2.1）
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

LABEL maintainer="UM-Project"
LABEL description="Uveal Melanoma - Pathology Foundation Model Benchmark"
LABEL version="1.0"

# ────────────────────────────────────────────────────────────────────────────
# 系统依赖 + 更换阿里云APT源（解决国内apt下载慢/失败）
# ────────────────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 修改2：替换为国内阿里云源，加速系统安装
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    openslide-tools \
    libopenslide-dev \
    libopenslide0 \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────────────────
# Python 环境配置（保留你的venv隔离环境）
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple  

# ────────────────────────────────────────────────────────────────────────────
# 安装 Python 依赖（删除重复的torch安装，全部换清华源）
# ────────────────────────────────────────────────────────────────────────────

# 修改4：删除了原有的 torch/torchvision 安装（镜像已自带，无需重复安装）

# 第 2 层：核心科学计算与数据处理
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    h5py>=3.8.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple  # 国内源

# 第 3 层：机器学习与评估
RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0 \
    lifelines>=0.27.0 \
    scikit-survival>=0.22.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第 4 层：可视化与文件输出
RUN pip install --no-cache-dir \
    matplotlib>=3.7.0 \
    openpyxl>=3.1.0 \
    Pillow>=10.0.0 \
    python-docx>=1.0.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第 5 层：病理图像读取 + MIL 架构 + 预训练模型工具
RUN pip install --no-cache-dir \
    openslide-python>=1.3.0 \
    einops>=0.7.0 \
    opt-einsum>=3.3.0 \
    timm>=0.9.0 \
    PyYAML>=6.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 修改5：安装你的requirements.txt（防止遗漏依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ────────────────────────────────────────────────────────────────────────────
# 复制项目代码（完全保留你的结构）
# ────────────────────────────────────────────────────────────────────────────
WORKDIR /workspace/UM
COPY code/ code/
COPY datasets/ datasets/
COPY README.md .

# ────────────────────────────────────────────────────────────────────────────
# 验证安装（完全保留你的验证脚本）
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -c "import torch; print('PyTorch', torch.__version__)" && \
    python3 -c "import torchvision; print('torchvision', torchvision.__version__)" && \
    python3 -c "import openslide; print('OpenSlide', openslide.__version__)" && \
    python3 -c "import h5py; print('h5py', h5py.__version__)" && \
    python3 -c "import einops; print('einops', einops.__version__)" && \
    python3 -c "import timm; print('timm', timm.__version__)" && \
    python3 -c "from lifelines import CoxPHFitter; print('lifelines OK')" && \
    python3 -c "from sksurv.metrics import cumulative_dynamic_auc; print('scikit-survival OK')" && \
    python3 -c "import matplotlib; print('matplotlib', matplotlib.__version__)" && \
    python3 -c "from openpyxl import Workbook; print('openpyxl OK')" && \
    echo "=== 所有依赖验证通过 ==="

# ────────────────────────────────────────────────────────────────────────────
# 工作目录与默认命令
# ────────────────────────────────────────────────────────────────────────────
WORKDIR /workspace/UM/code/train
CMD ["python3", "train.py"]