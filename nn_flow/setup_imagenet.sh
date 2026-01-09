#!/bin/bash
set -e
set -o pipefail

# 配置目标路径
IMAGENET_DIR="/checkpoint/flows/mingyangd/data/imagenet"
mkdir -p "$IMAGENET_DIR"
cd "$IMAGENET_DIR"

# 检查 aria2 是否安装
if ! command -v aria2c &> /dev/null; then
    echo "aria2c not found. Try installing with: micromamba install -c conda-forge aria2"
    exit 1
fi

echo "开始下载 ImageNet train.tar 和 val.tar ..."

# 下载训练集
wget -O ILSVRC2012_img_train.tar \
  https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
aria2c -x 16 -s 16 -c -o ILSVRC2012_img_train.tar \
  https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# 下载验证集
aria2c -x 16 -s 16 -c -o ILSVRC2012_img_val.tar \
  http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

echo "下载完成，开始解压 train.tar ..."

# 解压训练集
mkdir -p train
tar -xf ILSVRC2012_img_train.tar -C train
cd train

# 并行解压每个类 tar 包
echo "开始并行解压每个类别（可能需几分钟）..."
ls n*.tar | xargs -P 32 -n 1 -I{} bash -c \
'folder="{}"; mkdir -p "${folder%.tar}"; tar -xf "$folder" -C "${folder%.tar}"; rm "$folder"'

cd "$IMAGENET_DIR"

echo "解压 val.tar ..."
mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val

echo "整理 val 数据到子目录 ..."

# 下载并运行官方分类脚本
wget -O valprep.sh https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh

echo "ImageNet 数据集准备完成 ✅"

# torchrun --nproc_per_node=8 dataset/cache_latent.py --data_path /datasets/imagenet_fullsize/061417/ --cached_path /checkpoint/flows/mingyangd/data/imagenet_cache_val --split val
# torchrun --nproc_per_node=8 --master_port=29501 dataset/cache_latent.py --data_path /datasets/imagenet_fullsize/061417/ --cached_path /checkpoint/flows/mingyangd/data/imagenet_cache_train --split train