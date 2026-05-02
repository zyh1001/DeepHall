#!/bin/bash
#SBATCH -J debug_ckpt
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --ntasks-per-node=12
#SBATCH --partition=v100g32
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 加载 Python 环境
source /data/home/zyh/miniconda3/etc/profile.d/conda.sh
conda activate deephall

# 运行 DeepHall 测试
python ckpt_debug.py \
  --ckpt DeepHall_n6l18_batch6000_ckpt/ckpt_082319.npz \
  --config DeepHall_n6l18_batch6000_ckpt/config.yml \
  --out DeepHall_n6l18_batch6000_ckpt/energy.npy