#!/bin/bash
#SBATCH -J 6spins_lr1e-3
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --ntasks-per-node=12
#SBATCH --partition=v100
#SBATCH --gres=gpu:4
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 加载 Python 环境
source /data/home/zyh/miniconda3/etc/profile.d/conda.sh
conda activate deephall

# 运行 DeepHall 测试
deephall 'system.nspins=[6,0]' system.flux=18 batch_size=512 optim.optimizer=kfac optim.iterations=500000 system.d=0.1
