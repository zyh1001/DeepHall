#!/bin/bash
#SBATCH -J deephall_test
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=p1
#SBATCH --gres=gpu:0
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 加载 Python 环境
source /data/home/zyh/miniconda3/etc/profile.d/conda.sh
conda activate deephall

# 运行 DeepHall 测试
deephall 'system.nspins=[6,0]' system.flux=18 batch_size=256 optim.iterations=10000 system.d=0.1
