#!/bin/bash
#SBATCH -J different kapa
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --ntasks-per-node=3
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
deephall 'system.nspins=[6,0]' system.flux=15 system.interaction_strength=0.5 batch_size=256 optim.iterations=20000
# deephall 'system.nspins=[6,0]' system.flux=15 system.interaction_strength=1 batch_size=256 optim.iterations=50000
deephall 'system.nspins=[6,0]' system.flux=15 system.interaction_strength=3 batch_size=256 optim.iterations=50000
deephall 'system.nspins=[6,0]' system.flux=15 system.interaction_strength=10 batch_size=256 optim.iterations=100000