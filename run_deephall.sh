#!/bin/bash
#SBATCH -J kfac_debug
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --ntasks-per-node=3
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 加载 Python 环境
source /data/home/zyh/miniconda3/etc/profile.d/conda.sh
conda activate deephall

# 运行 DeepHall 测试
deephall --debug 'system.nspins=[1,0]' system.flux=3 batch_size=8 optim.optimizer=adam optim.iterations=25 system.d=0.1
