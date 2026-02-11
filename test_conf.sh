#!/bin/bash
#SBATCH -J test_conf
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
python conf_potential.py