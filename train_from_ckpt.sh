#!/bin/bash
#SBATCH -J train_from_ckpts1
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --ntasks-per-node=24
#SBATCH --partition=v100g32
#SBATCH --gres=gpu:8
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# 进入工作目录
cd $SLURM_SUBMIT_DIR

# 加载 Python 环境
source /data/home/zyh/miniconda3/etc/profile.d/conda.sh
conda activate deephall

# 运行 DeepHall 测试
deephall 'system.nspins=[12,0]' optim.optimizer=kfac \
    log.save_time_interval=1 \
    log.save_step_interval=100 \
    system.interaction_strength=0.33333333 \
    network.psiformer.envelope=no_m_learnable \
    system.flux=36 batch_size=4096 \
    optim.iterations=726000 system.d=0.1 \
    log.restore_path=DeepHall_n12l36_debug3/ckpt_720999.npz
