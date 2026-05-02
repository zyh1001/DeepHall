#!/bin/bash
#SBATCH -J netobs_analys
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

# overlap
# netobs deephall unused deephall@overlap --with steps=50 --net-restore DeepHall_n6l15step10000/ckpt_009999.npz --ckpt DeepHall_n6l15step10000/overlap
# pair_corr
# netobs deephall unused deephall@pair_corr --with steps=100000 --net-restore DeepHall_n6l15step10000/ckpt_009999.npz --ckpt DeepHall_n6l15step10000/pair_corr
# one_rdm
# netobs deephall unused deephall@density --with steps=20000 --net-restore DeepHall_n9l27/ckpt_299999.npz --ckpt DeepHall_n9l27/density
netobs deephall unused deephall@density --with steps=20000 --net-restore DeepHall_n12l36_s1500000/ckpt_1302999.npz --ckpt DeepHall_n12l36_s1500000/density