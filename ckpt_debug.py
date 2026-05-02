# compute_checkpoint_energies.py
import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.colors as colors  # 【新增这一行导入】


from deephall.config import Config
from deephall.hamiltonian import local_energy
from deephall.networks import make_network


def load_checkpoint(ckpt_path: str):
    with np.load(ckpt_path, allow_pickle=True) as f:
        params = f["params"].tolist()
        # 提取第 1921 个 walker 的构型，shape 变为 (1, n_elec, 2)
        data = f["data"][1921][None, ...]
    return params, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Checkpoint file: ckpt_XXXXX.npz")
    p.add_argument("--config", required=True, help="Config YAML from训练目录/config.yml")
    p.add_argument(
        "--out",
        default="energies.npy",
        help="输出文件 (numpy .npy)，每个 walker 的能量",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=500,
        help="如果样本很多，可分批计算以防止内存过大",
    )
    args = p.parse_args()

    # 需要在能找到 Vc_n*.npy / r_n*.npy 的目录下运行
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    cfg = OmegaConf.load(args.config)
    cfg = Config.from_dict(cfg)

    model = make_network(cfg.system, cfg.network)
    net_apply = model.apply

    params, data = load_checkpoint(args.ckpt)
    n_walkers = int(data.shape[0])

    # local_energy 本身生成单点能量。vmap 过后得到每个 walker 的能量。
    energy_fn = local_energy(net_apply, cfg.system)
    energy_vmap = jax.vmap(energy_fn, in_axes=(None, 0))

    energies = []
    observables = []

    for i in range(0, n_walkers, args.batch):
        batch = data[i : i + args.batch]
        e_batch, obs_batch = energy_vmap(params, batch)
        energies.append(np.array(jax.device_get(e_batch)))
        observables.append(jax.tree_map(lambda x: np.array(jax.device_get(x)), obs_batch))

    energies = np.concatenate(energies, axis=0)
    np.save(args.out, energies)

    print(f"计算完成：{n_walkers} 个构型，能量保存在 {args.out}")
    print("能量示例 (real part)：", energies.real[:10])

    # ==========================================
    # 画出波函数的模方 |\psi|^2，移动索引为 1 的电子
    # ==========================================
    print("\n开始计算网格上的波函数 log 值...")
    
    # 1. 生成二维网格
    grid_size = 1000  # 网格分辨率 100x100
    x_vals = np.linspace(-6, 6, grid_size)
    y_vals = np.linspace(-6, 6, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape: (1000000, 2)
    
    # 2. 准备网络输入数据
    base_config = data[0] 
    n_elec = base_config.shape[0]
    
    # 复制 1000000 份基准构型
    batched_grid_data = np.tile(base_config, (len(grid_points), 1, 1))
    
    moving_electron_idx = 1
    
    # 提取并保存移动电子的原始位置，用于后续画蓝点
    original_moving_pos = base_config[moving_electron_idx].copy()
    
    # 将这 10000 个构型中的第 1 号电子的坐标替换为网格点坐标
    batched_grid_data[:, moving_electron_idx, :] = grid_points
    
    # 3. 批量计算 log \psi
    logpsi_vmap = jax.vmap(net_apply, in_axes=(None, 0))
    
    logpsi_vals = []
    for i in range(0, len(batched_grid_data), args.batch):
        batch = batched_grid_data[i : i + args.batch]
        logpsi_batch = logpsi_vmap(params, batch)
        logpsi_vals.append(np.array(jax.device_get(logpsi_batch)))
        
    logpsi_vals = np.concatenate(logpsi_vals, axis=0)
    
    # 恢复成 2D 网格形状
    logpsi_grid = logpsi_vals.reshape(grid_size, grid_size)
    
    # 提取除移动电子以外的所有固定电子坐标
    fixed_electrons = np.delete(base_config, moving_electron_idx, axis=0)
    
    # 4. 保存计算结果
    np.savez(
        "wavefunction_grid.npz", 
        X=X, 
        Y=Y, 
        logpsi=logpsi_grid, 
        fixed_electrons=fixed_electrons,
        original_moving_pos=original_moving_pos
    )
    print("波函数计算结果已保存至 wavefunction_grid.npz")
    

    # 5. 绘制图像
    plt.figure(figsize=(8, 6))
    
    logpsi_real = np.real(logpsi_grid)
    
    #Z = np.exp(2 * logpsi_real)
    Z = logpsi_real
    
    # 【核心修改：使用 norm=colors.LogNorm】
    # vmin=1e-4 表示我们将颜色条的下限设置在最大值的万分之一处
    mesh = plt.pcolormesh(X, Y, Z, shading='auto', cmap='magma')
    
    plt.colorbar(mesh, label=r'Re $\log{\psi}$')
    
    # 把其余固定的电子位置用红点标出来
    fixed_x = fixed_electrons[:, 0]
    fixed_y = fixed_electrons[:, 1]
    plt.scatter(fixed_x, fixed_y, color='red', marker='o', edgecolors='white', s=30, label='Fixed Electrons')
    
    # 把移动电子的原始位置用蓝点标出来
    plt.scatter(original_moving_pos[0], original_moving_pos[1], color='cyan', marker='o', edgecolors='white', s=30, label='Original Pos (Moving)')
    
    plt.title(f"Probability Density $Re(log\\psi)|$ (Electron {moving_electron_idx} moving)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.legend()
    
    plt.savefig("wavefunction_prob_density_log_1921.png", dpi=300, bbox_inches='tight')
    print("对数尺度的波函数模方图像已保存至 wavefunction_prob_density_log.png")

if __name__ == "__main__":
    main()