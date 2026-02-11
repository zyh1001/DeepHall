#!/usr/bin/env python3
"""
复现论文 "Describing Landau Level Mixing in Fractional Quantum Hall States with Deep Learning" 的主要结果
论文链接: https://doi.org/10.1103/PhysRevLett.134.176503

本脚本运行所有关键实验配置:
1. 1/3 FQHE: N=6 electrons, flux=15 (v=1/3)
2. 1/3 FQHE: N=10 electrons, flux=25 (v=1/3)  
3. 2/5 FQHE: N=6 electrons, flux=12 (v=2/5)
4. 2/5 FQHE: N=10 electrons, flux=25 (v=2/5)

使用的神经网络: PSIformer (Transformer-based wavefunction)
使用的优化器: KFAC (K-FAC natural gradient optimizer)
"""
'''
import json
import sys
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import logging

from deephall import Config, train

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_experiment(
    name: str,
    nspins: tuple[int, int],
    flux: int,
    filling: str,
    iterations: int = 1000,
    num_layers: int = 2,
    num_heads: int = 4,
    heads_dim: int = 64,
    batch_size: int = 3360,
    interaction_type: str = "coulomb",
) -> Config:
    """创建一个实验配置"""
    config = Config()
    
    # 系统配置
    config.system.nspins = nspins
    config.system.flux = flux
    config.system.interaction_type = interaction_type
    
    # 网络配置
    config.network.psiformer.num_layers = num_layers
    config.network.psiformer.num_heads = num_heads
    config.network.psiformer.heads_dim = heads_dim
    
    # 优化配置
    config.optim.iterations = iterations
    config.optim.optimizer = "kfac"
    config.optim.kfac.lr.rate = 0.05
    
    # 批次大小
    config.batch_size = batch_size
    
    # 输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{name}_{filling}_{timestamp}"
    config. = save_dir
    config.log.save_step_interval = 100
    
    logger.info(f"创建实验: {name}")
    logger.info(f"  电子数: {nspins}")
    logger.info(f"  通量 (2Q): {flux}")
    logger.info(f"  填充因子: {filling}")
    logger.info(f"  网络: {num_layers}层 PSIformer")
    logger.info(f"  优化步数: {iterations}")
    logger.info(f"  输出目录: {save_dir}")
    
    return config


def run_experiment(exp_name: str, config:Config) -> dict:
    """运行一个实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始实验: {exp_name}")
    logger.info(f"{'='*60}\n")
    
    try:
        train(config)
        logger.info(f"\n✓ 实验 {exp_name} 完成!")
        return {"status": "success", "name": exp_name, "config": config.}
    except Exception as e:
        logger.error(f"\n✗ 实验 {exp_name} 失败: {str(e)}")
        return {"status": "failed", "name": exp_name, "error": str(e)}

def one_three_filling():
    '''
    1/3 FQHE 实验,N=8, Q=10.5, kapa=1
    '''
    config = create_experiment(
        name="nu_1_3_N8",
        nspins=(8, 0),
        flux=21,  # 2Q = 21, Q = 10.5
        filling="1/3",
        iterations=100,  # 较少迭代用于测试
        num_layers=2,
        num_heads=4,
        heads_dim=64,
        batch_size=3360,
    )
    return ("1/3 FQHE (N=8)", config)

def two_five_filling():
    '''
    2/5 FQHE 实验,N=8, Q=8, kapa=1
    '''
    config = create_experiment(
        name="nu_2_5_N8",
        nspins=(8, 0),
        flux=16,  # 2Q = 16, Q = 8
        filling="1/3",
        iterations=500,  # 较少迭代用于测试
        num_layers=2,
        num_heads=4,
        heads_dim=64,
        batch_size=3360,
    )
    return ("2/5 FQHE (N=8)", config)  

def main():
    """主程序: 运行所有实验"""
    
    # 创建结果目录
    Path("results").mkdir(exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("DeepHall 论文复现脚本")
    logger.info("="*60 + "\n")
    
    # 定义所有实验
    experiments = []
    
    # 1/3 FQHE 实验
    experiments.append(one_three_filling)
    
    # 2/5 FQHE 实验: N=8, v=2/5
    # experiments.append(two_five_filling)

    '''
    # ============================================
    # Landau 能级混合对比实验
    # ============================================
    
    # 实验 5: Coulomb vs Harmonic 势 (1/3 FQHE)
    config_6_1_3_harmonic = create_experiment(
        name="nu_1_3_N6_harmonic",
        nspins=(6, 0),
        flux=15,
        filling="1/3_harmonic",
        iterations=500,
        interaction_type="harmonic",  # 使用 Haldane 伪势
    )
    experiments.append(("1/3 FQHE with Harmonic Potential (N=6)", config_6_1_3_harmonic))
    '''
    # ============================================
    # 运行所有实验
    # ============================================
    
    results = []
    name, config = one_three_filling()
    result = run_experiment(name, config)
    results.append(result)
    
    # ============================================
    # 输出总结
    # ============================================
    
    logger.info("\n" + "="*60)
    logger.info("实验汇总")
    logger.info("="*60 + "\n")
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    
    for i, result in enumerate(results, 1):
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"{i}. [{status_icon}] {result['name']}")
        if result["status"] == "success":
            logger.info(f"   结果保存在: {result['config']}")
        else:
            logger.info(f"   错误信息: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n成功: {success_count}/{len(results)}")
    logger.info(f"失败: {failed_count}/{len(results)}\n")
    
    # 保存结果汇总
    summary = {
        "total_experiments": len(results),
        "success": success_count,
        "failed": failed_count,
        "experiments": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_path = Path("results/experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"实验汇总已保存到: {summary_path}\n")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
'''
