# config.py
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class AttnConfig:
    n_heads: int = 4
    act_fn: str = 'relu'  # 'relu' or 'gelu'
    ln: bool = False  # LayerNorm
    dr: float = 0.1  # Dropout


@dataclass
class ModelConfig:
    num_modality: int = 4  # 模态数量
    dim_modality: int = 128  # 各模态编码后特征维度
    dim_x: int = 5*128  # 输入特征维度（根据你的数据改）
    dim_hidden: int = 128  # 隐藏层维度
    dim_ys: dict = None  # 每个任务的输出维度
    lambda_Dec: float = 0.01
    lambda_KL: float = 0.01

    # 模块层数：[mlp_layers, attn_layers, global_attn_layers, decoder_layers]
    module_sizes: list = (2, 2, 2, 2)

    attn_config: AttnConfig = None


# 初始化配置
def get_config():
    attn_config = AttnConfig()

    config = ModelConfig(
        num_modality=4,
        dim_modality=128,
        dim_x=5*128,
        dim_hidden=128,
        dim_ys={
            'Workload': 3,  # 分类数：2类
            'Vigilance': 3,
        },
        module_sizes=(2, 2, 2, 2),
        attn_config=attn_config
    )
    return config
