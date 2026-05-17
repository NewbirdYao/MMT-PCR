import torch
import torch.nn as nn
import torch.nn.functional as F

from Loss_Dec import DecouplingLoss


class MultimodalFeatureDecoupling(nn.Module):
    """
    多模态特征解耦模块
    输入：m 个模态特征 [F_1, F_2, ..., F_m]
    输出：m 个 (共享特征, 专属特征) 对 [(F_1_sha, F_1_spe), ..., (F_m_sha, F_m_spe)]
    """
    def __init__(self, in_dim: int):
        """
        Args:
            in_dim: 输入特征的维度（所有模态特征维度必须一致）
        """
        super().__init__()
        # 单层 MLP：公式中 MLP(F_m)，无激活函数（严格对齐公式）
        self.mlp = nn.Linear(in_dim, in_dim, bias=True)

    def forward(self, features_list: list[torch.Tensor]):
        """
        Args:
            features_list: 列表形式输入，每个元素是一个模态特征 F_m
                           shape: [batch_size, in_dim]
        Returns:
            shared_features:  各模态共享特征列表 [F_1_sha, F_2_sha, ...]
            spe_features:     各模态专属特征列表 [F_1_spe, F_2_spe, ...]
        """
        shared_features = []
        spe_features = []

        # 逐个处理每个模态的特征 F_m
        for feat in features_list:
            # 1. MLP 映射 → M_m
            M_m = self.mlp(feat)

            # 2. 计算注意力分数
            M_m_sha = torch.sigmoid(M_m)    # 共享注意力
            M_m_spe = torch.sigmoid(-M_m)   # 专属注意力

            # 3. 逐元素相乘得到解耦特征
            F_sha = M_m_sha * feat    # 共享特征 F_m,sha
            F_spe = M_m_spe * feat   # 专属特征 F_m,spe

            shared_features.append(F_sha)
            spe_features.append(F_spe)

        F_sha_stack = torch.stack(shared_features, dim=0)
        F_sha_global = torch.mean(F_sha_stack, dim=0)

        fuse_concat_list = [F_sha_global] + spe_features
        F_fus = torch.cat(fuse_concat_list, dim=-1)

        return shared_features, spe_features, F_fus


if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 4
    FEAT_DIM = 128    # 每个模态特征维度
    NUM_MODALS = 4    # 模态数量
    TAU = 1.0

    # 构造模块
    model = MultimodalFeatureDecoupling(in_dim=FEAT_DIM)

    # 2. 构造解耦损失函数（新增）
    criterion = DecouplingLoss()

    # 构造输入：3个模态特征
    modal_inputs = [
        torch.randn(BATCH_SIZE, FEAT_DIM),
        torch.randn(BATCH_SIZE, FEAT_DIM),
        torch.randn(BATCH_SIZE, FEAT_DIM),
        torch.randn(BATCH_SIZE, FEAT_DIM)
    ]

    # 前向传播
    shared_feats, spe_feats, f_fus = model(modal_inputs)

    # 5. 计算解耦损失
    L_Dec, loss_dict = criterion(shared_feats, spe_feats)

    # 输出形状验证
    print("=== 输出形状 ===")
    print(f"单模态共享特征形状: {shared_feats[0].shape}")
    print(f"单模态专属特征形状: {spe_feats[0].shape}")
    print(f"全局共享特征形状: {torch.stack(shared_feats).mean(0).shape}")
    print(f"最终融合特征 F_fus 形状: {f_fus.shape}")
    print(f"融合特征维度 = 1×共享 + {NUM_MODALS}×专属 = {FEAT_DIM} + {NUM_MODALS}×{FEAT_DIM} = {f_fus.size(-1)}")

    # 6. 输出结果
    print("=== 解耦损失结果 ===")
    print(f"共享特征损失 L_sha: {loss_dict['L_sha']:.4f}")
    print(f"专属特征损失 L_spe: {loss_dict['L_spe']:.4f}")
    print(f"总解耦损失 L_Dec:   {loss_dict['L_Dec']:.4f}")