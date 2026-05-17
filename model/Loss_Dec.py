import torch
import torch.nn as nn
import torch.nn.functional as F


class DecouplingLoss(nn.Module):
    """
    多模态特征解耦损失函数
    计算：共享特征对齐损失 + 专属特征正交损失
    总损失 L_Dec = L_sha + L_spe
    """
    def __init__(self):
        super().__init__()

    def forward(self, shared_features: list[torch.Tensor], spe_features: list[torch.Tensor]):
        """
        Args:
            shared_features:  各模态共享特征列表 [F_1_sha, F_2_sha, ...]
            spe_features:     各模态专属特征列表 [F_1_spe, F_2_spe, ...]
        Returns:
            total_loss: L_Dec = L_sha + L_spe
            loss_dict:  各部分损失（方便日志打印）
        """
        # -------------------- 1. 计算共享特征损失 L_sha --------------------
        # 目标：让所有模态的共享特征尽量相似 → 余弦相似度最大化
        shared_tensor = torch.stack(shared_features, dim=1)  # [B, num_modals, D]
        L_sha = self._compute_similarity_loss(shared_tensor, maximize=True)

        # -------------------- 2. 计算专属特征损失 L_spe --------------------
        # 目标：让所有模态的专属特征尽量正交 → 余弦相似度最小化
        spe_tensor = torch.stack(spe_features, dim=1)         # [B, num_modals, D]
        L_spe = self._compute_similarity_loss(spe_tensor, maximize=False)

        # -------------------- 3. 总解耦损失 --------------------
        L_Dec = L_sha + L_spe

        loss_dict = {
            "L_sha": L_sha.item(),
            "L_spe": L_spe.item(),
            "L_Dec": L_Dec.item()
        }

        return L_Dec, loss_dict

    def _compute_similarity_loss(self, feat_tensor: torch.Tensor, maximize: bool):
        """
        核心：计算模态间的成对余弦相似度，生成损失
        Args:
            feat_tensor: [B, num_modals, D]
            maximize:    True → 最大化相似度（共享特征）
                         False → 最小化相似度（专属特征）
        """
        B, M, D = feat_tensor.shape
        if M < 2:
            return torch.tensor(0.0, device=feat_tensor.device)  # 单模态无成对损失

        # 归一化特征（余弦相似度需要）
        feat_norm = F.normalize(feat_tensor, p=2, dim=-1)  # [B, M, D]

        # 计算两两模态间余弦相似度 [B, M, M]
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))

        # 去掉对角线（自己和自己的相似度）
        mask = torch.eye(M, device=feat_tensor.device, dtype=torch.bool).unsqueeze(0)
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)

        # 计算所有成对相似度均值
        mean_sim = sim_matrix.sum() / (B * M * (M - 1))

        # 构造损失
        if maximize:
            # 共享特征：损失 = 1 - 平均相似度 → 让相似度趋近于1
            loss = 1.0 - mean_sim
        else:
            # 专属特征：损失 = 平均相似度 → 让相似度趋近于0
            loss = mean_sim

        return loss
