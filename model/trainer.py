import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 导入自定义模块
from config import get_config
from HMNPs import HMNP
from MFD import MultimodalFeatureDecoupling
from Loss_Dec import DecouplingLoss
from Loss_ELBO import Loss_elbo


class HMNPTrainer:
    def __init__(self):
        # 初始化配置
        self.epochs = 100
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化模型
        self._init_models()

        # 初始化优化器和损失函数
        self._init_optimizer()
        self._init_loss_functions()

        # 训练参数

        self.batch_size = 8
        self.n_context = 10  # 上下文点数量
        self.n_target = 15  # 目标点数量
        self.num_modals = self.config.num_modality  # 模态数量

        # 训练记录
        self.train_log = {
            "epoch": [],
            "total_loss": [],
            "ce_loss": [],
            "kl_loss": [],
            "dec_loss": []
        }

    def _init_models(self):
        """初始化主模型和特征解耦模块"""
        # HMNP主模型
        self.hmnp_model = HMNP(self.config).to(self.device)

        # 多模态特征解耦模块
        self.mfd_model = MultimodalFeatureDecoupling(
            in_dim=self.config.dim_modality
        ).to(self.device)

        # 设置训练模式
        self.hmnp_model.train()
        self.mfd_model.train()

    def _init_optimizer(self):
        """初始化优化器"""
        # 联合优化所有参数
        params = list(self.hmnp_model.parameters()) + list(self.mfd_model.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=1e-4,
            weight_decay=1e-5
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )

    def _init_loss_functions(self):
        """初始化损失函数"""
        self.dec_criterion = DecouplingLoss().to(self.device)

    def _generate_sim_data(self, batch_size):
        """
        生成仿真多模态数据
        Returns:
            modal_features: 多模态特征列表 [(B, N, D), ...]
            Y_C: 上下文标签字典
            Y_D: 目标标签字典
        """
        n_total = self.n_context + self.n_target
        dim_modality = self.config.dim_modality

        # 1. 生成多模态特征
        modal_features = []
        for _ in range(self.num_modals):
            # 生成带模态特征的随机数据，添加轻微模态特异性
            feat = torch.randn(batch_size, n_total, dim_modality, device=self.device)
            modal_features.append(feat)

        # 2. 生成3分类标签 (Workload和Vigilance都是3分类)
        Y_C = {
            'Workload': torch.randint(0, 3, (batch_size, self.n_context), device=self.device),
            'Vigilance': torch.randint(0, 3, (batch_size, self.n_context), device=self.device)
        }

        Y_D = {
            'Workload': torch.randint(0, 3, (batch_size, self.n_target), device=self.device),
            'Vigilance': torch.randint(0, 3, (batch_size, self.n_target), device=self.device)
        }

        return modal_features, Y_C, Y_D

    def _process_features(self, modal_features):
        """
        处理多模态特征：解耦并融合
        Args:
            modal_features: 原始多模态特征列表 [(B, N, D), ...]
        Returns:
            f_fus: 融合后的特征 (B, N, new_dim)
            dec_loss: 解耦损失值
        """
        batch_size, n_total, dim_modality = modal_features[0].shape

        # 1. 展平特征以适配MFD模块输入
        modal_features_flat = [
            feat.reshape(batch_size * n_total, dim_modality)
            for feat in modal_features
        ]

        # 2. 特征解耦
        shared_feats, spe_feats, f_fus = self.mfd_model(modal_features_flat)

        # 3. 计算解耦损失
        dec_loss, _ = self.dec_criterion(shared_feats, spe_feats)

        # 4. 恢复特征形状
        new_dim = f_fus.size(-1)
        f_fus = f_fus.reshape(batch_size, n_total, new_dim)

        # 5. 更新config的dim_x（适配融合后的维度）
        self.config.dim_x = new_dim

        return f_fus, dec_loss

    def _train_step(self):
        """单步训练"""
        self.optimizer.zero_grad()

        # 1. 生成仿真数据
        modal_features, Y_C, Y_D = self._generate_sim_data(self.batch_size)

        # 2. 特征解耦和融合
        f_fus, dec_loss = self._process_features(modal_features)

        # 3. 分割上下文/目标集
        X_C = f_fus[:, :self.n_context, :]  # 上下文特征
        X_D = f_fus[:, self.n_context:, :]  # 目标特征

        # 4. HMNP前向传播
        outputs = self.hmnp_model(X_C, Y_C, X_D, Y_D)
        p_Y, q_D_G, q_C_G, q_D_T, q_C_T = outputs

        # 5. 计算ELBO损失
        ce_loss, kl_loss = Loss_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T)

        # 6. 总损失计算
        total_loss = (
                ce_loss +
                self.config.lambda_KL * kl_loss +
                self.config.lambda_Dec * dec_loss
        )

        # 7. 反向传播
        total_loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(
            list(self.hmnp_model.parameters()) + list(self.mfd_model.parameters()),
            max_norm=1.0
        )

        self.optimizer.step()

        # 返回损失值（用于记录）
        return {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "dec_loss": dec_loss.item()
        }

    def train(self):
        """主训练循环"""
        print(f"\n开始训练 HMNP+MFD 模型 (epochs={self.epochs})")
        print("=" * 80)

        start_time = datetime.now()

        for epoch in tqdm(range(self.epochs), desc="训练进度"):
            # 单轮训练
            loss_dict = self._train_step()

            # 更新学习率
            self.scheduler.step()

            # 记录训练日志
            self.train_log["epoch"].append(epoch + 1)
            self.train_log["total_loss"].append(loss_dict["total_loss"])
            self.train_log["ce_loss"].append(loss_dict["ce_loss"])
            self.train_log["kl_loss"].append(loss_dict["kl_loss"])
            self.train_log["dec_loss"].append(loss_dict["dec_loss"])

            # 每10轮打印一次信息
            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                tqdm.write(
                    f"\nEpoch [{epoch + 1}/{self.epochs}] | "
                    f"Total Loss: {loss_dict['total_loss']:.4f} | "
                    f"CE Loss: {loss_dict['ce_loss']:.4f} | "
                    f"KL Loss: {loss_dict['kl_loss']:.4f} | "
                    f"Dec Loss: {loss_dict['dec_loss']:.4f} | "
                    f"LR: {lr:.6f}"
                )

        # 训练结束
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print(f"训练完成！总耗时: {training_time:.2f} 秒")
        print(f"最终总损失: {self.train_log['total_loss'][-1]:.4f}")
        print("=" * 80)

        # 保存模型
        self.save_model()

        # 打印训练日志
        self.print_training_summary()

    def save_model(self):
        """保存训练好的模型"""
        save_path = f"hmnp_mfd_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

        torch.save({
            "epoch": self.epochs,
            "hmnp_state_dict": self.hmnp_model.state_dict(),
            "mfd_state_dict": self.mfd_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_log": self.train_log,
            "config": self.config
        }, save_path)

        print(f"\n模型已保存至: {save_path}")

    def print_training_summary(self):
        """打印训练总结"""
        print("\n=== 训练总结 ===")
        print(f"平均总损失: {np.mean(self.train_log['total_loss']):.4f}")
        print(f"平均CE损失: {np.mean(self.train_log['ce_loss']):.4f}")
        print(f"平均KL损失: {np.mean(self.train_log['kl_loss']):.4f}")
        print(f"平均解耦损失: {np.mean(self.train_log['dec_loss']):.4f}")

        # 损失下降率
        initial_loss = self.train_log['total_loss'][0]
        final_loss = self.train_log['total_loss'][-1]
        drop_rate = (initial_loss - final_loss) / initial_loss * 100
        print(f"总损失下降率: {drop_rate:.2f}%")


if __name__ == "__main__":
    # 初始化训练器并开始训练
    trainer = HMNPTrainer()
    trainer.train()
