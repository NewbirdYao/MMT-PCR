import torch
import torch.nn as nn
from torch.distributions import Normal

# 导入模型和配置
from HMNPs import HMNP
from config import get_config
from MFD import MultimodalFeatureDecoupling
from Loss_Dec import DecouplingLoss
from Loss_ELBO import Loss_elbo


def main():
    # ============== 1. 初始化配置和模型 ==============
    config = get_config()
    model = HMNP(config)
    model.train()  # 先切换到训练模式

    # 初始化多模态特征解耦模块和损失函数
    num_modals = 4  # 设定模态数量，可根据实际需求调整
    mfd_model = MultimodalFeatureDecoupling(in_dim=config.dim_modality)  # MFD模块
    decoupling_criterion = DecouplingLoss()  # 解耦损失函数

    # 打印模型结构
    print("=" * 50)
    print("🔥 HMNP + MFD 模型初始化成功")
    print(f"任务列表: {model.tasks}")
    print(f"输入维度 dim_x: {config.dim_x}")
    print(f"隐藏维度 dim_hidden: {config.dim_hidden}")
    print(f"模态数量: {num_modals}")
    print("=" * 50)

    # ============== 2. 生成多模态随机模拟数据 ==============
    # 批次大小
    B = 4
    # 上下文点数量（支持集）
    n_context = 10
    # 目标点数量（查询集）
    n_target = 15
    # 总样本数 = 上下文点 + 目标点
    n_total = n_context + n_target

    # 生成多模态原始特征 [模态1, 模态2, 模态3]，每个模态形状: (B, n_total, dim_x)
    modal_features = [
        torch.randn(B, n_total, config.dim_modality) for _ in range(num_modals)
    ]

    # 通过MFD模块进行特征解耦
    # 先将每个模态特征展平为 (B*n_total, dim_x) 适配MFD输入格式
    modal_features_flat = [
        feat.reshape(B * n_total, config.dim_modality) for feat in modal_features
    ]
    shared_feats, spe_feats, f_fus = mfd_model(modal_features_flat)

    # 恢复f_fus形状为 (B, n_total, new_dim)，并分割为上下文/目标集
    new_dim = f_fus.size(-1)  # 融合特征维度 = dim_x + num_modals*dim_x
    f_fus = f_fus.reshape(B, n_total, new_dim)
    # 重新设置config的dim_x为融合后的维度（适配HMNP输入）
    config.dim_x = new_dim

    model = HMNP(config)
    model.train()

    # 分割上下文/目标集（替代原有X_C/X_D）
    X_C = f_fus[:, :n_context, :]  # (B, n_context, new_dim)
    X_D = f_fus[:, n_context:, :]  # (B, n_target, new_dim)

    # 分类任务输出 Y：类别标签 LongTensor，形状 (B, N)
    Y_C = {
        'Workload': torch.randint(0, 3, (B, n_context)),  # 0/1 二分类
        'Vigilance': torch.randint(0, 3, (B, n_context)),
    }
    Y_D = {
        'Workload': torch.randint(0, 3, (B, n_target)),
        'Vigilance': torch.randint(0, 3, (B, n_target)),
    }

    print("\n📊 多模态数据生成&解耦完成")
    print(f"原始单模态特征形状: {modal_features[0].shape}")
    print(f"融合特征 f_fus 形状: {f_fus.shape}")
    print(f"X_C (融合后) shape: {X_C.shape}")
    print(f"X_D (融合后) shape: {X_D.shape}")
    print(f"Y_C['Workload'] shape: {Y_C['Workload'].shape}")
    print(f"Y_D['Vigilance'] shape: {Y_D['Vigilance'].shape}")

    # ============== 3. 计算解耦损失 ==============
    L_Dec, loss_dict = decoupling_criterion(shared_feats, spe_feats)
    print("\n📉 解耦损失计算完成")
    print(f"共享特征损失 L_sha: {loss_dict['L_sha']:.4f}")
    print(f"专属特征损失 L_spe: {loss_dict['L_spe']:.4f}")
    print(f"总解耦损失 L_Dec:   {loss_dict['L_Dec']:.4f}")

    # ============== 4. 训练模式前向测试 ==============
    print("\n🚀 训练模式前向传播...")
    try:
        outputs = model(X_C, Y_C, X_D, Y_D)
        p_Y, q_D_G, q_C_G, q_D_T, q_C_T = outputs
        Loss_CE, Loss_KL = Loss_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T)

        # 总损失 = ELBO损失 + 解耦损失
        total_loss = Loss_CE + config.lambda_KL*Loss_KL + config.lambda_Dec*L_Dec

        print("✅ 训练模式前向成功！")
        print(f"输出 p_Y['Workload'][0] shape: {p_Y['Workload'][0].shape}")
        print(f"全局隐变量 q_D_G[0] shape: {q_D_G[0].shape}")
        print(f"任务隐变量 q_D_T['Workload'][0] shape: {q_D_T['Workload'][0].shape}")
        print(f"交叉熵损失 Loss_CE: {Loss_CE:.4f}")
        print(f"KL约束损失 Loss_KL: {Loss_KL:.4f}")
        print(f"解耦损失 L_Dec: {L_Dec:.4f}")
        print(f"总训练损失 total_loss: {total_loss:.4f}")

    except Exception as e:
        print(f"❌ 训练模式报错: {e}")
        raise e

    # ============== 5. 测试模式（采样）测试 ==============
    model.eval()
    print("\n🚀 测试模式（采样）前向传播...")
    with torch.no_grad():
        try:
            p_Y_eval = model(X_C, Y_C, X_D, MAP=False, ns_G=3, ns_T=2)
            print("✅ 测试（采样）模式前向成功！")
            print(f"测试输出 mu shape: {p_Y_eval['Workload'][0].shape}")
        except Exception as e:
            print(f"❌ 测试模式报错: {e}")
            raise e

    # ============== 6. 测试模式（MAP）测试 ==============
    print("\n🚀 MAP 推理模式前向传播...")
    with torch.no_grad():
        try:
            p_Y_map = model(X_C, Y_C, X_D, MAP=True)
            print("✅ MAP 推理模式前向成功！")
            print(f"MAP输出 mu shape: {p_Y_map['Workload'][0].shape}")
        except Exception as e:
            print(f"❌ MAP推理报错: {e}")
            raise e

    # ============== 7. 全部通过 ==============
    print("\n" + "=" * 50)
    print("🎉 所有代码运行正常！无任何错误！")
    print("=" * 50)


if __name__ == '__main__':
    main()