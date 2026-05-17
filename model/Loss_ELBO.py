import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal


def Loss_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T):
    '''
    Compute (prior-approximated) elbo objective for NP-based models.
    '''
    log_prob = 0
    for task in p_Y:
        mu, _ = p_Y[task]  # shape: [B, N, C] = [4,15,2]
        y_true = Y_D[task]  # shape: [B, N]    = [4,15]

        log_prob_ = F.cross_entropy(
            mu.permute(0, 2, 1),  # 变成 [4, 2, 15]
            y_true  # 保持 [4, 15]
        )
        log_prob += log_prob_

    kld_G = 0
    if q_D_G is not None:
        kld_G = kl_divergence(Normal(*q_D_G), Normal(*q_C_G)).mean(0).sum()

    kld_T = 0
    if q_D_T is not None:
        for task in q_D_T:
            kld_T_ = kl_divergence(Normal(*q_D_T[task]), Normal(*q_C_T[task])).mean(0).sum()
            kld_T += kld_T_

    Loss_KL = kld_G + kld_T
    Loss_CE = log_prob

    return Loss_CE, Loss_KL

