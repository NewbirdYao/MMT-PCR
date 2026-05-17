import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from config import get_config
from HMNPs_modules import *


class HMNP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tasks = ['Workload', 'Vigilance']
        self.feat_select = nn.ModuleList([FeatureSelector(config.dim_x) for task in self.tasks])

        # Task-specific Encoders
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                     config.module_sizes[0], config.module_sizes[1], config.attn_config)
                                          for task in self.tasks])

        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, config.attn_config, hierarchical=True)
                                           for task in self.tasks])

        # Task-shared Encoder
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2], config.attn_config)

        # Task-specific Decoders
        self.decoder = nn.ModuleList(
            [TaskDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3],
                        config.attn_config, sigma=True)
             for task in self.tasks])

    def state_dict_(self):
        return self.state_dict()

    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)

    def encode_global(self, X, Y):
        w = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks):
            X = self.feat_select[t_idx](X)
            y = Y[task]  # shape [B, N]
            y_onehot = F.one_hot(y, num_classes=self.config.dim_ys[task])  # [B, N, dim_ys[task]]
            D_t = torch.cat((X, y_onehot), -1)  # 拼接X和one-hot编码的Y
            w[task] = self.set_encoder[t_idx](D_t)

        # global latent in across-task inference of latent path
        w_G = torch.stack([w[task] for task in w], 1)
        q_G = self.global_encoder(w_G)
        return q_G, w

    def encode_task(self, w, z):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            w_t = w[task]
            if not self.training:
                w_t = w_t.unsqueeze(1).repeat(1, z.size(1), 1)
            q_T[task] = self.task_encoder[t_idx](w_t, z)

        return q_T

    def decode(self, X, v):
        if not self.training:
            X = X.unsqueeze(1).repeat(1, v.size(2), 1, 1)

        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X, v[:, t_idx])
        return p_Y

    def forward(self, X_C, Y_C, X_D, Y_D=None, MAP=False, ns_G=5, ns_T=5):
        if self.training:
            assert Y_D is not None

            q_C_G, w_C = self.encode_global(X_C, Y_C)
            q_D_G, w_D = self.encode_global(X_D, Y_D)
            z = Normal(*q_D_G).rsample()

            q_C_T = self.encode_task(w_C, z)
            q_D_T = self.encode_task(w_D, z)
            v = torch.stack([Normal(*q_D_T[task]).rsample() for task in self.tasks], 1)

            p_Y = self.decode(X_D, v)

            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:
            q_C_G, w_C = self.encode_global(X_C, Y_C)
            if MAP:
                z = q_C_G[0].unsqueeze(1)
            else:
                z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)

            q_C_T = self.encode_task(w_C, z)
            if MAP:
                v = torch.stack([q_C_T[task][0] for task in q_C_T], 1)
            else:
                v = torch.stack(
                    [Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1).reshape(z.size(0), ns_G * ns_T, -1)
                     for task in q_C_T], 1)

            p_Y = self.decode(X_D, v)

            return p_Y





