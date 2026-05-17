import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FeatureSelector', 'SetEncoder', 'GlobalEncoder', 'TaskEncoder', 'TaskDecoder']


class AttnPool(nn.Module):
    def __init__(self, dim, num_heads=None, num_seeds=1, act_fn='relu', ln=False, dr=0.1):
        super().__init__()
        self.dim = dim
        self.num_seeds = num_seeds

        self.attn_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        # ==================================================

    def forward(self, X, mask=None):
        B, N, D = X.shape

        attn_weights = self.attn_net(X)

        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1).float()

        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        pooled = torch.bmm(attn_weights.transpose(1, 2), X)

        return pooled


class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, act_fn, ln):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if ln else nn.Identity(),
            act_fn(),
        )

    def forward(self, x):
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, act_fn='relu', ln=False):
        super().__init__()
        assert n_layers >= 1
        act_fn = nn.GELU if act_fn == 'gelu' else nn.ReLU

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        layers = []
        for l_idx in range(n_layers):
            di = dim_in if l_idx == 0 else dim_hidden
            do = dim_out if l_idx == n_layers - 1 else dim_hidden
            layers.append(FFB(di, do, act_fn, ln))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x


class LatentMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers=2, act_fn='relu', ln=False,
                 epsilon=0.1, sigma=True, sigma_act=torch.sigmoid):
        super().__init__()

        self.epsilon = epsilon
        self.sigma = sigma

        assert n_layers >= 1
        if n_layers >= 2:
            self.mlp = MLP(dim_in, dim_hidden, dim_hidden, n_layers - 1, act_fn, ln)
        else:
            self.mlp = None

        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        if self.sigma:
            self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
            self.sigma_act = sigma_act

    def forward(self, x):
        hidden = self.mlp(x) if self.mlp is not None else x

        mu = self.hidden_to_mu(hidden)
        if self.sigma:
            log_sigma = self.hidden_to_log_sigma(hidden)
            sigma = self.epsilon + (1 - self.epsilon) * self.sigma_act(log_sigma)

            return mu, sigma
        else:
            return mu


class FeatureSelector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.selector = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # 调整维度顺序
        attn = self.selector(self.avg_pool(x))
        x = x * attn
        x = x.transpose(1, 2)  # 恢复原来的维度顺序
        return x
        # return x * self.selector(self.avg_pool(x))


class SetEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers, attn_config):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, act_fn=attn_config.act_fn, ln=attn_config.ln)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,  # 特征维度
            nhead=attn_config.n_heads,  # 注意力头数
            dim_feedforward=dim_hidden * 4,  # Transformer 前馈层维度
            dropout=attn_config.dr,  # dropout
            activation=attn_config.act_fn,  # 激活函数
            batch_first=True,  # 输入形状 [B, seq_len, dim]
            norm_first=attn_config.ln  # Pre-LN / Post-LN
        )

        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_layers
        )
        self.pool = AttnPool(dim_hidden, attn_config.n_heads, 1, act_fn=attn_config.act_fn, ln=attn_config.ln,
                        dr=attn_config.dr)

    def forward(self, C):

        # project (x, y) to w
        w = self.mlp(C)  # (B, n, h)

        # intra-task attention
        w = self.attention(w)  # (B, n, h)

        # intra-task aggregation
        w = self.pool(w).squeeze(1)  # (B, h)

        return w


class GlobalEncoder(nn.Module):
    def __init__(self, dim_hidden, attn_layers, attn_config):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,  # 特征维度
            nhead=attn_config.n_heads,  # 注意力头数
            dim_feedforward=dim_hidden * 4,  # 通用前馈维度
            dropout=attn_config.dr,  # dropout
            activation=attn_config.act_fn,  # 激活函数
            batch_first=True,  # 输入形状 [B, seq_len, dim]
            norm_first=attn_config.ln  # 归一化位置（Pre-LN / Post-LN）
        )
        # 堆叠多层 Transformer 编码器
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attn_layers
        )
        self.pool = AttnPool(dim_hidden, attn_config.n_heads, 1, act_fn=attn_config.act_fn, ln=attn_config.ln,
                        dr=attn_config.dr)

        self.global_amortizer = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2, attn_config.act_fn, attn_config.ln)

    def forward(self, w):
        # inter-task attention
        w = self.attention(w)  # (B, T, h)

        # inter-task aggregation
        w = self.pool(w).squeeze(1)  # (B, h)

        # global latent distribution
        q_G = self.global_amortizer(w)

        return q_G


class TaskEncoder(nn.Module):
    def __init__(self, dim_hidden, attn_config, hierarchical=True):
        super().__init__()
        self.hierarchical = hierarchical
        self.task_amortizer = LatentMLP(dim_hidden * (1 + int(hierarchical)), dim_hidden, dim_hidden,
                                        2, attn_config.act_fn, attn_config.ln)

    def forward(self, w, z=None):
        # hierarchical conditioning
        if self.hierarchical:
            assert z is not None
            w = torch.cat((w, z), -1)

        # task latent distribution
        q_T = self.task_amortizer(w)

        return q_T


class TaskDecoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, n_layers, attn_config, sigma):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.input_projection = nn.Linear(dim_x, dim_hidden)

        self.n_heads = attn_config.n_heads
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=self.n_heads,
            batch_first=True,  # 输入形状: (B, seq_len, dim)
            dropout=0.1
        )

        self.norm_x = nn.LayerNorm(dim_hidden)
        self.norm_v = nn.LayerNorm(dim_hidden)

        self.output_amortizer = LatentMLP(dim_hidden, dim_y, dim_hidden, n_layers,
                                          attn_config.act_fn, attn_config.ln, sigma=sigma, sigma_act=F.softplus)

    def forward(self, X, v):

        x = self.input_projection(X)  # (B, n, h) or (B, ns, n, h)

        v = v.unsqueeze(-2).repeat(*([1] * (len(x.size()) - 2)), x.size(-2), 1)

        if x.dim() == 4:
            B, ns, n, h = x.shape
            x = x.reshape(B * ns, n, h)
            v = v.reshape(B * ns, n, h)

        x_norm = self.norm_x(x)  # Q
        v_norm = self.norm_v(v)  # K, V

        attn_output, _ = self.cross_attn(
            query=x_norm,
            key=v_norm,
            value=v_norm
        )

        if X.dim() == 4:
            attn_output = attn_output.reshape(B, ns, n, self.dim_hidden)

        decoder_input = attn_output

        # output distribution
        p_Y = self.output_amortizer(decoder_input)

        return p_Y
