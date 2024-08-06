import torch
from torch import nn, Tensor

from modules.conv_hydra import ConvHydra
from modules.crossatt import MSACrossAttention
from modules.msa_sa import MSASelfAttention
from modules.utils.mlp import MLP
from modules.utils.rmsnorm import RMSNorm


class MSAMambaV2(nn.Module):
    def __init__(self, d_model: int, n_query: int, n_kv: int, n_channels: int, kernel_size: int, expand: int, 
                 d_conv: int, mlp_expand: int = 4, act: str = "silu", d_attn: int | None = None, norm_eps: float = 1e-6, 
                 dropout_p: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        
        # attention hyperparams
        self.n_query = n_query
        self.n_kv = n_kv
        self.d_attn = d_attn
        self.norm_eps = norm_eps
        self.dropout_p = dropout_p
        
        # convhydra params
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.expand = expand
        self.d_conv = d_conv
        
        # mlp params
        self.mlp_expand = mlp_expand
        self.act = act
        
        self.msa_ca = MSACrossAttention(d_model, n_kv, n_query, d_attn, norm_eps, dropout_p)
        self.msa_sa = MSASelfAttention(d_model, n_query, n_kv, d_attn, dropout_p, norm_eps)
        self.convhydra = ConvHydra(d_model, n_channels, kernel_size, expand, d_conv, act)
        self.mlp = MLP(d_model, mlp_expand, act)
        
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.norm3 = RMSNorm(d_model, eps=norm_eps)
        
    # x: B x M x S x D
    def forward(self, x: Tensor) -> Tensor:
        x_main = self.msa_ca(x)
        x_out_main = self.norm1(self.convhydra(x_main) + x_main)
        
        x_out = torch.concat([x_out_main, x[:, 1:, :, :]], dim=1)
        x_out = self.norm2(self.msa_sa(x_out) + x_out)
        
        return self.norm3(self.mlp(x_out) + x_out)

