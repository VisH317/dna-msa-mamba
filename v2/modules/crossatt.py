import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils.rmsnorm import RMSNorm

class MSACrossAttention(nn.Module):
    def __init__(self, d_model: int, n_out_channels: int, n_query_heads: int, d_attn: int | None = None, norm_eps: float = 1e-6, dropout_p: float = 0.0) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.n_out_channels = n_out_channels
        self.n_query_heads = n_query_heads
        self.d_attn = d_attn
        self.norm_eps = norm_eps
        self.dropout_p = dropout_p
        
        self.w_kv = nn.Linear(d_model, 2 * n_out_channels * d_attn)
        self.w_q = nn.Linear(d_model, n_query_heads * d_attn)
        self.w_o = nn.Linear(d_attn, d_model)
        
        self.norm = RMSNorm(d_model, eps=norm_eps)

    # X: B x M x S x D
    def forward(self, x: Tensor) -> Tensor:
        b, m, s, d = x.size()
        x = x.view(b, s, m, d)
        
        Q = self.w_q(x).reshape(b, s, self.n_query_heads, m, self.d_attn)
        kv = self.w_kv(x[:, :, 0, :]).reshape(b, s, self.n_out_channels, 1, 2 * self.d_attn)
        K, V = kv.split([self.d_attn, self.d_attn])
        
        o = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p)
        O = self.w_o(o).squeeze().reshape(b, self.n_out_channels, s, d) # B x S x C x D
        return self.norm(O + x[:, :, 0, :])
        
        
        
        
        