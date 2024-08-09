import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils.rmsnorm import RMSNorm
from modules.utils.gqa import grouped_query_attention
from einops import rearrange, einsum    
from rotary_embedding_torch import RotaryEmbedding


class MSACrossAttention(nn.Module):
    def __init__(self, d_model: int, n_out_channels: int, n_query_heads: int, d_attn: int | None = None, norm_eps: float = 1e-6, dropout_p: float = 0.0) -> None:
        super().__init__()
        
        assert n_query_heads % n_out_channels == 0, "query heads must be divisible into valid group sizes"
        
        self.d_model = d_model
        self.n_out_channels = n_out_channels
        self.n_query_heads = n_query_heads
        self.d_attn = d_attn
        self.norm_eps = norm_eps
        self.dropout_p = dropout_p
        
        self.w_kv = nn.Linear(d_model, 2 * n_out_channels * d_attn)
        self.w_q = nn.Linear(d_model, n_query_heads * d_attn)
        self.w_o = nn.Linear(d_attn * n_query_heads, d_model)
        
        self.norm = RMSNorm(d_model, eps=norm_eps)
        
        self.rotary = RotaryEmbedding(d_model)

    # X: B x M x S x D
    def forward(self, x: Tensor) -> Tensor:
        x = x.clone()
        b, m, s, d = x.size()
        x1 = x.reshape(b, s, m, d).contiguous()
        q = x[:, 0, :, :].contiguous()
        
        kv = self.w_kv(x1)
        Q = self.w_q(q)
        Q = Q.reshape(b * s, self.n_query_heads, 1, self.d_attn)
        kv = kv.reshape(b * s, self.n_out_channels, m, 2 * self.d_attn)
        K, V = kv.split([self.d_attn, self.d_attn], dim=-1)
        
        Q = self.rotary.rotate_queries_or_keys(Q)
        K = self.rotary.rotate_queries_or_keys(K)
        
        o = grouped_query_attention(Q, K, V, dropout=self.dropout_p)
        O = self.w_o(o.view(b, s, d * self.n_out_channels).squeeze()) # B x C x S x D
        return self.norm(O + q)
        
