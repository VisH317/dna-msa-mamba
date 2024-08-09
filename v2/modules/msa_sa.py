import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils.rmsnorm import RMSNorm
from modules.utils.gqa import grouped_query_attention

# sparse version that only maps main sequence to auxiliary (saves memory!)
class MSASelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int, d_attn: int | None = None, dropout_p: float = 0.0, norm_eps: float = 1e-5) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.d_attn = d_attn if d_attn is not None else d_model // 2
        self.dropout_p = dropout_p
        
        self.w_kv = nn.Linear(d_model, 2 * self.n_kv * d_attn)
        self.w_q = nn.Linear(d_model, self.n_heads * d_attn)
        self.w_o = nn.Linear(d_attn * self.n_heads, d_model)
        
        self.norm = RMSNorm(d_model, eps=norm_eps)

    # X: B x M x S x D
    def forward(self, x: Tensor) -> Tensor:
        b, m, s, d = x.size()
        x = x.view(b, s, m, d)
        
        Q = self.w_q(x).reshape(b * s, self.n_heads, m, self.d_attn)
        kv = self.w_kv(x[:, :, 0, :]).reshape(b * s, self.n_kv, 1, 2 * self.d_attn)
        K, V = kv.split([self.d_attn, self.d_attn], dim=-1)
        
        o = grouped_query_attention(Q, K, V, dropout=self.dropout_p)
        O = self.w_o(o.view(b, m, s, d * self.n_kv).squeeze()) # B x C x S x D
        return self.norm(O + x.view(b, m, s, d))
        
