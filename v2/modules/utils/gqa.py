import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils.rmsnorm import RMSNorm
from einops import rearrange, einsum


def grouped_query_attention(Q: Tensor, K: Tensor, V: Tensor, dropout: float = 0.0, scale = None):
    if scale is None:
        scale = Q.size(-1) ** 0.5
    
    assert K.size(1) == V.size(1), "unequal KV head sizes"
    hq, hk = Q.size(1), K.size(1)
    assert hq % hk == 0, "improbable choice of kv heads, must be a factor of n_query_heads"
    n_head_groups = hq // hk
    
    Q /= scale
    
    Q = rearrange(Q, "b (h g) m d -> b g h m d", g=n_head_groups)
    similarity = einsum(Q, K, "b g h n d, b h s d -> b g h n s")
    attention = F.softmax(similarity, dim=-1)
    if dropout > 0:
        attention = F.dropout(attention, p=dropout)
        
    out = einsum(attention, V, "b g h n s, b h s d -> b g h n d")
    
    return rearrange(out, "b g h n d -> b (g h) n d")
