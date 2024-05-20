import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.utils import RMSNorm
import numpy as np

class ColumnAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_attn: int = -1, dropout_p: float = 0) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_attn = d_model//2 if d_attn == -1 else d_attn
        self.dropout_p = dropout_p
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, 3 * self.d_attn * n_heads, bias=False)
        self.o = nn.Linear(self.d_attn * self.n_heads, d_model)

        nn.init.normal_(self.qkv.weight, mean=0, std=np.sqrt(2 / (self.d_model + self.d_attn)))
    
    # x: B x M x L x D
    def forward(self, x: Tensor) -> Tensor:
        b, m, l, d = x.size()
        qkv = self.qkv(x).reshape(b, l, self.n_heads, m, self.d_attn * 3) # check if features being moved properly here
        # print(qkv.size())
        Q, K, V = qkv.split(self.d_attn, dim=-1)
        # mask = mask.transpose(-2, -1).unsqueeze(-2).repeat(1, 1, self.n_heads, 1).unsqueeze(-1)
        # print(mask.size())

        att = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout_p)
        att = att.view(b, m, l, self.d_attn * self.n_heads)
        return self.o(att)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mem: int = -1, act = "silu") -> None:
        super().__init__()

        self.d_model = d_model
        self.d_mem = d_model * 4 if d_mem == -1 else d_mem

        assert act in ["silu", "relu", "gelu"], "MLP activation not valid"

        self.act = nn.SiLU() if act == "silu" else nn.ReLU() if act == "relu" else nn.GELU()

        self.lin1 = nn.Linear(self.d_model, self.d_mem)
        self.lin2 = nn.Linear(self.d_mem, self.d_model)


    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model: int, norm_type="rmsnorm") -> None:
        super().__init__()
        
        assert norm_type in ["rmsnorm", "layernorm"], "AddNorm: norm type is not valid"
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        return self.norm(x + res)





