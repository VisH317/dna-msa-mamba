import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, d_model: int, n_expand: int = 4, act: str = "silu") -> None:
        super().__init__()
        
        assert act in ["silu", "relu", "gelu"], "activation function invalid"
        
        self.d_model = d_model
        self.n_expand = n_expand
        self.act_name = act
        
        self.ln1 = nn.Linear(d_model, d_model * n_expand)
        self.act = nn.SiLU() if act == "silu" else nn.ReLU() if act == "relu" else nn.GELU()
        self.ln2 = nn.Linear(d_model * n_expand, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.ln2(self.act(self.ln1(x)))