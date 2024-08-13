import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.hydra.modules.hydra import Hydra
from fft_conv_pytorch import FFTConv1d
from modules.utils.rmsnorm import RMSNorm

class ConvHydra(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kernel_size: int = 128, expand: int = 2, d_conv: int = 7, act: str = "silu") -> None:
        super().__init__()
        
        assert act in ["silu", "relu", "gelu"], "activation function invalid"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.expand = expand
        self.d_conv = d_conv
        self.kernel_size = kernel_size
        self.act = act
        
        self.act1 = nn.SiLU() if act == "silu" else nn.ReLU() if act == "relu" else nn.GELU()
        self.act2 = nn.SiLU() if act == "silu" else nn.ReLU() if act == "relu" else nn.GELU()
        
        assert (self.expand * self.d_model) % self.n_heads == 0, "heads do not work"
        self.headdim = (self.expand * self.d_model) // self.n_heads
        
        self.in_conv = FFTConv1d(d_model, d_model * n_heads, kernel_size=self.kernel_size, padding="same") # TODO: test this value
        self.out_conv = FFTConv1d(d_model * n_heads, d_model, kernel_size=self.kernel_size, padding="same")
        
        self.hydra = Hydra(d_model, d_conv=d_conv, headdim=self.headdim, expand=expand)
        
    # x: B x S x D
    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.size()
        
        x_proj = self.in_conv(x.view(b, d, s))
        x_proj = self.act1(x_proj.view(b * self.n_heads, s, d))
        print(x_proj.size())
        x_out = self.hydra(x_proj)
        print(x_out.size())
        x_out = self.act2(self.out_conv(x_out.view(b, d * self.n_heads, s)).view(b, s, d))
        
        return x_out
        
