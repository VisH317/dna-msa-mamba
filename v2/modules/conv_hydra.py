import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.hydra.modules.hydra import Hydra
from fft_conv_pytorch import FFTConv1d
from modules.utils.rmsnorm import RMSNorm
from mamba_ssm import Mamba

class ConvHydra(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kernel_size: int = 128, expand: int = 2, d_conv: int = 7, act: str = "silu", norm_eps: float = 1e-8) -> None:
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
        
        # self.hydra = Hydra(d_model, d_conv=d_conv, headdim=self.headdim, expand=expand) # NOTE: Mamba2 fused kernel is buggy on T4, so switched to original mamba impl
        self.mamba_forward = Mamba(d_model, d_state=128, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model, d_state=128, d_conv=d_conv, expand=expand)
        
        self.norm = RMSNorm(d_model, eps=norm_eps)
        
    # x: B x S x D
    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.size()
        
        x_proj = self.in_conv(x.view(b, d, s))
        x_proj = self.act1(x_proj.view(b * self.n_heads, s, d))

        x_out_forward = self.mamba_forward(x_proj)
        x_out_backward = self.mamba_forward(x_proj.flip(-2)).flip(-2)
        x_out = self.norm(x_out_forward + x_out_backward)

        x_out = self.act2(self.out_conv(x_out.view(b, d * self.n_heads, s)).view(b, s, d))
        
        return x_out
        
