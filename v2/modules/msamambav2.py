import torch
from torch import nn, Tensor

from modules.conv_hydra import ConvHydra
from modules.crossatt import MSACrossAttention
from modules.msa_sa import MSASelfAttention
from modules.utils.mlp import MLP
from modules.utils.rmsnorm import RMSNorm
from dataclasses import dataclass


class MSAMambaV2Block(nn.Module):
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
        
    # x: B x M x S x D
    def forward(self, x: Tensor) -> Tensor:
        x_main = self.msa_ca(x)
        x_out_main = self.norm1(self.convhydra(x_main) + x_main)
        
        x_out = torch.concat([x_out_main, x[:, 1:, :, :]], dim=1)
        x_out = self.msa_sa(x_out)
        
        return self.norm2(self.mlp(x_out) + x_out)

@dataclass
class MSAMambaV2Config:
    n_layers: int
    vocab_size: int
    d_model: int
        
    # attention hyperparams
    n_query: int
    n_kv: int
    d_attn: int
    norm_eps: float
    dropout_p: float
    
    # convhydra params
    n_channels: int
    kernel_size: int
    expand: int
    d_conv: int
    
    # mlp params
    mlp_expand: int
    act: str
    
    

class MSAMambaV2(nn.Module):
    def __init__(self, n_layers: int, vocab_size: int, d_model: int, n_query: int, n_kv: int, n_channels: int, kernel_size: int, expand: int, 
                 d_conv: int, mlp_expand: int = 4, act: str = "silu", d_attn: int | None = None, norm_eps: float = 1e-6, 
                 dropout_p: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
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
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            MSAMambaV2Block(d_model, n_query, n_kv, n_channels, kernel_size, expand, d_conv, mlp_expand, act, d_attn, norm_eps, dropout_p)
            for _ in range(self.n_layers)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for block in self.blocks: x = block(x)
        
        return x
    
    @staticmethod
    def from_config(config: MSAMambaV2Config):
        return MSAMambaV2(
            config.n_layers, 
            config.vocab_size, 
            config.d_model, 
            config.n_query, 
            config.n_kv, 
            config.n_channels, 
            config.kernel_size, 
            config.expand, 
            config.d_conv, 
            config.mlp_expand, 
            config.act, 
            config.d_attn, 
            config.norm_eps, 
            config.dropout_p
        )
        

class MSAMambaV2ForMLM(nn.Module):
    def __init__(self, config: MSAMambaV2Config) -> None:
        super().__init__()
        
        self.model = MSAMambaV2.from_config(config)
        self.lmhead = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.lmhead(self.model(x))

