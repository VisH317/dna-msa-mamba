import torch
from torch import nn, Tensor
from src.blocks import ColumnAttention, AddNorm, MLP
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights


class MSAMambaBlock(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int = 4, d_attn: int = -1, dropout_p: float = 0, d_mem: int = -1, act="silu", norm="rmsnorm") -> None:
        super().__init__()

        self.mamba = create_block(d_model)
        self.att = ColumnAttention(d_model, n_heads, d_attn, dropout_p)
        self.transition = MLP(d_model, d_mem, act)

        # res connections
        self.norm1 = AddNorm(d_model, norm)
        self.norm2 = AddNorm(d_model, norm)
        self.norm3 = AddNorm(d_model, norm)

        # init weights
        _init_weights(self.mamba, n_layers)


    def forward(self, x: Tensor) -> Tensor:
        
        b, m, l, d = x.size()

        x_mamba = x.flatten(0, 1)
        out_mamba = self.mamba(x_mamba)[0].reshape(b, m, l, d)
        out = self.norm1(out_mamba, x)
        out = self.norm2(self.att(out), out)
        out = self.norm3(self.transition(out), out)

        return out


class MSAMamba(nn.Module):
    def __init__(self, n_layers: int, d_model: int, vocab_size: int, n_heads: int = 4, d_attn: int = -1, dropout_p: float = 0, d_mem: int = -1, act="silu", norm="rmsnorm") -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MSAMambaBlock(n_layers, d_model, n_heads, d_attn, dropout_p, d_mem, act, norm) for i in range(n_layers)])
    
    def forward(self, x: Tensor) -> Tensor:
        # print("masking...")
        # mask = x != 5
        # print("masked")
        y = self.embed(x)
        for layer in self.layers: y = layer(y)
        return y


class MSAMambaConfig:
    def __init__(self, n_layers: int, d_model: int, vocab_size: int, n_heads: int = 4, d_attn: int = -1, dropout_p: float = 0, d_mem: int = -1, act="silu", norm="rmsnorm") -> None:
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.dropout_p = dropout_p
        self.d_mem = d_mem
        self.act = act
        self.norm = norm
    
    def to_dict(self):
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout_p": self.dropout_p,
            "d_attn": self.d_attn,
            "d_mem": self.d_mem,
            "act": self.act,
            "norm": self.norm,
        }


class MSAMambaClassificationConfig:
    def __init__(self, n_layers: int, d_model: int, vocab_size: int, n_heads: int = 4, d_attn: int = -1, dropout_p: float = 0, d_mem: int = -1, act="silu", norm="rmsnorm", n_classes: int = 2, class_dropout_p: float = 0.2, pooler_select = "last") -> None:
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.dropout_p = dropout_p
        self.d_mem = d_mem
        self.act = act
        self.norm = norm
        self.n_classes = n_classes
        self.class_dropout_p = class_dropout_p
        self.pooler_select = pooler_select
    
    def to_dict(self):
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "dropout_p": self.dropout_p,
            "d_attn": self.d_attn,
            "d_mem": self.d_mem,
            "act": self.act,
            "norm": self.norm,
            "n_classes": self.n_classes,
            "class_dropout_p": self.class_dropout_p,
            "pooler_select": self.pooler_select,
        }
        


class MSAMambaForMLM(nn.Module):
    def __init__(self, config: MSAMambaConfig) -> None:
        super().__init__()

        self.mamba = MSAMamba(config.n_layers, config.d_model, config.vocab_size, config.n_heads, config.d_attn, config.dropout_p, config.d_mem, config.act, config.norm)
        self.lmhead = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.lmhead(self.mamba(x))

class MSAMambaForSequenceClassification(nn.Module):
    def __init__(self, config: MSAMambaClassificationConfig):
        super().__init__()

        self.config = config
        self.mamba = MSAMamba(config.n_layers, config.d_model, config.vocab_size, config.n_heads, config.d_attn, config.dropout_p, config.d_mem, config.act, config.norm)
        self.pooler = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh()
        )

        assert config.pooler_select in ["last", "first"], "pooler selection is invalid"
        self.pooler_select = config.pooler_select

        self.dropout = nn.Dropout(config.class_dropout_p)
        self.classifier = nn.Linear(config.d_model, config.n_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.mamba(x) # b m l d
        out = out[:, 0, 0 if self.pooler_select == "first" else -1]
        return self.classifier(self.dropout(self.pooler(out)))
    
    def load_mamba(self, mamba_dict):
        mambamlm = MSAMambaForMLM(self.config)
        mambamlm.load_state_dict(mamba_dict)
        self.mamba = mambamlm.mamba
        del mambamlm

