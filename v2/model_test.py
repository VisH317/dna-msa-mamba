import torch
from modules.msamambav2 import MSAMambaV2Block

model = MSAMambaV2Block(
    d_model=8,
    n_query=4,
    n_kv=2,
    n_channels=2,
    kernel_size=16,
    expand=2,
    d_conv=7,
    mlp_expand=4,
    act="silu",
    d_attn=4,
    norm_eps=1e-6,
    dropout_p=0.0
)

msa = torch.rand(2, 5, 32, 8)

out = model(msa)

print(out)