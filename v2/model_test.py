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
).cuda()

msa = torch.rand(2, 5, 32, 8).cuda()

out = model(msa)

print(out)

# from modules.hydra.modules.hydra import Hydra

# hydra = Hydra(8, d_conv=5, headdim=8, expand=2)

# hydra(torch.rand(2, 32, 8))