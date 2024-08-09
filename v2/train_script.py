from train import TrainConfig, train
from modules.msamambav2 import MSAMambaV2Config

model_config = MSAMambaV2Config(
    n_layers=1,
    vocab_size=6,
    d_model=8,
    n_query=4,
    n_kv=2,
    d_attn=4,
    norm_eps=1e-5,
    dropout_p=0,
    n_channels=2,
    kernel_size=64,
    expand=2,
    d_conv=7,
    mlp_expand=4,
    act="silu"
)

train_config = TrainConfig(
    datapath="../data/msa_seq1k_2k.pkl",
    n_epochs=2,
    lr=3e-4,
    batch_size=4,
    val_batch_size=2,
    val_step=8,
    weight_decay=1e-3,
    grad_accum_iter=8,
    grad_clip=5.0
)

if __name__ == "__main__":
    model = train(train_config, model_config)
    
    