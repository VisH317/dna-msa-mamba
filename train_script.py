from src.model import MSAMambaConfig
from train import TrainConfig, train

model_config = MSAMambaConfig(
    n_layers=4,
    d_model=128,
    vocab_size=6,
    n_heads=6,
    dropout_p=0,
)

train_config = TrainConfig(
    datapath="data/msa_seq1k_2k.pkl",
    n_epochs=4,
    lr=3e-4,
    batch_size=8,
    val_batch_size=1,
    val_step=32,
    weight_decay=0.08,
    grad_accum_iter=8,
    grad_clip=1.0
)

if __name__ == "__main__":
    train(train_config, model_config)
