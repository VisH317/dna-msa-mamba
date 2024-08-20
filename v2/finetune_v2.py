import torch
from torch import nn
from modules.msamambav2 import MSAMambaV2Config, MSAMambaV2ClassificationConfig, MSAMambaV2ForSequenceClassification
from data.genome import MSAGenome, collate_binary
import wandb
from tqdm import tqdm
from dataclasses import dataclass, asdict
import pickle
import os

os.chdir("..")

WANDB_KEY = "b2e79ea06ca3e1963c1b930a9944bce6938bbb59"

@dataclass
class FinetuneConfig:
    lr: float
    data_path: str
    task_name: str
    n_epochs: int
    batch_size: int
    val_batch: int
    grad_accum_steps: int
    max_steps: int
    weight_decay: float
    
    

def finetune(model_path: str, model_config: MSAMambaV2ClassificationConfig, tune_config: FinetuneConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=WANDB_KEY)
    wandb.init(project="msa-finetune-v2", config=asdict(tune_config).update(asdict(model_config)))

    print("setting up model...")
    model_dict = torch.load(model_path)
    model = MSAMambaV2ForSequenceClassification(model_config)
    model.load_mamba(model_dict)

    model = model.to(device=device)

    wandb.watch(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model setup complete, param count: ", pytorch_total_params)

    dataset = MSAGenome(tune_config.data_path, tune_config.batch_size, tune_config.val_batch, collate_fn=collate_binary)

    losses = []
    val_losses = []
    accs = []

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tune_config.lr, betas=(0.9, 0.95), weight_decay=tune_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=dataset.train_steps//(tune_config.grad_accum_steps * 10))

    for epoch in range(tune_config.n_epochs):
        train_loader, val_loader = dataset.get_dataloaders()
        ival_loader = iter(val_loader)

        opt.zero_grad()
        for ix, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=min(tune_config.max_steps, dataset.train_steps)):
            x, target = data
            print(x.size(), target.size(), x.dtype)

            y = model(x)
            loss = criterion(y[:, 0], target)
            loss.backward()
            accuracy = torch.sum(target==torch.argmax(y, dim=-1))/tune_config.batch_size

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy.item(), "lr": opt.param_groups[0]["lr"]})
            losses.append(loss.item())
            wandb.log({"running train loss": sum(losses[max(0, len(losses)-tune_config.grad_accum_steps):])/len(losses[max(0, len(losses)-tune_config.grad_accum_steps):])})
            accs.append(accuracy.item())


            if (ix+1)%tune_config.grad_accum_steps == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad()
            
            if ix >= tune_config.max_steps: break
            
            # validation
            if ix % 32 == 0:
                with torch.no_grad():
                    try:
                        x, target = next(ival_loader)
                    except:
                        val_loader = dataset.get_val_dataloader()
                        ival_loader = iter(val_loader)
                        x, target = next(ival_loader)
                    
                    y = model(x)
                    loss = criterion(y[:, 0], target)
                    accuracy = torch.sum(target==torch.argmax(y, dim=-1))/tune_config.val_batch

                    val_losses.append(loss.item(0))
                    wandb.log({"val_loss": loss.item(), "val_accuracy": accuracy.item()})

    torch.save(model.state_dict(), f"model_{tune_config.task_name}.pt")
    wandb.save(f"model_{tune_config.task_name}.pt")
    with open(f"losses_{tune_config.task_name}.pkl", "wb") as f:
        pickle.dump([losses, val_losses, accs], f)
    
    wandb.save(f"losses_{tune_config.task_name}.pkl")

    return model


if __name__ == "__main__":
    
    model_config = MSAMambaV2Config(
        n_layers=4,
        vocab_size=6,
        d_model=64,
        n_query=6,
        n_kv=3,
        d_attn=32,
        norm_eps=1e-5,
        dropout_p=0,
        n_channels=2,
        kernel_size=128,
        expand=2,
        d_conv=4,
        mlp_expand=4,
        act="silu"
    )

    classification_config = MSAMambaV2ClassificationConfig(
        model_config=model_config,
        n_classes = 2,
        cls_dropout_p = 0.1
    )

    tune_config = FinetuneConfig(
        lr=3e-4,
        data_path="../data/msa_seq1k_30k_clinvar.pkl",
        task_name="clinvar",
        n_epochs=3,
        batch_size=2,
        val_batch=1,
        grad_accum_steps=16,
        weight_decay=0.001
    )


    finetune("msamamba.pt", classification_config, tune_config)
