import torch
from torch import nn
from src.model import MSAMambaClassificationConfig, MSAMambaForSequenceClassification
from data.dataset import MSAGenome, collate_binary
import wandb
from tqdm import tqdm
import pickle


WANDB_KEY = "b2e79ea06ca3e1963c1b930a9944bce6938bbb59"

class FinetuneConfig:
    def __init__(self, data_path: str, lr: float, task_name: str, n_epochs: int, batch_size: int, val_batch: int = 2, grad_accum_steps: int = 4, max_steps: int = 7500, weight_decay: float = 0.001) -> None:
        self.lr = lr
        self.data_path = data_path
        self.task_name = task_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.val_batch = val_batch
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
    
    def to_dict(self):
        return {
            "data_path": self.data_path,
            "task_name": self.task_name,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "val_batch": self.val_batch,
            "grad_accum_steps": self.grad_accum_steps,
            "max_steps": self.max_steps,
        }


def finetune(model_path: str, model_config: MSAMambaClassificationConfig, tune_config: FinetuneConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=WANDB_KEY)
    wandb.init(project="msa-finetune", config=tune_config.to_dict().update(model_config.to_dict()))

    print("setting up model...")
    model_dict = torch.load(model_path)
    model = MSAMambaForSequenceClassification(model_config)
    model.load_mamba(model_dict)

    model = model.to(device=device)

    wandb.watch(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model setup complete, param count: ", pytorch_total_params)

    dataset = MSAGenome(tune_config.data_path, tune_config.batch_size, tune_config.val_batch, collate_fn=collate_binary)

    losses = []
    accs = []

    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tune_config.lr, betas=(0.9, 0.99), weight_decay=tune_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, total_iters=dataset.train_steps//(tune_config.grad_accum_steps * 10))

    for epoch in range(tune_config.n_epochs):
        train_loader, val_loader = dataset.get_dataloaders()
        ival_loader = iter(val_loader)

        opt.zero_grad()
        for ix, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=max(tune_config.max_steps, dataset.train_steps)):
            x, target = data
            print(x.size(), target.size(), x.dtype)

            y = model(x)
            loss = criterion(y[:, 0], target)
            loss.backward()
            accuracy = torch.sum(target==torch.argmax(y, dim=-1))/tune_config.batch_size

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})
            losses.append(loss.item())
            accs.append(accuracy.item())

            if (ix+1)%tune_config.grad_accum_steps == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad()
            
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
                    accuracy = torch.sum(target==torch.argmax(y, dim=-1))/tune_config.batch_size

                    wandb.log({"val_loss": loss.item(), "val_accuracy": accuracy.item()})

    torch.save(model.state_dict(), f"model_{tune_config.task_name}.pt")
    wandb.save(f"model_{tune_config.task_name}.pt")
    with open(f"losses_{tune_config.task_name}.pkl", "wb") as f:
        pickle.dump([losses, accs], f)
    
    wandb.save(f"losses_{tune_config.task_name}.pkl")

    return model


if __name__ == "__main__":
    model_config = MSAMambaClassificationConfig(
        n_layers=3,
        d_model=118,
        vocab_size=6,
        n_heads=4,
        dropout_p=0,
    )

    finetune_config = FinetuneConfig(
        lr=9e-6, 
        data_path="data/msa_seq1k_30k_clinvar.pkl",
        task_name="finetune_test",
        n_epochs=4,
        batch_size=2,
        val_batch=1,
        grad_accum_steps=16
    )

    finetune("msamamba.pt", model_config, finetune_config)
