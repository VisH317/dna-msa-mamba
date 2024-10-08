import torch
from torch import nn, optim
from data.genome import MSAGenome
from modules.msamambav2 import MSAMambaV2ForMLM, MSAMambaV2Config
from tqdm import tqdm
import pickle
import wandb
from dataclasses import asdict

# BRO PLEASE DONT TAKE THIS
WANDB_KEY = "b2e79ea06ca3e1963c1b930a9944bce6938bbb59"


class TrainConfig:
    def __init__(self, datapath: str, n_epochs: int, lr: float, batch_size: int, val_batch_size: int, val_step: int, weight_decay: float, grad_accum_iter: int, grad_clip: float | None = None) -> None:
        self.datapath = datapath
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.weight_decay = weight_decay
        self.grad_accum_iter = grad_accum_iter
        self.grad_clip = grad_clip
        self.val_step = val_step
    
    def to_dict(self):
        return {
            "seqlen": 1024,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "grad_accum_iter": self.grad_accum_iter,
            "grad_clip": self.grad_clip,
        }


def train(train_config: TrainConfig, model_config: MSAMambaV2Config, train_ckpt: str | None = None):

    wandb.login(key=WANDB_KEY)
    
    wandb.init(project="msamambav2", config=train_config.to_dict() | asdict(model_config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model & data...")
    model = MSAMambaV2ForMLM(model_config).to(device=device)
    if train_ckpt: model.load_state_dict(torch.load(train_ckpt))
    wandb.watch(model, log="all")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("param count: ", pytorch_total_params)

    MAX_EPOCH_LEN = 15000

    dataset = MSAGenome(train_config.datapath, train_config.batch_size, train_config.val_batch_size)
    
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    opt = optim.AdamW(model.parameters(), lr=train_config.lr, betas=(0.9, 0.99), weight_decay=train_config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    # cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(4*MAX_EPOCH_LEN)//train_config.grad_accum_iter, eta_min=0.15) # cosine screws stuff up badly
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.15, total_iters=MAX_EPOCH_LEN//train_config.grad_accum_iter)
    # no scheduler yet but will add later

    # data storage
    losses = []
    val_losses = []

    for epoch in range(train_config.n_epochs):
        train_loader, val_loader = dataset.get_dataloaders()
        val_loader_iter = iter(val_loader)

        torch.autograd.set_detect_anomaly(True)
        
        opt.zero_grad()
        for ix, data in (bar := tqdm(enumerate(train_loader), desc=f"Epoch: {epoch+1}, Loss: N/A, Val: N/A", total=min(MAX_EPOCH_LEN, len(dataset)//train_config.batch_size))):
            input, target = data
            y = model(input.to(dtype=torch.int64))[:, 0]
            
            loss = criterion(y.transpose(2, 1), target)
            loss.backward()
            losses.append(loss.item())
            wandb.log({ "train_loss": loss.item() })

            if (ix+1) % train_config.grad_accum_iter == 0:
                if train_config.grad_clip is not None: nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                opt.step()
                opt.zero_grad()
                warmup.step()
                # cosine.step()
            
            bar.set_description(f"Epoch: {epoch+1}, Loss: {losses[-1]}")

            if ix >= MAX_EPOCH_LEN: break

            if ix % train_config.val_step == 0:
                with torch.no_grad():
                    try:
                        input, target = next(val_loader_iter)
                    except:
                        val_loader = dataset.get_val_dataloader()
                        val_loader_iter = iter(val_loader)
                        input, target = next(val_loader_iter)

                    y = model(input.to(device=device))[:, 0]
                    loss = criterion(y.transpose(2, 1), target.to(device=device))
                    val_losses.append(loss.item())
                    wandb.log({"val_loss": loss.item()})

        scheduler.step()

    torch.save(model.state_dict(), "model.pt")
    with open("losses.pkl", "wb") as f:
        pickle.dump([losses, val_losses], f)
    
    wandb.save("model.pt")
    wandb.save("losses.pkl")
