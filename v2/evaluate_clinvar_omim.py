import torch
from torch import nn
from modules.msamambav2 import MSAMambaV2Config, MSAMambaV2ClassificationConfig, MSAMambaV2ForSequenceClassification
from data.genome import MSAGenome, collate_binary
import wandb
from tqdm import tqdm
from dataclasses import dataclass, asdict
import pickle
from torchmetrics import AUROC
import os

WANDB_KEY = "b2e79ea06ca3e1963c1b930a9944bce6938bbb59"    

def finetune(model_path: str, model_config: MSAMambaV2ClassificationConfig, data_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("setting up model...")
    model_dict = torch.load(model_path)
    model = MSAMambaV2ForSequenceClassification(model_config)
    model.load_mamba_from_mlm(model_dict)

    model = model.to(device=device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model setup complete, param count: ", pytorch_total_params)
    
    BATCH = 1

    dataset = MSAGenome(data_path, batch_size=2, val_batch_size=BATCH, train_size=0.9)
    
    y_target = torch.empty(len(dataset.val_data))
    
    y_pred = torch.empty(len(dataset.val_data))
        
    auroc = AUROC("binary")

    with torch.no_grad():
        for ix, data in tqdm(enumerate(dataset.val_data), desc=f"Evaluation", total=min(len(dataset)//BATCH)):
            x, target = data
            
            target_t = torch.zeros(x.size()[0], 2)
            for ix, item in enumerate(target): target_t[ix, 0 if target[ix]==0 else 1] = item

            y = model(x.to(device))
            
            y_pred[ix] = y[0]
            y_target[ix] = target
        
    return auroc(y_pred, y_target)


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
        max_steps=15000,
        weight_decay=0.001
    )

    os.chdir("v2")
    finetune("model.pt", classification_config, tune_config)
