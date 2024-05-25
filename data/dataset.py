import torch
from torch import Tensor
import pickle
from torch.utils.data import Dataset, DataLoader, random_split

class MSAGenomeCore(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        with open(path, "rb") as f:
            self.msa = pickle.load(f)
        
    def __len__(self):
        return len(self.msa)
    
    def __getitem__(self, idx: int):
        return self.msa[idx]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(data: list[tuple[list, list]]) -> tuple[Tensor, Tensor]:
    x = torch.stack([torch.as_tensor(d[0], device=device) for d in data], dim=0)
    target = torch.stack([torch.as_tensor(d[1], device=device) for d in data], dim=0)
    return x, target

def collate_binary(data: list[tuple[list, int]]) -> tuple[Tensor, Tensor]:
    x = torch.stack([torch.as_tensor(d[0], device=device) for d in data], dim=0)
    target = torch.as_tensor([d[0] for d in data], device=device, dtype=torch.float)
    return x, target


class MSAGenome:
    def __init__(self, path: str, batch_size: int, val_batch_size: int, train_size: float = 0.75, collate_fn=collate) -> None:
        data = MSAGenomeCore(path)

        self.length = len(data)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.collate_fn = collate

        self.train_data, self.val_data = random_split(data, [train_size, 1-train_size])

        self.train_steps = len(self.train_data) // batch_size
        self.val_steps = len(self.val_data) // val_batch_size

    def __len__(self):
        return self.length

    def get_train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def get_val_dataloader(self):
        return DataLoader(self.val_data, self.val_batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def get_dataloaders(self):
        return self.get_train_dataloader(), self.get_val_dataloader()

