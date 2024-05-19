import torch
import pickle
from torch import Tensor
from tqdm import tqdm

SEQLEN = 1024

# t: M x L
def mask(t: Tensor):
    rand = torch.rand(SEQLEN)

    x0 = torch.where(rand <= 0.15, torch.full([SEQLEN], 5), t[0])
    x0 = torch.where(rand <= 0.015, torch.randint(low=1, high=5, size=[SEQLEN]), x0)
    x0 = torch.where(torch.logical_and(rand > 0.015, rand <= 0.03), t[0], x0)

    x = torch.concat([x0.unsqueeze(0), t[1:]])
    target = torch.where(rand <= 0.15, t[0], torch.full([SEQLEN], -100))

    return (x.detach().tolist(), target.detach().tolist())

torch.set_printoptions(profile="full")

def process(data: str):
    with open(data, "rb") as f:
        msas = pickle.load(f)
    
    proc = []

    for ix, msa in tqdm(enumerate(msas), desc="parsing"):
        msa_t = torch.as_tensor(msa).t()
        d = mask(msa_t)
        proc.append(d)
    
    print("Storing data...")
    with open("msa_seq1k_30k.pkl", "wb") as f:
        pickle.dump(proc, f)
    


if __name__ == "__main__":
    process("msa_seq1k_24k_raw.pkl")