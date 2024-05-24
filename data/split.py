import torch
import pickle
import numpy as np
from tqdm import tqdm

SEQLEN = 1024

def process(data: str):
    print("loading data...")
    with open(data, "rb") as f:
        msas = pickle.load(f)

    proc = []

    for ix, msa in tqdm(enumerate(msas), desc="parsing"):
        x, target = msa
        x = np.asarray(x, dtype=np.uint8)
        proc.append(tuple([x[:, :512].tolist(), target[:512]]))
        proc.append(tuple([x[:, 512:].tolist(), target[512:]]))
        x = None
    
    print("Storing data...")
    with open("msa_seq1k_split_30k.pkl", "wb") as f:
        pickle.dump(proc, f)
    


if __name__ == "__main__":
    process("msa_seq1k_2k.pkl")