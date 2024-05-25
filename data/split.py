import torch
import pickle
# import numpy as np
from tqdm import tqdm

SEQLEN = 1024

def process(data: str):
    print("loading data...")
    with open(data, "rb") as f:
        msas = pickle.load(f)

    ogl = len(msas)
    print(ogl)

    for ix in tqdm(range(ogl), desc="parsing"):
        x, target = msas[ix]

        msas.append(([i[512:] for i in x], target))
        msas[ix] = ([i[:512] for i in x], target)

        # proc.append([[i[:512] for i in x], target[:512]])
        # proc.append([[i[512:] for i in x], target[512:]])
        # x = None
    
    print("Storing data...")
    with open("msa_seq1k_split_30k_clinvar.pkl", "wb") as f:
        pickle.dump(msas, f)
    


if __name__ == "__main__":
    process("msa_seq1k_2k.pkl")