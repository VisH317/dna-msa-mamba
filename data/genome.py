from Bio import SeqIO
from Bio.Seq import Seq
from gpn.data import GenomeMSA
from tqdm import tqdm
import csv
import pickle

msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"

msa = GenomeMSA(msa_path)

print("testing MSA")
X = msa.get_msa("1", 100000, 101024, strand="+", tokenize=True)
print("MSA retrieval: ", X)

MSAs = []

with open("genome.csv", "r", newline="") as f:
    reader = csv.reader(f)
    for ix, row in tqdm(enumerate(reader), desc="getting MSAs", total=1280*24):
        if ix == 0: continue
        m = msa.get_msa(row[0], int(row[1]), int(row[2]), strand="+", tokenize=True)
        MSAs.append(m)
        if (ix+1) % 1000 == 0:
            with open(f"msa_seq1k_24k_{ix}_ckpt.pkl", "wb") as f:
                pickle.dump(MSAs, f)

with open("msa_seq1k_24k.pkl", "wb") as f:
    pickle.dump(MSAs, f)
