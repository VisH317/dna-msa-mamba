from Bio import SeqIO
from Bio.Seq import Seq
from gpn.data import GenomeMSA
from tqdm import tqdm
import csv
import pickle

def get_stuff():
    msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"

    msa = GenomeMSA(msa_path)

    print("testing MSA")
    X = msa.get_msa("1", 100000, 101024, strand="+", tokenize=True)
    print("MSA retrieval: ", X)

    MSAs = []

    with open("data/genome_512.csv", "r", newline="") as f:
        reader = csv.reader(f)
        for ix, row in tqdm(enumerate(reader), desc="getting MSAs", total=1500*24):
            if ix == 0: continue
            m = msa.get_msa(row[0], int(row[1]), int(row[2]), strand="+", tokenize=True)
            MSAs.append(m)
            if (ix+1) % 10000 == 0:
                with open(f"msa_seq1k_24k_{ix}_ckpt.pkl", "wb") as f:
                    pickle.dump(MSAs, f)

    with open("msa_seq512_30k.pkl", "wb") as f:
        pickle.dump(MSAs, f)


def get_data(csv_file: str):
    msa_path = "zip:///::https://huggingface.co/datasets/songlab/multiz100way/resolve/main/89.zarr.zip"
    msa = GenomeMSA(msa_path)
    print("testing MSA")
    X = msa.get_msa("1", 100000, 101024, strand="+", tokenize=True)
    print("MSA retrieval: ", X)
    
    MSAs = []

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        for ix, row in tqdm(enumerate(reader), desc="getting MSAs", total=1280*24):
            if ix == 0: continue
            m = msa.get_msa(row[0], int(row[1]), int(row[2]), strand="+", tokenize=True)
            MSAs.append((m, int(row[3])))

            # checkpoint
            if (ix+1) % 5000 == 0:
                with open(f"msa_seq1k_24k_{ix}_ckpt.pkl", "wb") as f:
                    pickle.dump(MSAs, f)

    with open(f"{csv_file}.pkl", "wb") as f:
        pickle.dump(MSAs, f)