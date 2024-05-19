import csv
import torch
from tqdm import tqdm, trange

CHROM_LEN = {
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    "X": 156040895,
    "Y": 57227415
}

TOTAL_PER_CHROM = 1280

VOCAB_SIZE = 6

if __name__ == "__main__":
    with open("genome.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chrom", "start", "end"])
        for c,l in tqdm(CHROM_LEN.items(), desc="chrom"):
            rand = torch.randint(0, l-1024, [TOTAL_PER_CHROM])
            for i in trange(TOTAL_PER_CHROM):
                writer.writerow([c, rand[i].item(), rand[i].item() + 1024])

    print("done")