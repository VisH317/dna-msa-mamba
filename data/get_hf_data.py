from datasets import load_dataset
import csv
from tqdm import tqdm

SEQLEN = 1024

def get_finetune_data(data_path: str):
    data = load_dataset(data_path)["test"]
    data = data.shuffle()

    MAX_TRAIN = 30000
    MAX_TEST = 5000

    data_name = data_path.split("/")[1].strip()
    with open(f"msa_seq1k_{data_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["chrom", "start", "end", "label"])
        for ix, item in tqdm(enumerate(data), total=MAX_TRAIN):
            writer.writerow([item['chrom'], int(item["pos"]), int(item["pos"])+SEQLEN, 1 if item["label"] else 0])
            if ix >= MAX_TRAIN: break

if __name__ == "__main__":
    get_finetune_data("songlab/clinvar")
