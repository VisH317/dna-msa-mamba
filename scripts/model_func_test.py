import torch
from src.model import MSAMambaForMLM, MSAMambaConfig
import pickle

config = MSAMambaConfig(
    n_layers=2,
    d_model=8,
    vocab_size=6,
    n_heads=2
)

def test_on_sample(file_path: str, idx: int = 0):
    with open(file_path, "rb") as f:
        sample = pickle.load(f)[idx]
    
    sample = torch.as_tensor(sample, dtype=torch.long).cuda().unsqueeze(0).permute(0, 2, 1)

    b, m, l = sample.size()

    print(sample.size())

    print("sample: ", sample)

    model = MSAMambaForMLM(config).cuda()

    out = model(sample)
    print(out)

    print(torch.nonzero(torch.isnan(out.view(-1))))


if __name__ == "__main__":
    test_on_sample("data/msa_dummy.pkl", 12)