import pickle

print("opening")
with open("msa_seq1k_30k.pkl", "rb") as f:
    print("loaded msa")
    msa = pickle.load(f)[:2500]
    print("selected msa")

print("dumping msa")
with open("msa_seq1k_2k.pkl", "wb") as f:
    pickle.dump(msa, f)