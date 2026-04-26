from datasets import load_dataset
import os

# load small subset (IMPORTANT)
dataset = load_dataset("PleIAs/SEC", split="train[:100]")

os.makedirs("data/raw", exist_ok=True)

def chunk_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

for i, row in enumerate(dataset):
    chunks = chunk_text(row["text"])
    for j, chunk in enumerate(chunks):
        with open(f"data/raw/doc_{i}_{j}.txt", "w") as f:
            f.write(chunk)

print("Done creating raw text files.")