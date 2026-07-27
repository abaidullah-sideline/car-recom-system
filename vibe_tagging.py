import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import time
import os

cars = pd.read_csv("cars_cleaned.csv")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

vibe_label_map = {
    "a family-friendly car with practical space": "vibe_family",
    "a sporty high-performance car": "vibe_sporty",
    "a luxury car with premium features": "vibe_luxury",
    "a fuel-efficient eco-friendly car": "vibe_ecofriendly",
    "an off-road or rugged utility vehicle": "vibe_offroad",
    "a basic reliable daily commuter car": "vibe_commuter",
}
vibe_labels = list(vibe_label_map.keys())
hypothesis_template = "This vehicle listing describes {}."

def build_classifier_input(row):
    type_phrase = f"a {row['type']}" if row["type"] != "unknown" else "a vehicle"
    return f"{int(row['year'])} {row['manufacturer']} {row['model']}, {type_phrase} in {row['condition']} condition. {row['description_clean']}"

cars["classifier_input"] = cars.apply(build_classifier_input, axis=1)

CHECKPOINT_FILE = "vibe_checkpoint.csv"
BATCH_SIZE = 16
texts = cars["classifier_input"].tolist()

# resume support: pick up where a prior run left off
all_results = []
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    prev = pd.read_csv(CHECKPOINT_FILE)
    all_results = prev.to_dict("records")
    start_idx = len(all_results)
    print(f"Resuming from row {start_idx}")

start = time.time()
for i in tqdm(range(start_idx, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    batch_results = classifier(batch, vibe_labels, multi_label=False,
                                hypothesis_template=hypothesis_template, batch_size=BATCH_SIZE)
    for res in batch_results:
        all_results.append(dict(zip(res["labels"], res["scores"])))

    if (i // BATCH_SIZE) % 20 == 0:
        pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)
        print(f"Checkpoint @ row {len(all_results)}: {(time.time()-start)/60:.1f} min elapsed")

vibes_df = pd.DataFrame(all_results).rename(columns=vibe_label_map)
cars_with_vibes = pd.concat([cars.reset_index(drop=True), vibes_df], axis=1)
cars_with_vibes = cars_with_vibes.drop(columns=["classifier_input"])
cars_with_vibes.to_csv("cars_with_vibes.csv", index=False)
print("Done:", cars_with_vibes.shape)