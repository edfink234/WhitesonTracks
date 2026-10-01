from pathlib import Path
import pandas as pd
import numpy as np

folder = Path("/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/validation_weird25_10k")

rows = []
for path in sorted(folder.glob("event*-hits.csv")):
    hits = pd.read_csv(path)
    layers = hits["layer_id"].astype(int).to_list()
    unique_layers = sorted(set(layers))

    rows.append({
        "file": path.name,
        "n_hits": len(layers),
        "n_unique_layers": len(unique_layers),
        "n_duplicate_hits": len(layers) - len(unique_layers),
        "layers": layers,
    })

df = pd.DataFrame(rows)

dups = df[df["n_duplicate_hits"] > 0].copy()
print("tracks with duplicate-layer hits:", len(dups), "/", len(df))
print(dups[["file", "n_hits", "n_unique_layers", "n_duplicate_hits"]].head(20).to_string(index=False))

print("\nExample layer sequences:")
for _, row in dups.head(10).iterrows():
    print(row["file"], row["layers"])
