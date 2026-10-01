from pathlib import Path
import pandas as pd
import numpy as np

folder = Path("/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/validation_weird25_10k")

max_B_radius = 53.0 - 0.1
detector_length = 320.0
z_half = detector_length / 2.0

rows = []

for path in sorted(folder.glob("event*-hits.csv")):
    hits = pd.read_csv(path).copy()
    hits["layer_id"] = hits["layer_id"].astype(int)

    layers = hits["layer_id"].to_list()
    n_hits = len(hits)
    n_unique = hits["layer_id"].nunique()
    n_dup = n_hits - n_unique

    exit_mask = (hits["r"].to_numpy(float) >= max_B_radius) | (
        np.abs(hits["z"].to_numpy(float)) >= z_half
    )

    exit_indices = np.where(exit_mask)[0]

    if len(exit_indices) == 0:
        first_exit_idx = np.nan
        n_hits_after_exit = 0
        n_dup_after_exit = 0
        layers_after_exit = []
    else:
        first_exit_idx = int(exit_indices[0])
        after = hits.iloc[first_exit_idx + 1:].copy()
        n_hits_after_exit = len(after)
        n_dup_after_exit = len(after) - after["layer_id"].nunique() if len(after) else 0
        layers_after_exit = after["layer_id"].astype(int).to_list()

    rows.append({
        "file": path.name,
        "n_hits": n_hits,
        "n_unique_layers": n_unique,
        "n_duplicate_hits": n_dup,
        "first_exit_idx": first_exit_idx,
        "n_hits_after_exit": n_hits_after_exit,
        "n_dup_after_exit": n_dup_after_exit,
        "layers": layers,
        "layers_after_exit": layers_after_exit,
    })

df = pd.DataFrame(rows)

print("tracks:", len(df))
print("tracks with duplicate-layer hits:", (df["n_duplicate_hits"] > 0).sum())
print("tracks with any hits after first exit:", (df["n_hits_after_exit"] > 0).sum())
print("total duplicate hits:", df["n_duplicate_hits"].sum())
print("total hits after first exit:", df["n_hits_after_exit"].sum())
print("total duplicate hits after first exit:", df["n_dup_after_exit"].sum())

print("\nExamples with hits after first exit:")
ex = df[df["n_hits_after_exit"] > 0].head(10)
for _, row in ex.iterrows():
    print(row["file"])
    print("  full layers:       ", row["layers"])
    print("  after-exit layers: ", row["layers_after_exit"])
