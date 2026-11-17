#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import system

FILES = {
    "SM": "../data_files/split_helix_ratio_values_mu20_finder_SM_100__split_helix.csv",
    "PID15": "../data_files/split_helix_ratio_values_mu20_finder_PID15_100__split_helix.csv",
    "Background-only fakes": "../data_files/split_helix_ratio_values_mu20_finder_fakes_100__split_helix.csv",
}

OUT_HIST = "../pngs/combined_split_helix_ratio_populations_mu20.png"
OUT_SCATTER = "../pngs/combined_split_helix_scatter_populations_mu20_loglog.png"

results = {}

for label, path in FILES.items():
    df = pd.read_csv(path)
    req = {"log_chi2_split_over_helix", "chi2nu_split", "chi2nu_helix"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path}: missing columns {sorted(miss)}")

    vals = df["log_chi2_split_over_helix"].to_numpy(float)
    split = df["chi2nu_split"].to_numpy(float)
    helix = df["chi2nu_helix"].to_numpy(float)

    m = np.isfinite(vals) & np.isfinite(split) & np.isfinite(helix) & (split > 0) & (helix > 0)
    vals, split, helix = vals[m], split[m], helix[m]

    results[label] = {"vals": vals, "split": split, "helix": helix}

    print(f"\n=== {label} ===")
    print(f"n = {len(vals)}")
    print(f"mean   = {np.mean(vals):.4f}")
    print(f"median = {np.median(vals):.4f}")
    print(f"std    = {np.std(vals, ddof=1) if len(vals)>1 else 0.0:.4f}")
    print(f"fraction split better = {np.mean(vals < 0):.4f}")

plt.figure(figsize=(8,5))
all_vals = np.concatenate([d["vals"] for d in results.values()])
bins = np.linspace(all_vals.min(), all_vals.max(), 21)
for label, d in results.items():
    vals = d["vals"]
    plt.hist(vals, bins=bins, histtype="step", linewidth=2,
             label=f"{label}: mean={np.mean(vals):.2f}, med={np.median(vals):.2f}, sd={np.std(vals,ddof=1):.2f}")
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel(r"$\log(\chi^2_{\nu,\mathrm{split}}/\chi^2_{\nu,\mathrm{helix}})$")
plt.ylabel("Tracks")
plt.title("Split helix vs single helix: mu=20 finder-output populations")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_HIST, dpi=300)
plt.close()

plt.figure(figsize=(6,6))
all_split, all_helix = [], []
for label, d in results.items():
    all_split.append(d["split"])
    all_helix.append(d["helix"])
    plt.scatter(d["helix"], d["split"], label=label, alpha=0.75)

all_split = np.concatenate(all_split)
all_helix = np.concatenate(all_helix)
lo = min(all_split.min(), all_helix.min())
hi = max(all_split.max(), all_helix.max())
plt.plot([lo,hi], [lo,hi], linestyle="--", linewidth=1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\chi^2_{\nu,\mathrm{helix}}$")
plt.ylabel(r"$\chi^2_{\nu,\mathrm{split}}$")
plt.title("Split helix vs single helix: mu=20 finder-output populations")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=300)
plt.close()

print(f"Saved {OUT_HIST}")
print(f"Saved {OUT_SCATTER}")
system(f"open {OUT_HIST}")
system(f"open {OUT_SCATTER}")
