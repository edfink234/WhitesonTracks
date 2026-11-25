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
GOOD_SPLIT_CHI2 = 2.0
LOG_FACTOR_TWO = np.log(2.0)
ANNOTATE_FAKES = False
PLOT_BACKGROUND_ONLY = True

results = {}

for label, path in FILES.items():
    df = pd.read_csv(path)
    req = {"log_chi2_split_over_helix", "chi2nu_split", "chi2nu_helix"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path}: missing columns {sorted(miss)}")

    row_idx = np.arange(len(df))
    vals = df["log_chi2_split_over_helix"].to_numpy(float)
    split = df["chi2nu_split"].to_numpy(float)
    helix = df["chi2nu_helix"].to_numpy(float)

    m = np.isfinite(vals) & np.isfinite(split) & np.isfinite(helix) & (split > 0) & (helix > 0)
    vals, split, helix, row_idx = (
        vals[m],
        split[m],
        helix[m],
        row_idx[m],
    )

    results[label] = {
        "vals": vals,
        "split": split,
        "helix": helix,
        "row_idx": row_idx,
    }

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
    if PLOT_BACKGROUND_ONLY and label != "Background-only fakes":
        continue
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
    if PLOT_BACKGROUND_ONLY and label != "Background-only fakes":
        continue
    all_split.append(d["split"])
    all_helix.append(d["helix"])
    if label != "Background-only fakes":

        plt.scatter(
            d["helix"],
            d["split"],
            label=label,
            alpha=0.75,
        )

    else:

        mask_A = (
            d["split"] <= GOOD_SPLIT_CHI2
        )

        mask_B = (
            np.abs(d["vals"]) <= LOG_FACTOR_TWO
        )

        mask_C = (
            (d["split"] > GOOD_SPLIT_CHI2)
            &
            (d["vals"] < -LOG_FACTOR_TWO)
        )

        plt.scatter(
            d["helix"][mask_A],
            d["split"][mask_A],
            label="Fake A: easy split",
            alpha=0.85,
        )

        plt.scatter(
            d["helix"][mask_B],
            d["split"][mask_B],
            label="Fake B: split does little",
            alpha=0.85,
        )

        plt.scatter(
            d["helix"][mask_C],
            d["split"][mask_C],
            label="Fake C: intermediate",
            alpha=0.85,
        )
        
        if ANNOTATE_FAKES:
            for idx, x, y in zip(
                d["row_idx"],
                d["helix"],
                d["split"],
            ):
                plt.annotate(
                    str(idx),
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )

all_split = np.concatenate(all_split)
all_helix = np.concatenate(all_helix)
lo = min(all_split.min(), all_helix.min())
hi = max(all_split.max(), all_helix.max())
plt.plot([lo,hi], [lo,hi], linestyle="--", linewidth=1, color="black")
# A/B/C population boundaries

# A boundary: split fit becomes "good"
plt.axhline(
    GOOD_SPLIT_CHI2,
    linestyle="--",
    linewidth=1,
    color="black"
)

# B boundaries: split changes chi2 by less than factor 2
xline = np.logspace(
    np.log10(lo),
    np.log10(hi),
    200,
)

plt.plot(
    xline,
    0.5 * xline,
    linestyle="--",
    linewidth=1,
    color="black"
)

plt.plot(
    xline,
    2.0 * xline,
    linestyle="--",
    linewidth=1,
    color="black"
)
plt.xscale("log")
plt.yscale("log")
plt.text(
    hi / 100,
    0.5,
    "A",
    fontsize=14,
    fontweight="bold",
)

plt.text(
    hi / 5,
    hi / 5,
    "B",
    fontsize=14,
    fontweight="bold",
)

plt.text(
    hi / 30,
    1000,#hi / 300,
    "C",
    fontsize=14,
    fontweight="bold",
)
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
