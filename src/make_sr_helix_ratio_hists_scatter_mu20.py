#!/usr/bin/env python3
"""
Combine the three mu=20 SR-vs-helix finder-output populations into:
  1) histogram of log(chi2nu_SR / chi2nu_helix)
  2) log-log scatter of chi2nu_helix vs chi2nu_SR

Reads the direct CSV outputs written by fit_tracks.py.
No HTML parsing. No post-hoc DOF reconstruction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import system

POP_B_ONLY = True

FILES = {
    "SM": "../data_files/chi2_ratio_values_mu20_finder_SM_100.csv",
    "PID15": "../data_files/chi2_ratio_values_mu20_finder_PID15_100.csv",
    f"Background-only fakes{' (B) ' if POP_B_ONLY else ''}": ["../data_files/chi2_ratio_values_mu20_finder_fakes_100.csv", "../data_files/chi2_ratio_values_mu20_finder_fakes_100_popB.csv"][POP_B_ONLY],
}

OUT_HIST = "../pngs/combined_SR_helix_ratio_populations_mu20.png"
OUT_SCATTER = "../pngs/combined_SR_helix_scatter_populations_mu20_loglog.png"

results = {}

for label, path in FILES.items():
    df = pd.read_csv(path)

    required = {
        "log_chi2_sr_over_helix",
        "chi2nu_sr",
        "chi2nu_helix",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path}: missing required columns {sorted(missing)}"
        )

    vals = df["log_chi2_sr_over_helix"].to_numpy(float)
    chi2_sr = df["chi2nu_sr"].to_numpy(float)
    chi2_helix = df["chi2nu_helix"].to_numpy(float)

    mask = (
        np.isfinite(vals)
        & np.isfinite(chi2_sr)
        & np.isfinite(chi2_helix)
        & (chi2_sr > 0)
        & (chi2_helix > 0)
    )

    vals = vals[mask]
    chi2_sr = chi2_sr[mask]
    chi2_helix = chi2_helix[mask]

    results[label] = {
        "vals": vals,
        "chi2_sr": chi2_sr,
        "chi2_helix": chi2_helix,
    }

    print(f"\n=== {label} ===")
    print(f"n = {len(vals)}")
    print(f"mean log ratio   = {np.mean(vals):.4f}")
    print(f"median log ratio = {np.median(vals):.4f}")
    print(
        "std log ratio    = "
        f"{np.std(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}"
    )
    print(
        "fraction SR better (log ratio < 0) = "
        f"{np.mean(vals < 0):.4f}"
    )


# ------------------------------------------------------------
# Combined histogram
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

all_vals = np.concatenate([d["vals"] for d in results.values()])
bins = np.linspace(all_vals.min(), all_vals.max(), 21)

for label, d in results.items():
    vals = d["vals"]

    mean = np.mean(vals)
    median = np.median(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    plt.hist(
        vals,
        bins=bins,
        histtype="step",
        linewidth=2,
        label=(
            f"{label}: mean={mean:.2f}, "
            f"med={median:.2f}, sd={sd:.2f}"
        ),
    )

plt.axvline(0.0, linestyle="--", linewidth=1)

plt.xlabel(
    r"$\log(\chi^2_{\nu,\mathrm{SR}}/"
    r"\chi^2_{\nu,\mathrm{helix}})$"
)
plt.ylabel("Tracks")
plt.title("SR vs single helix: mu=20 finder-output populations")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_HIST, dpi=300)
plt.close()

print(f"\nSaved {OUT_HIST}")


# ------------------------------------------------------------
# Combined log-log scatter
# x = helix
# y = SR
# Below diagonal => SR better
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))

all_sr = []
all_helix = []

for label, d in results.items():
    chi2_sr = d["chi2_sr"]
    chi2_helix = d["chi2_helix"]

    all_sr.append(chi2_sr)
    all_helix.append(chi2_helix)

    plt.scatter(
        chi2_helix,
        chi2_sr,
        label=label,
        alpha=0.75,
    )

all_sr = np.concatenate(all_sr)
all_helix = np.concatenate(all_helix)

lo = min(all_sr.min(), all_helix.min())
hi = max(all_sr.max(), all_helix.max())

plt.plot(
    [lo, hi],
    [lo, hi],
    linestyle="--",
    linewidth=1,
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\chi^2_{\nu,\mathrm{helix}}$")
plt.ylabel(r"$\chi^2_{\nu,\mathrm{SR}}$")
plt.title("SR vs single helix: mu=20 finder-output populations")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SCATTER, dpi=300)
plt.close()

print(f"Saved {OUT_SCATTER}")

system(f"open {OUT_HIST}")
system(f"open {OUT_SCATTER}")
