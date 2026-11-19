#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import system


SPLIT_FILES = {
    "SM":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_SM_100__split_helix.csv",

    "PID15":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_PID15_100__split_helix.csv",

    "Background-only fakes":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_fakes_100__split_helix.csv",
}

SR_FILES = {
    "SM":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_SM_100.csv",

    "PID15":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_PID15_100.csv",

    "Background-only fakes":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_fakes_100.csv",
}

OUT = (
    "../pngs/"
    "combined_split_vs_SR_discriminant_mu20.png"
)


plt.figure(figsize=(7, 6))

all_x = []
all_y = []

for label in SPLIT_FILES:

    split_df = pd.read_csv(SPLIT_FILES[label])
    sr_df = pd.read_csv(SR_FILES[label])

    # These runs should contain the same finder-output tracks
    # in the same order.
    if len(split_df) != len(sr_df):
        raise RuntimeError(
            f"{label}: split has {len(split_df)} rows "
            f"but SR has {len(sr_df)} rows."
        )

    x = split_df[
        "log_chi2_split_over_helix"
    ].to_numpy(float)

    y = sr_df[
        "log_chi2_sr_over_helix"
    ].to_numpy(float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    all_x.append(x)
    all_y.append(y)

    plt.scatter(
        x,
        y,
        alpha=0.75,
        label=label,
    )

    print(f"\n=== {label} ===")
    print(f"n = {len(x)}")
    print(
        "median split ratio = "
        f"{np.median(x):.3f}"
    )
    print(
        "median SR ratio    = "
        f"{np.median(y):.3f}"
    )


# Reference lines:
# x = 0 -> split helix and single helix equally good
# y = 0 -> SR and single helix equally good
plt.axvline(
    0.0,
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)

# y=x:
# below this line -> SR improves more than split helix
# above this line -> split helix improves more than SR
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

lo = min(
    np.min(all_x),
    np.min(all_y),
)

hi = max(
    np.max(all_x),
    np.max(all_y),
)

plt.plot(
    [lo, hi],
    [lo, hi],
    linestyle=":",
    linewidth=1,
    label="SR = split improvement",
)

plt.xlabel(
    r"$\log("
    r"\chi^2_{\nu,\mathrm{split}}/"
    r"\chi^2_{\nu,\mathrm{helix}}"
    r")$"
)

plt.ylabel(
    r"$\log("
    r"\chi^2_{\nu,\mathrm{SR}}/"
    r"\chi^2_{\nu,\mathrm{helix}}"
    r")$"
)

plt.title(
    "SR vs split-helix discriminants: "
    r"$\mu=20$ finder-output populations"
)

plt.legend()
plt.tight_layout()
plt.savefig(
    OUT,
    dpi=300,
)
plt.close()

print(f"\nSaved {OUT}")

system(f"open {OUT}")
