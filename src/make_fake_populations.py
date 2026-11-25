#!/usr/bin/env python3

from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

FAKE_DIR = Path(
    "../tracks_for_ed/mu20_finder_fakes_100"
).resolve()

SPLIT_CSV = Path(
    "../data_files/"
    "split_helix_ratio_values_mu20_finder_fakes_100__split_helix.csv"
).resolve()

DISPLAY_DIR = Path(
    "../pngs/mu20_finder_event_displays/background_fake"
).resolve()

OUT_DIR = Path("../data_files").resolve()


# ------------------------------------------------------------
# Population definitions
#
# A:
#   Split helix gives a genuinely good fit.
#   These are the obvious/easy merged-track fakes.
#
# B:
#   Split helix does little compared with single helix.
#   These are the hard / "real" fakes Daniel is interested in.
#
# C:
#   Split helps substantially, but the resulting fit is still bad.
#   These are the intermediate population to inspect.
# ------------------------------------------------------------

GOOD_SPLIT_CHI2 = 2.0

# "Split does little" = improvement smaller than a factor of 2.
LOG_FACTOR_TWO = np.log(2.0)


# ------------------------------------------------------------
# Load split-fit results
# ------------------------------------------------------------

df = pd.read_csv(SPLIT_CSV)

required_columns = [
    "chi2nu_helix",
    "chi2nu_split",
    "log_chi2_split_over_helix",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns in {SPLIT_CSV}:\n{missing}"
    )


# ------------------------------------------------------------
# Match each CSV row to its input fake-track file
#
# IMPORTANT:
# This assumes fit_tracks.py processed the input files in the
# same sorted filename order used here.
# ------------------------------------------------------------

hit_files = sorted(
    FAKE_DIR.glob("*.csv")
)

if len(hit_files) != len(df):
    raise RuntimeError(
        f"Found {len(hit_files)} fake hit files but "
        f"{len(df)} rows in the split-fit CSV."
    )

df["hit_file"] = [
    str(path.resolve())
    for path in hit_files
]

def nearest_neighbor_order(x, y, z, r=None):
    """
    MUST match fit_tracks.py ordering.
    """
    xyz = np.column_stack([x, y, z])
    n = len(xyz)

    if n <= 1:
        return np.arange(n)

    if r is None:
        r = np.sqrt(np.asarray(x)**2 + np.asarray(y)**2)

    remaining = set(range(n))
    current = int(np.argmin(r))

    order = [current]
    remaining.remove(current)

    while remaining:
        candidates = np.array(sorted(remaining), dtype=int)

        d2 = np.sum(
            (xyz[candidates] - xyz[current])**2,
            axis=1,
        )

        current = int(candidates[np.argmin(d2)])

        order.append(current)
        remaining.remove(current)

    return np.asarray(order, dtype=int)


def truth_structure(row):

    hits = pd.read_csv(row["hit_file"])

    x = hits["x"].to_numpy(dtype=float)
    y = hits["y"].to_numpy(dtype=float)
    z = hits["z"].to_numpy(dtype=float)

    if "r" in hits.columns:
        r = hits["r"].to_numpy(dtype=float)
    else:
        r = np.sqrt(x**2 + y**2)

    # IMPORTANT:
    # Put truth IDs into exactly the same ordering used by fit_tracks.py.
    order = nearest_neighbor_order(x, y, z, r)

    pids = hits["particle_id"].to_numpy()[order]

    unique, counts = np.unique(
        pids,
        return_counts=True,
    )

    count_order = np.argsort(counts)[::-1]
    counts = counts[count_order]

    n_hits = len(pids)
    n_contributors = len(unique)

    dominant_fraction = (
        counts[0] / n_hits
        if n_hits > 0
        else np.nan
    )

    second_fraction = (
        counts[1] / n_hits
        if len(counts) > 1
        else 0.0
    )

    # Locations where truth identity changes.
    #
    # +1 is crucial:
    # if pids[9] != pids[10],
    # then truth_split_idx = 10,
    # matching fit_tracks.py's definition that split_idx
    # is the FIRST hit of the right-hand segment.
    switches = np.flatnonzero(
        pids[1:] != pids[:-1]
    ) + 1

    n_truth_switches = len(switches)

    # All your current fakes appear to have one switch.
    fitted_split_idx = int(row["split_idx"])

    if len(switches) > 0:
        nearest_truth_switch_idx = int(
            switches[np.argmin(np.abs(switches - fitted_split_idx))]
        )

        split_offset = (
            fitted_split_idx - nearest_truth_switch_idx
        )

        abs_split_offset = abs(split_offset)

    else:
        nearest_truth_switch_idx = np.nan
        split_offset = np.nan
        abs_split_offset = np.nan
    
    left_pids = pids[:fitted_split_idx]
    right_pids = pids[fitted_split_idx:]

    def segment_purity(segment):
        if len(segment) == 0:
            return np.nan

        _, counts = np.unique(
            segment,
            return_counts=True,
        )

        return counts.max() / len(segment)


    left_truth_purity = segment_purity(left_pids)
    right_truth_purity = segment_purity(right_pids)

    mean_side_purity = np.nanmean([
        left_truth_purity,
        right_truth_purity,
    ])

    return pd.Series({
        "n_hits": n_hits,
        "n_truth_contributors": n_contributors,
        "dominant_truth_fraction": dominant_fraction,
        "second_truth_fraction": second_fraction,
        "n_truth_switches": n_truth_switches,

        "nearest_truth_switch_idx": nearest_truth_switch_idx,
        "fitted_split_idx": fitted_split_idx,
        "split_offset": split_offset,
        "abs_split_offset": abs_split_offset,
        "left_truth_purity": left_truth_purity,
        "right_truth_purity": right_truth_purity,
        "mean_side_purity": mean_side_purity,
    })


truth_info = df.apply(
    truth_structure,
    axis=1,
)

df = pd.concat(
    [df, truth_info],
    axis=1,
)


# ------------------------------------------------------------
# Remove any non-finite fit results
# ------------------------------------------------------------

finite = (
    np.isfinite(df["chi2nu_helix"])
    & np.isfinite(df["chi2nu_split"])
    & np.isfinite(df["log_chi2_split_over_helix"])
)

df = df.loc[finite].copy()


# ------------------------------------------------------------
# Define A / B / C
# ------------------------------------------------------------

# A:
# split helix essentially solves the track
mask_A = (
    df["chi2nu_split"]
    <= GOOD_SPLIT_CHI2
)

# B:
# split helix changes the fit by less than a factor of 2
mask_B = (
    np.abs(
        df["log_chi2_split_over_helix"]
    )
    <= LOG_FACTOR_TWO
)

# C:
# split substantially improves the fit,
# but does not actually make it a good fit
mask_C = (
    (df["chi2nu_split"] > GOOD_SPLIT_CHI2)
    &
    (
        df["log_chi2_split_over_helix"]
        < -LOG_FACTOR_TWO
    )
)

A = df.loc[mask_A].copy()
B = df.loc[mask_B].copy()
C = df.loc[mask_C].copy()


# ------------------------------------------------------------
# Check that populations do not overlap
# ------------------------------------------------------------

overlap_AB = set(A.index) & set(B.index)
overlap_AC = set(A.index) & set(C.index)
overlap_BC = set(B.index) & set(C.index)

if overlap_AB or overlap_AC or overlap_BC:
    print("\nWARNING: population definitions overlap.")
    print("A/B overlap:", len(overlap_AB))
    print("A/C overlap:", len(overlap_AC))
    print("B/C overlap:", len(overlap_BC))


# ------------------------------------------------------------
# Save and print results
# ------------------------------------------------------------

populations = {
    "A_easy_split": A,
    "B_hard_fake": B,
    "C_intermediate": C,
}

print("\n" + "=" * 70)
print("OPTIMAL SPLIT VS TRUTH TRANSITION")
print("=" * 70)

for name, pop in populations.items():

    d = pop["abs_split_offset"].dropna()

    print(f"\n=== {name} ===")
    print(f"N = {len(d)}")

    print(
        f"median |best split - truth split| = "
        f"{d.median():.2f} hits"
    )

    print(
        f"mean |best split - truth split| = "
        f"{d.mean():.2f} hits"
    )

    print(
        f"exactly aligned: "
        f"{np.mean(d == 0):.3f}"
    )

    print(
        f"within 1 hit: "
        f"{np.mean(d <= 1):.3f}"
    )

    print(
        f"within 2 hits: "
        f"{np.mean(d <= 2):.3f}"
    )

    print(
        f"within 3 hits: "
        f"{np.mean(d <= 3):.3f}"
    )

print("\n" + "=" * 70)
print("TRUTH-STRUCTURE COMPARISON")
print("=" * 70)

for name, pop in populations.items():

    print(f"\n=== {name} ===")

    print(
        pop[
            [
                "n_hits",
                "n_truth_contributors",
                "dominant_truth_fraction",
                "second_truth_fraction",
                "n_truth_switches",
            ]
        ].describe()
    )

    print(
        "\ntruth-switch counts:"
    )

    print(
        pop["n_truth_switches"]
        .value_counts()
        .sort_index()
    )

print("\n" + "=" * 70)
print("FAKE POPULATION SUMMARY")
print("=" * 70)

print(f"\nTotal finite fake tracks: {len(df)}")

for name, pop in populations.items():

    print(
        f"\n{name}: "
        f"{len(pop)} tracks"
    )

    print("-" * 70)
    print(r'cp ', end='')

    for _, row in pop.iterrows():

        hit_path = Path(row["hit_file"])

        event_stem = hit_path.stem.replace("-hits", "")

        png_path = DISPLAY_DIR / f"{event_stem}.png"
        png_path = str(png_path).replace("/event", "/*event").replace(".png", "*.png")
        print(f'{png_path} ', end="")
    
    print(f"~/temp_fakes_{name}/")
    print()
    out_csv = (
        OUT_DIR
        / f"fake_population_{name}.csv"
    )

    pop.to_csv(
        out_csv,
        index=False,
    )

    print(
        f"\nSaved: {out_csv}"
    )


# ------------------------------------------------------------
# Also write plain-text path lists for easy terminal use
# ------------------------------------------------------------

for name, pop in populations.items():

    out_txt = (
        OUT_DIR
        / f"fake_population_{name}_paths.txt"
    )

    with open(out_txt, "w") as f:
        for path in pop["hit_file"]:
            f.write(path + "\n")

    print(
        f"Saved path list: {out_txt}"
    )


# ------------------------------------------------------------
# Anything not assigned?
# ------------------------------------------------------------

assigned = (
    set(A.index)
    | set(B.index)
    | set(C.index)
)

unassigned = df.loc[
    ~df.index.isin(assigned)
].copy()

print(
    f"\nUnassigned tracks: "
    f"{len(unassigned)}"
)

if len(unassigned) > 0:

    out_unassigned = (
        OUT_DIR
        / "fake_population_unassigned.csv"
    )

    unassigned.to_csv(
        out_unassigned,
        index=False,
    )

    print(
        f"Saved: {out_unassigned}"
    )

for name, pop in populations.items():
    print(
        f"\n{name}: "
        f"median dominant fraction = "
        f"{pop['dominant_truth_fraction'].median():.3f}, "
        f"median switches = "
        f"{pop['n_truth_switches'].median():.1f}"
    )

print("\n" + "=" * 70)
print("SPLIT-ALIGNMENT CORRELATIONS")
print("=" * 70)

valid = df[
    np.isfinite(df["abs_split_offset"])
    &
    np.isfinite(df["log_chi2_split_over_helix"])
    &
    np.isfinite(df["chi2nu_split"])
].copy()


rho_ratio, p_ratio = spearmanr(
    valid["abs_split_offset"],
    valid["log_chi2_split_over_helix"],
)

rho_split_chi2, p_split_chi2 = spearmanr(
    valid["abs_split_offset"],
    valid["chi2nu_split"],
)


print(
    "\n|split offset| vs log(split/helix):"
)
print(
    f"Spearman rho = {rho_ratio:.3f}, "
    f"p = {p_ratio:.3e}"
)

print(
    "\n|split offset| vs split chi2nu:"
)
print(
    f"Spearman rho = {rho_split_chi2:.3f}, "
    f"p = {p_split_chi2:.3e}"
)

plt.figure(figsize=(7, 5))

for name, pop in populations.items():
    plt.scatter(
        pop["abs_split_offset"],
        pop["log_chi2_split_over_helix"],
        label=name,
        alpha=0.8,
    )

plt.axhline(0.0, linestyle="--", linewidth=1)

plt.xlabel(
    "|best split index - truth transition index| [hits]"
)

plt.ylabel(
    r"$\log(\chi^2_{\nu,\mathrm{split}}/"
    r"\chi^2_{\nu,\mathrm{helix}})$"
)

plt.legend()
plt.tight_layout()

out_png = (
    OUT_DIR
    / "fake_split_alignment_vs_improvement.png"
)

plt.savefig(
    out_png,
    dpi=300,
)

plt.close()

print(f"Saved: {out_png}")

print("\n" + "=" * 70)
print("TRUTH SEPARATION AT BEST SPLIT")
print("=" * 70)

for name, pop in populations.items():

    print(f"\n=== {name} ===")

    print(
        f"median truth switches = "
        f"{pop['n_truth_switches'].median():.1f}"
    )

    print(
        f"median distance to nearest truth switch = "
        f"{pop['abs_split_offset'].median():.1f} hits"
    )

    print(
        f"median left purity = "
        f"{pop['left_truth_purity'].median():.3f}"
    )

    print(
        f"median right purity = "
        f"{pop['right_truth_purity'].median():.3f}"
    )

    print(
        f"median mean-side purity = "
        f"{pop['mean_side_purity'].median():.3f}"
    )
