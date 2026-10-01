import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NLAYERS = 25
LAYER_IDS = np.arange(1, NLAYERS + 1)


# ---------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------

def read_ryan_helical_csv(path):
    """
    Ryan format:

        d0, phi0, pt, dz, tanl
        layer_id, x, y, z
        layer_id, x, y, z
        ...
        EOT

    Returns one row per track with detector-footprint diagnostics.
    """
    rows = []

    with open(path, "r") as f:
        track_idx = -1
        params = None
        layer_ids = []
        xs, ys, zs = [], [], []

        def flush_track():
            if params is None:
                return None

            d0, phi0, pt, dz, tanl = params
            lids = np.asarray(layer_ids, dtype=int)
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            z = np.asarray(zs, dtype=float)

            return summarize_one_track(
                label="helical",
                source="ryan_csv",
                track_idx=track_idx,
                layer_ids=lids,
                x=x,
                y=y,
                z=z,
                extra={
                    "d0": d0,
                    "phi0": phi0,
                    "pt": pt,
                    "dz": dz,
                    "tanl": tanl,
                },
            )

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "EOT":
                row = flush_track()
                if row is not None:
                    rows.append(row)

                params = None
                layer_ids = []
                xs, ys, zs = [], [], []
                continue

            vals = [float(v.strip()) for v in line.split(",")]

            if params is None:
                track_idx += 1
                params = vals
            else:
                layer_ids.append(int(vals[0]))
                xs.append(vals[1])
                ys.append(vals[2])
                zs.append(vals[3])

    return pd.DataFrame(rows)


def read_make_tracks_folder(folder, label):
    """
    Your make_tracks_60.py format:

        event*-hits.csv

    Expected columns include:
        r, phi, z, layer_id

    Returns one row per event/track file with detector-footprint diagnostics.
    """
    rows = []
    paths = sorted(glob.glob(os.path.join(folder, "event*-hits.csv")))

    if len(paths) == 0:
        raise FileNotFoundError(f"No event*-hits.csv files found in {folder}")

    for track_idx, path in enumerate(paths):
        hits = pd.read_csv(path)

        if "layer_id" not in hits.columns:
            raise ValueError(f"{path} has no layer_id column")

        layer_ids = hits["layer_id"].to_numpy(dtype=int)

        if {"x", "y", "z"}.issubset(hits.columns):
            x = hits["x"].to_numpy(float)
            y = hits["y"].to_numpy(float)
            z = hits["z"].to_numpy(float)
        else:
            r = hits["r"].to_numpy(float)
            phi = hits["phi"].to_numpy(float)
            z = hits["z"].to_numpy(float)
            x = r * np.cos(phi)
            y = r * np.sin(phi)

        rows.append(
            summarize_one_track(
                label=label,
                source="make_tracks_folder",
                track_idx=track_idx,
                layer_ids=layer_ids,
                x=x,
                y=y,
                z=z,
                extra={"file": os.path.basename(path)},
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_one_track(label, source, track_idx, layer_ids, x, y, z, extra=None):
    extra = extra or {}

    layer_ids = np.asarray(layer_ids, dtype=int)
    unique_layers = np.unique(layer_ids)

    if len(unique_layers) == 0:
        n_hits = 0
        innermost_layer = np.nan
        outermost_layer = np.nan
        missing_layers = NLAYERS
        contiguous_from_inner = False
    else:
        n_hits = len(layer_ids)
        innermost_layer = int(np.min(unique_layers))
        outermost_layer = int(np.max(unique_layers))
        missing_layers = NLAYERS - len(unique_layers)

        expected = np.arange(innermost_layer, outermost_layer + 1)
        contiguous_from_inner = np.array_equal(unique_layers, expected)

    r = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2) if len(x) else np.array([])

    row = {
        "label": label,
        "source": source,
        "track_idx": track_idx,
        "n_hits": n_hits,
        "n_unique_layers": len(unique_layers),
        "innermost_layer": innermost_layer,
        "outermost_layer": outermost_layer,
        "missing_layers": missing_layers,
        "frac_layers_hit": len(unique_layers) / NLAYERS,
        "hits_layer_1": int(1 in unique_layers),
        "hits_layer_2": int(2 in unique_layers),
        "hits_inner_2": int((1 in unique_layers) or (2 in unique_layers)),
        "contiguous_layers": bool(contiguous_from_inner),
        "min_r_hit": float(np.min(r)) if len(r) else np.nan,
        "max_r_hit": float(np.max(r)) if len(r) else np.nan,
        "max_abs_z": float(np.max(np.abs(z))) if len(z) else np.nan,
    }

    row.update(extra)
    return row


def layer_occupancy_from_summary_and_hits_make_tracks(folder, label):
    """
    For make_tracks folders, compute layer occupancy directly from hit CSVs.
    """
    counts = np.zeros(NLAYERS, dtype=int)
    paths = sorted(glob.glob(os.path.join(folder, "event*-hits.csv")))

    for path in paths:
        hits = pd.read_csv(path)
        unique_layers = np.unique(hits["layer_id"].to_numpy(dtype=int))
        for lid in unique_layers:
            if 1 <= lid <= NLAYERS:
                counts[lid - 1] += 1

    return pd.DataFrame({
        "label": label,
        "layer_id": LAYER_IDS,
        "occupancy": counts / max(len(paths), 1),
        "count": counts,
    })


def layer_occupancy_from_ryan_csv(path, label="helical"):
    counts = np.zeros(NLAYERS, dtype=int)
    n_tracks = 0
    current_layers = []

    with open(path, "r") as f:
        expecting_params = True

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "EOT":
                n_tracks += 1
                for lid in np.unique(current_layers):
                    if 1 <= lid <= NLAYERS:
                        counts[lid - 1] += 1
                current_layers = []
                expecting_params = True
                continue

            vals = [float(v.strip()) for v in line.split(",")]

            if expecting_params:
                expecting_params = False
            else:
                current_layers.append(int(vals[0]))

    return pd.DataFrame({
        "label": label,
        "layer_id": LAYER_IDS,
        "occupancy": counts / max(n_tracks, 1),
        "count": counts,
    })


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def overlay_hist(df, column, outdir, bins=None, xlabel=None, title=None):
    plt.figure(figsize=(7, 5))

    for label, sub in df.groupby("label"):
        vals = sub[column].dropna()
        plt.hist(vals, bins=bins, histtype="step", linewidth=2, label=f"{label}, n={len(vals)}")

    plt.xlabel(xlabel or column)
    plt.ylabel("Tracks")
    plt.title(title or column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"compare_{column}.png"), dpi=300)
    plt.close()


def overlay_layer_occupancy(occ_df, outdir):
    plt.figure(figsize=(7, 5))

    for label, sub in occ_df.groupby("label"):
        sub = sub.sort_values("layer_id")
        plt.plot(sub["layer_id"], sub["occupancy"], marker="o", label=label)

    plt.xlabel("Layer ID")
    plt.ylabel("Fraction of tracks with hit")
    plt.title("Layer occupancy comparison")
    plt.xticks(np.arange(1, NLAYERS + 1, 2))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_layer_occupancy.png"), dpi=300)
    plt.close()


def make_comparison_plots(summary_df, occ_df, outdir):
    os.makedirs(outdir, exist_ok=True)

    summary_df.to_csv(os.path.join(outdir, "detector_footprint_summary.csv"), index=False)
    occ_df.to_csv(os.path.join(outdir, "layer_occupancy_summary.csv"), index=False)

    print("\n=== Detector footprint summary ===")
    print(summary_df.groupby("label")[
        [
            "n_hits",
            "n_unique_layers",
            "innermost_layer",
            "outermost_layer",
            "missing_layers",
            "frac_layers_hit",
            "hits_layer_1",
            "hits_layer_2",
            "hits_inner_2",
            "contiguous_layers",
            "max_abs_z",
        ]
    ].mean())

    overlay_hist(
        summary_df,
        "n_hits",
        outdir,
        bins=np.arange(0.5, NLAYERS + 1.5, 1),
        xlabel="Number of hits",
        title="Hit-count comparison",
    )

    overlay_hist(
        summary_df,
        "innermost_layer",
        outdir,
        bins=np.arange(0.5, NLAYERS + 1.5, 1),
        xlabel="Innermost layer hit",
        title="Innermost-layer comparison",
    )

    overlay_hist(
        summary_df,
        "outermost_layer",
        outdir,
        bins=np.arange(0.5, NLAYERS + 1.5, 1),
        xlabel="Outermost layer hit",
        title="Outermost-layer comparison",
    )

    overlay_hist(
        summary_df,
        "missing_layers",
        outdir,
        bins=np.arange(-0.5, NLAYERS + 1.5, 1),
        xlabel="Number of missing layers",
        title="Missing-layer comparison",
    )

    overlay_layer_occupancy(occ_df, outdir)

    print(f"\nWrote comparison plots to: {outdir}")
    os.system(f"open {outdir}")

# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------

if __name__ == "__main__":

    RYAN_HELIX_CSV = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/RyanDanielHelical/validation_10k.csv"

    MAKE_TRACKS_SAMPLES = {
        "weird25": "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/validation_weird25_10k",
    }

    OUTDIR = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/RyanDanielHelical/detector_footprint_validation_helical_vs_weird25"

    summaries = []
    occupancies = []

    # Ryan helix sample
    summaries.append(read_ryan_helical_csv(RYAN_HELIX_CSV))
    occupancies.append(layer_occupancy_from_ryan_csv(RYAN_HELIX_CSV, label="helical"))

    # Your make_tracks_60 samples
    for label, folder in MAKE_TRACKS_SAMPLES.items():
        summaries.append(read_make_tracks_folder(folder, label=label))
        occupancies.append(layer_occupancy_from_summary_and_hits_make_tracks(folder, label=label))

    summary_df = pd.concat(summaries, ignore_index=True)
    occ_df = pd.concat(occupancies, ignore_index=True)

    make_comparison_plots(summary_df, occ_df, OUTDIR)
