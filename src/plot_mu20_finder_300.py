#!/usr/bin/env python3
"""
Make sanity-check displays for the 300 selected mu=20 finder-output tracks.

For each candidate:
  left:  r-phi display with detector-layer radii shown as horizontal guides
  right: 3D x-y-z candidate trajectory

Expected inputs (when run from TrackProject/src):
  ../tracks_for_ed/mu20_finder_SM_100
  ../tracks_for_ed/mu20_finder_PID15_100
  ../tracks_for_ed/mu20_finder_fakes_100

Outputs:
  ../pngs/mu20_finder_event_displays/
      SM_helical/
      PID15_nonhelical/
      background_fake/

Run:
    python plot_mu20_finder_300.py

Quick test:
    python plot_mu20_finder_300.py --max-per-class 2
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIRS = {
    "SM_helical": "../tracks_for_ed/mu20_finder_SM_100",
    "PID15_nonhelical": "../tracks_for_ed/mu20_finder_PID15_100",
    "background_fake": "../tracks_for_ed/mu20_finder_fakes_100",
}

DEFAULT_OUTPUT_DIR = "../pngs/mu20_finder_event_displays"


def load_candidate(path):
    df = pd.read_csv(path)

    required = {"r", "phi", "z", "layer_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path}: missing required columns {sorted(missing)}"
        )

    if "x" not in df.columns:
        df["x"] = df["r"] * np.cos(df["phi"])
    if "y" not in df.columns:
        df["y"] = df["r"] * np.sin(df["phi"])

    if "hit_order" in df.columns:
        df = df.sort_values("hit_order", kind="stable").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def infer_layer_radii(all_files):
    """
    Infer one representative radius per physical detector layer from all 300 files.
    """
    pieces = []

    for path in all_files:
        df = pd.read_csv(
            path,
            usecols=lambda c: c in {"layer_id", "r"},
        )
        if len(df):
            pieces.append(df[["layer_id", "r"]])

    if not pieces:
        return {}

    x = pd.concat(pieces, ignore_index=True)

    med = (
        x.groupby("layer_id", sort=True)["r"]
        .median()
        .sort_index()
    )

    return med.to_dict()


def candidate_title(df, class_name, filename):
    event = (
        str(df["event_stem"].iloc[0])
        if "event_stem" in df.columns
        else os.path.basename(filename).replace("-hits.csv", "")
    )

    candidate_idx = (
        int(df["candidate_idx"].iloc[0])
        if "candidate_idx" in df.columns
        else None
    )

    parts = [
        class_name,
        event,
        f"{len(df)} hits",
    ]

    if candidate_idx is not None:
        parts.append(f"candidate {candidate_idx}")

    if "n_truth_contributors" in df.columns:
        parts.append(
            f"{int(df['n_truth_contributors'].iloc[0])} truth contributors"
        )

    return " | ".join(parts)


def plot_candidate(
    df,
    class_name,
    source_path,
    output_path,
    layer_radii,
):
    phi = df["phi"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    # Unwrap only to keep a trajectory that crosses +/-pi visually continuous.
    phi_unwrapped = np.unwrap(phi)
    if len(phi):
        shift = 2 * np.pi * np.round(
            (np.mean(phi_unwrapped) - np.mean(phi)) / (2 * np.pi)
        )
        phi_plot = phi_unwrapped - shift
    else:
        phi_plot = phi

    fig = plt.figure(figsize=(14, 6))

    ax_rphi = fig.add_subplot(1, 2, 1)
    ax_xyz = fig.add_subplot(1, 2, 2, projection="3d")

    # ------------------------------------------------------------
    # Left: r-phi / 25-layer view
    # ------------------------------------------------------------
    for _, radius in sorted(
        layer_radii.items(),
        key=lambda kv: kv[1],
    ):
        ax_rphi.axhline(
            radius,
            linewidth=0.5,
            alpha=0.22,
        )

    ax_rphi.plot(
        phi_plot,
        r,
        linewidth=1.2,
        alpha=0.8,
    )

    ax_rphi.scatter(
        phi_plot,
        r,
        s=28,
        zorder=3,
    )

    # Annotate ordered hit number so mergers/jumps are easy to inspect.
    for i, (ph, rr) in enumerate(zip(phi_plot, r)):
        ax_rphi.annotate(
            str(i),
            (ph, rr),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )

    ax_rphi.set_xlabel(r"$\phi$")
    ax_rphi.set_ylabel(r"$r$ [cm]")
    ax_rphi.set_title(r"$r$-$\phi$ detector-layer view")
    ax_rphi.grid(alpha=0.15)

    # ------------------------------------------------------------
    # Right: x-y-z view
    # ------------------------------------------------------------
    ax_xyz.plot(
        x,
        y,
        z,
        linewidth=1.2,
        alpha=0.8,
    )

    ax_xyz.scatter(
        x,
        y,
        z,
        s=28,
    )

    if len(df):
        ax_xyz.scatter(
            [x[0]],
            [y[0]],
            [z[0]],
            s=65,
            marker="o",
            label="first hit",
        )
        ax_xyz.scatter(
            [x[-1]],
            [y[-1]],
            [z[-1]],
            s=65,
            marker="x",
            label="last hit",
        )

    ax_xyz.set_xlabel("x [cm]")
    ax_xyz.set_ylabel("y [cm]")
    ax_xyz.set_zlabel("z [cm]")
    ax_xyz.set_title("3D candidate trajectory")
    ax_xyz.legend(fontsize=8)

    fig.suptitle(
        candidate_title(df, class_name, source_path),
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional quick-test limit per class.",
    )
    args = parser.parse_args()

    files_by_class = {}

    for class_name, folder in DEFAULT_INPUT_DIRS.items():
        files = sorted(
            glob.glob(os.path.join(folder, "*-hits.csv"))
        )

        if len(files) != 100:
            raise RuntimeError(
                f"{class_name}: expected 100 *-hits.csv files in "
                f"{folder}, found {len(files)}"
            )

        if args.max_per_class is not None:
            files = files[:args.max_per_class]

        files_by_class[class_name] = files

    # Infer detector layer radii from the full 300-file sample even for quick tests.
    all_files_full = []
    for folder in DEFAULT_INPUT_DIRS.values():
        all_files_full.extend(
            sorted(glob.glob(os.path.join(folder, "*-hits.csv")))
        )

    layer_radii = infer_layer_radii(all_files_full)

    print(f"Inferred {len(layer_radii)} detector layers")
    if len(layer_radii) != 25:
        print(
            "WARNING: expected 25 layer IDs across the sample, "
            f"but inferred {len(layer_radii)}"
        )

    total_written = 0

    for class_name, files in files_by_class.items():
        class_out = os.path.join(
            args.output_dir,
            class_name,
        )
        os.makedirs(class_out, exist_ok=True)

        print(f"\n{class_name}: {len(files)} candidates")

        for i, source_path in enumerate(files, start=1):
            df = load_candidate(source_path)

            event_stem = (
                str(df["event_stem"].iloc[0])
                if "event_stem" in df.columns
                else os.path.basename(source_path).replace("-hits.csv", "")
            )

            candidate_idx = (
                int(df["candidate_idx"].iloc[0])
                if "candidate_idx" in df.columns
                else 0
            )

            out_name = (
                f"{i:03d}_{event_stem}_candidate{candidate_idx}.png"
            )
            output_path = os.path.join(
                class_out,
                out_name,
            )

            plot_candidate(
                df,
                class_name,
                source_path,
                output_path,
                layer_radii,
            )

            total_written += 1

            if i == 1 or i % 10 == 0 or i == len(files):
                print(
                    f"  {i:3d}/{len(files)} -> {output_path}"
                )

    print(
        f"\nSUCCESS: wrote {total_written} plots to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
