import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PURITY_CUT = 0.6
MIN_HITS = 20

PATH = Path(
    f"~/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/"
    f"Levi_dominant_purity_lt_{str(PURITY_CUT).replace('.', 'p')}_minhits_{MIN_HITS}/"
).expanduser()
PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("~/Downloads/Train_SM+Schwartz_Set_19_Test_SM+Schwartz_Set_10.csv")

# Compute purity directly just to be safe
df["purity_calc"] = df["n_shared"] / df["n_reco_hits"]

# ------------------------------------------------------------
# Geometry / occupancy audit from hit-level finder output
# ------------------------------------------------------------

print("\n=== Hit-level geometry / occupancy audit ===")

# Basic event-level occupancy
event_occ = (
    df.groupby("event_id")
      .agg(
          n_hit_rows=("hit_id", "size"),
          n_unique_hits=("hit_id", "nunique"),
          n_reco_tracks=("track_id", "nunique"),
          n_unique_particles=("particle_id", "nunique"),
      )
      .reset_index()
)

print("\nEvent-level occupancy:")
print(event_occ.describe())

event_occ.to_csv("levi_event_level_occupancy.csv", index=False)

# Hit sharing: how often the same hit_id is used by multiple reco tracks in the same event
hit_use = (
    df.groupby(["event_id", "hit_id"])
      .agg(
          n_reco_tracks_using_hit=("track_id", "nunique"),
          n_rows=("track_id", "size"),
          r=("r", "first"),
          z=("z", "first"),
          phi=("phi", "first"),
      )
      .reset_index()
)

shared_hits = hit_use[hit_use["n_reco_tracks_using_hit"] > 1].copy()

print("\nHit sharing:")
print("total event-hit entries:", len(hit_use))
print("shared event-hit entries:", len(shared_hits))
print("fraction shared:", len(shared_hits) / len(hit_use) if len(hit_use) else np.nan)
print(hit_use["n_reco_tracks_using_hit"].describe())

hit_use.to_csv("levi_hit_sharing_by_event_hit.csv", index=False)

# Approximate radial layers.
# This assumes repeated/near-repeated r values correspond to detector layers.
# Rounding tolerance can be adjusted.
df["r_layer_round_0p1"] = df["r"].round(1)
df["r_layer_round_1p0"] = df["r"].round(0)

layer_occ = (
    df.groupby("r_layer_round_1p0")
      .agg(
          n_hit_rows=("hit_id", "size"),
          n_unique_hits=("hit_id", "nunique"),
          n_events=("event_id", "nunique"),
          median_abs_z=("z", lambda x: np.median(np.abs(x))),
          min_r=("r", "min"),
          max_r=("r", "max"),
      )
      .reset_index()
      .sort_values("r_layer_round_1p0")
)

print("\nApproximate radial-layer occupancy, r rounded to 1.0:")
print(layer_occ.to_string(index=False))

layer_occ.to_csv("levi_radial_layer_occupancy.csv", index=False)

# Per-event per-layer occupancy: this is closest to "occupancy per detector layer"
event_layer_occ = (
    df.groupby(["event_id", "r_layer_round_1p0"])
      .agg(
          n_hit_rows=("hit_id", "size"),
          n_unique_hits=("hit_id", "nunique"),
          n_reco_tracks=("track_id", "nunique"),
      )
      .reset_index()
)

event_layer_summary = (
    event_layer_occ.groupby("r_layer_round_1p0")
      .agg(
          median_hit_rows_per_event=("n_hit_rows", "median"),
          mean_hit_rows_per_event=("n_hit_rows", "mean"),
          q90_hit_rows_per_event=("n_hit_rows", lambda x: x.quantile(0.9)),
          max_hit_rows_per_event=("n_hit_rows", "max"),
          median_unique_hits_per_event=("n_unique_hits", "median"),
          mean_unique_hits_per_event=("n_unique_hits", "mean"),
      )
      .reset_index()
      .sort_values("r_layer_round_1p0")
)

print("\nPer-event radial-layer occupancy summary:")
print(event_layer_summary.to_string(index=False))

event_layer_occ.to_csv("levi_event_layer_occupancy.csv", index=False)
event_layer_summary.to_csv("levi_event_layer_occupancy_summary.csv", index=False)

# Eta/phi occupancy
eta_phi_summary = df[["eta", "phi", "r", "z"]].describe()
print("\nEta/phi/r/z summary:")
print(eta_phi_summary)

eta_phi_summary.to_csv("levi_eta_phi_rz_summary.csv")

# Optional binned eta-phi map
eta_bins = np.linspace(df["eta"].quantile(0.001), df["eta"].quantile(0.999), 41)
phi_bins = np.linspace(-np.pi, np.pi, 65)

eta_phi_counts, eta_edges, phi_edges = np.histogram2d(
    df["eta"].to_numpy(float),
    df["phi"].to_numpy(float),
    bins=[eta_bins, phi_bins],
)

eta_phi_df = pd.DataFrame(eta_phi_counts)
eta_phi_df.to_csv("levi_eta_phi_occupancy_hist2d.csv", index=False)

# Track-level geometric extent
track_geom = (
    df.groupby(["event_id", "track_id"])
      .agg(
          n_hit_rows=("hit_id", "size"),
          n_unique_hits=("hit_id", "nunique"),
          r_min=("r", "min"),
          r_max=("r", "max"),
          z_min=("z", "min"),
          z_max=("z", "max"),
          phi_min=("phi", "min"),
          phi_max=("phi", "max"),
          eta_min=("eta", "min"),
          eta_max=("eta", "max"),
          PID=("PID", "first"),
          purity_reco=("purity_reco", "max"),
          eff_true=("eff_true", "max"),
      )
      .reset_index()
)

track_geom["delta_r"] = track_geom["r_max"] - track_geom["r_min"]
track_geom["delta_z"] = track_geom["z_max"] - track_geom["z_min"]
track_geom["delta_eta"] = track_geom["eta_max"] - track_geom["eta_min"]
track_geom["is_PID15"] = track_geom["PID"] == 15

print("\nTrack-level geometric extent:")
print(track_geom[[
    "n_hit_rows", "r_min", "r_max", "delta_r",
    "z_min", "z_max", "delta_z",
    "eta_min", "eta_max", "delta_eta",
]].describe())

print("\nTrack-level geometric extent split by PID15:")
print(
    track_geom.groupby("is_PID15")[[
        "n_hit_rows", "r_min", "r_max", "delta_r", "delta_z", "delta_eta"
    ]].median()
)

track_geom.to_csv("levi_track_geometric_extent.csv", index=False)

# chi_square_rz: summarize, but do not call this noise unless defined by Levi.
if "chi_square_rz" in df.columns:
    chi_rz_track = (
        df.groupby(["event_id", "track_id"])
          .agg(
              chi_square_rz=("chi_square_rz", "first"),
              PID=("PID", "first"),
              purity_reco=("purity_reco", "max"),
              eff_true=("eff_true", "max"),
              n_hit_rows=("hit_id", "size"),
          )
          .reset_index()
    )

    print("\nchi_square_rz track-level summary:")
    print(chi_rz_track["chi_square_rz"].describe())

    print("\nchi_square_rz split by PID15:")
    print(chi_rz_track.groupby(chi_rz_track["PID"] == 15)["chi_square_rz"].describe())

    chi_rz_track.to_csv("levi_chi_square_rz_track_summary.csv", index=False)

for decimals in [3, 2, 1]:
    col = f"r_round_{decimals}"
    df[col] = df["r"].round(decimals)

    occ = (
        df.groupby(col)
          .agg(
              n_hit_rows=("hit_id", "size"),
              n_events=("event_id", "nunique"),
              median_abs_z=("z", lambda x: np.median(np.abs(x))),
              min_r=("r", "min"),
              max_r=("r", "max"),
          )
          .reset_index()
          .sort_values(col)
    )

    print(f"\nRadial occupancy with r rounded to {decimals} decimals:")
    print(occ.head(50).to_string(index=False))
    print("num radial bins:", len(occ))

print("\nMost populated rounded-r bins:")
print(
    df.groupby(df["r"].round(3))
      .size()
      .sort_values(ascending=False)
      .head(30)
)

# For each reconstructed candidate, choose the row with the largest n_shared.
# This should correspond to the dominant truth/generated contributor.
idx = (
    df.groupby(["event_id", "track_id"])["n_shared"]
      .idxmax()
)

tracks = df.loc[idx].copy()

# Add actual number of rows/hits present in the file for each reco candidate
n_rows = (
    df.groupby(["event_id", "track_id"])["hit_id"]
      .size()
      .rename("n_rows")
      .reset_index()
)

tracks = tracks.merge(n_rows, on=["event_id", "track_id"], how="left")

# Number of distinct truth contributors associated with each reco candidate
n_truth_contrib = (
    df.groupby(["event_id", "track_id"])["particle_id"]
      .nunique()
      .rename("n_truth_contributors")
      .reset_index()
)

tracks = tracks.merge(n_truth_contrib, on=["event_id", "track_id"], how="left")

# Second-largest contributor, if present
contrib = (
    df.assign(shared_frac=df["n_shared"] / df["n_reco_hits"])
      .sort_values(["event_id", "track_id", "n_shared"], ascending=[True, True, False])
      .groupby(["event_id", "track_id"])
)

second = (
    contrib.nth(1)
      .reset_index()[["event_id", "track_id", "n_shared"]]
      .rename(columns={"n_shared": "n_shared_second"})
)

tracks = tracks.merge(second, on=["event_id", "track_id"], how="left")
tracks["n_shared_second"] = tracks["n_shared_second"].fillna(0)

tracks["second_frac"] = tracks["n_shared_second"] / tracks["n_reco_hits"]
tracks["extra_hit_frac"] = np.maximum(0., 1.0 - tracks["n_shared"] / tracks["n_rows"])

# How many reco candidates share the same dominant generated particle in an event?
dominant_particle_reco_count = (
    tracks.groupby(["event_id", "particle_id"])
          .size()
          .rename("n_reco_for_same_dominant_particle")
          .reset_index()
)

tracks = tracks.merge(
    dominant_particle_reco_count,
    on=["event_id", "particle_id"],
    how="left",
)

tracks["duplicate_like"] = tracks["n_reco_for_same_dominant_particle"] > 1

for cut_name, mask in {
    "purity < 0.9, n_rows >= 20": (
        (tracks["purity_reco"] < 0.9) &
        (tracks["n_rows"] >= 20)
    ),
    "purity < 0.6, n_rows >= 20": (
        (tracks["purity_reco"] < 0.6) &
        (tracks["n_rows"] >= 20)
    ),
    "0.5 <= purity < 0.6, n_rows >= 20": (
        (tracks["purity_reco"] >= 0.5) &
        (tracks["purity_reco"] < 0.6) &
        (tracks["n_rows"] >= 20)
    ),
    "real-track-plus-extra-hits, n_rows >= 20": (
        (tracks["n_reco_hits"] > tracks["n_shared"]) &
        (tracks["n_rows"] >= 20)
    ),
}.items():
    sub = tracks.loc[mask].copy()

    print(f"\n=== {cut_name} ===")
    print("n candidates:", len(sub))
    print("n events:", sub["event_id"].nunique())

    print("\nDominant PID counts:")
    print(sub["PID"].value_counts(dropna=False).sort_index())

    print("\nDominant PID fractions:")
    print(sub["PID"].value_counts(normalize=True, dropna=False).sort_index())

    print("\nWeird-dominant vs SM-dominant:")
    print((sub["PID"] == 15).value_counts(dropna=False))
    print((sub["PID"] == 15).value_counts(normalize=True, dropna=False))

    print("\nSummary:")
    print(sub[["purity_reco", "eff_true", "n_rows", "n_shared", "n_reco_hits", "n_true_hits"]].describe())

# Clean track-level table
tracks = tracks[[
    "event_id",
    "track_id",
    "particle_id",
    "n_shared",
    "n_reco_hits",
    "n_true_hits",
    "purity_reco",
    "purity_calc",
    "eff_true",
    "PID",
    "pt",
    "n_rows",

    # fake-model audit diagnostics
    "n_truth_contributors",
    "n_shared_second",
    "second_frac",
    "extra_hit_frac",
    "n_reco_for_same_dominant_particle",
    "duplicate_like",
]].copy()

def write_track_sample(df_hits, selected_tracks, out_path, filename_prefix, metadata_filename):
    """
    Write one CSV per selected reconstructed track.

    selected_tracks should be a track-level dataframe with event_id and track_id.
    df_hits is the original hit-level dataframe.
    """
    out_path = Path(out_path).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    for old in out_path.glob(f"{filename_prefix}_event*_track*-hits.csv"):
        old.unlink()

    selected_tracks.to_csv(out_path / metadata_filename, index=False)

    selected_keys = selected_tracks[["event_id", "track_id"]]

    selected_hits = df_hits.merge(
        selected_keys,
        on=["event_id", "track_id"],
        how="inner"
    )

    selected_hits = selected_hits.sort_values(["event_id", "track_id", "r"])

    sigma_xyz = 0.01
    eps = 1e-12

    for (event_id, track_id), track_df in selected_hits.groupby(["event_id", "track_id"]):
        track_df = track_df.sort_values("r").copy()

        track_df["sigma_r"] = sigma_xyz
        track_df["sigma_phi"] = sigma_xyz / np.maximum(track_df["r"].to_numpy(float), eps)
        track_df["sigma_z"] = sigma_xyz

        filename = f"{filename_prefix}_event{int(event_id)}_track{int(track_id)}-hits.csv"
        track_df.to_csv(out_path / filename, index=False)

    print(f"\nSaved {selected_hits.groupby(['event_id', 'track_id']).ngroups} tracks to:")
    print(out_path)

    return selected_hits

def make_glued_event_display(df_hits, event_id, track_id, out_dir, *, show=False):
    """
    Event display for one reconstructed candidate.
    Points are colored by truth particle_id.

    Makes:
      1. 3D x-y-z display
      2. x-y display
      3. r-z display
      4. phi-z display
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    h = df_hits[
        (df_hits["event_id"] == event_id) &
        (df_hits["track_id"] == track_id)
    ].copy()

    if len(h) == 0:
        print(f"[event display] no hits for event={event_id}, track={track_id}")
        return

    h = h.sort_values("r").copy()
    pid_summary = (
        h.groupby(["particle_id", "PID"])
         .size()
         .reset_index(name="n")
         .sort_values("n", ascending=False)
    )

    pid_title = "; ".join(
        f"pid={int(row.PID)}, particle={int(row.particle_id)}, n={int(row.n)}"
        for row in pid_summary.itertuples(index=False)
    )

    # Use stored x,y if present; otherwise reconstruct from r,phi.
    if ("x" in h.columns) and ("y" in h.columns):
        h["x_plot"] = h["x"].astype(float)
        h["y_plot"] = h["y"].astype(float)
    else:
        h["x_plot"] = h["r"].astype(float) * np.cos(h["phi"].astype(float))
        h["y_plot"] = h["r"].astype(float) * np.sin(h["phi"].astype(float))

    h["z_plot"] = h["z"].astype(float)

    # Approximate detector layers from the observed radii.
    # This avoids depending on make_tracks_60.py's ATLASradii globals.
    layer_radii = np.sort(df_hits["r"].round(2).unique())

    base = f"event{int(event_id)}_track{int(track_id)}"

    title = (
        f"{base}: purity={h['purity_reco'].iloc[0]:.3f}, "
        f"n_rows={len(h)}, "
        f"n_truth_contrib={h['particle_id'].nunique()}\n"
        f"{pid_title}"
    )

    # -------------------------
    # 3D display
    # -------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for pid, g in h.groupby("particle_id", sort=False):
        pid_val = g["PID"].iloc[0]
        ax.scatter(
            g["x_plot"], g["y_plot"], g["z_plot"],
            s=28,
            label=f"particle_id={pid}, PID={pid_val}, n={len(g)}",
        )

    # Draw a few outer detector cylinders, like make_track_plot.
    theta = np.linspace(0, 2 * np.pi, 80)
    zmin = float(h["z_plot"].min())
    zmax = float(h["z_plot"].max())
    z_range = np.linspace(zmin, zmax, 60)

    for radius in layer_radii[-4:]:
        Theta, Z = np.meshgrid(theta, z_range)
        X = radius * np.cos(Theta)
        Y = radius * np.sin(Theta)
        ax.plot_surface(X, Y, Z, alpha=0.08, rstride=8, cstride=8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base}_3d_by_particle.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # -------------------------
    # x-y display
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    for pid, g in h.groupby("particle_id", sort=False):
        ax.scatter(
            g["x_plot"], g["y_plot"],
            s=28,
            label=f"particle_id={pid}, n={len(g)}",
        )

    for radius in layer_radii:
        circ = plt.Circle((0, 0), radius, fill=False, alpha=0.08)
        ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title + " | x-y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base}_xy_by_particle.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # -------------------------
    # r-z display
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for pid, g in h.groupby("particle_id", sort=False):
        ax.scatter(
            g["r"], g["z"],
            s=28,
            label=f"particle_id={pid}, n={len(g)}",
        )

    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_title(title + " | r-z")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base}_rz_by_particle.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # -------------------------
    # phi-z display
    # -------------------------
    phi_col = "phi" if "phi" in h.columns else "phi_reco"

    fig, ax = plt.subplots(figsize=(8, 6))
    for pid, g in h.groupby("particle_id", sort=False):
        ax.scatter(
            g[phi_col], g["z"],
            s=28,
            label=f"particle_id={pid}, n={len(g)}",
        )

    ax.set_xlabel(phi_col)
    ax.set_ylabel("z")
    ax.set_title(title + f" | {phi_col}-z")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{base}_{phi_col}_z_by_particle.png", dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    # Save the actual hit rows too.
    h.to_csv(out_dir / f"{base}_hits.csv", index=False)

    print(f"[event display] saved {base} with {len(h)} hits to {out_dir}")

WEIRD_PID = 15

n_events = tracks["event_id"].nunique()

weird_tracks = tracks[tracks["PID"] == WEIRD_PID].copy()
sm_tracks = tracks[tracks["PID"] != WEIRD_PID].copy()

print("\n=== Inserted non-helical reconstruction summary ===")
print("total events:", n_events)
print("num reco candidates with dominant PID == 15:", len(weird_tracks))
print("num events with >=1 dominant PID == 15 candidate:", weird_tracks["event_id"].nunique())
print("event-level reconstruction fraction:", weird_tracks["event_id"].nunique() / n_events)

print("\nPID==15 candidate purity:")
print(weird_tracks["purity_reco"].describe())

print("\nPID==15 candidate efficiency:")
print(weird_tracks["eff_true"].describe())

print("\nPID==15 candidate hit counts:")
print(weird_tracks[["n_shared", "n_reco_hits", "n_true_hits", "n_rows"]].describe())

best_weird_per_event = (
    weird_tracks
    .sort_values(
        ["event_id", "n_shared", "purity_reco", "eff_true", "n_rows"],
        ascending=[True, False, False, False, False],
    )
    .groupby("event_id")
    .head(1)
    .copy()
)

print("\n=== Best PID==15 candidate per event ===")
print("events with reconstructed PID==15:", len(best_weird_per_event))
print("total events:", n_events)
print("fraction:", len(best_weird_per_event) / n_events)

print("\nBest PID==15 purity:")
print(best_weird_per_event["purity_reco"].describe())

print("\nBest PID==15 efficiency:")
print(best_weird_per_event["eff_true"].describe())

print("\nBest PID==15 hit counts:")
print(best_weird_per_event[["n_shared", "n_reco_hits", "n_true_hits", "n_rows"]].describe())

print("\n=== PID==15 candidates by quality cuts ===")

for purity_cut in [0.5, 0.7, 0.9, 0.95]:
    for eff_cut in [0.5, 0.7, 0.9]:
        sel = best_weird_per_event[
            (best_weird_per_event["purity_reco"] >= purity_cut) &
            (best_weird_per_event["eff_true"] >= eff_cut) &
            (best_weird_per_event["n_rows"] >= MIN_HITS)
        ]

        print(
            f"purity>={purity_cut}, eff>={eff_cut}, n_rows>={MIN_HITS}: "
            f"{len(sel)} events / {n_events} = {len(sel)/n_events:.4f}"
        )

WEIRD_PURITY_CUT = 0.9
WEIRD_EFF_CUT = 0.5
WEIRD_MIN_HITS = 20

weird_good = best_weird_per_event[
    (best_weird_per_event["purity_reco"] >= WEIRD_PURITY_CUT) &
    (best_weird_per_event["eff_true"] >= WEIRD_EFF_CUT) &
    (best_weird_per_event["n_rows"] >= WEIRD_MIN_HITS)
].copy()

print("\n=== Good reconstructed non-helical sample ===")
print("num good PID==15 candidates:", len(weird_good))
print(weird_good["purity_reco"].describe())
print(weird_good["eff_true"].describe())
print(weird_good[["n_rows", "n_reco_hits", "n_shared", "n_true_hits"]].describe())

# ------------------------------------------------------------
# Finder-output category accounting
# Truth quantities are used here only to label/evaluate
# what categories exist in the simulated finder output.
# ------------------------------------------------------------

tracks["is_weird"] = tracks["PID"] == WEIRD_PID
tracks["is_fragment"] = tracks["eff_true"] < 0.5
tracks["is_completeish"] = tracks["eff_true"] >= 0.5
tracks["is_pure"] = tracks["purity_reco"] >= 0.9
tracks["is_impure"] = tracks["purity_reco"] < 0.9
tracks["is_strict_fake"] = tracks["purity_reco"] < 0.5
tracks["is_near_fake"] = (tracks["purity_reco"] >= 0.5) & (tracks["purity_reco"] < 0.6)
tracks["has_extra_hits"] = tracks["n_reco_hits"] > tracks["n_shared"]
tracks["enough_hits"] = tracks["n_rows"] >= MIN_HITS

# ------------------------------------------------------------
# Exclusive reconstructed-track category accounting
# Use a minimal reconstructed-track hit requirement first.
# These bins are mutually exclusive.
# ------------------------------------------------------------

RECO_MIN_HITS = 20

tracks["is_reco_candidate"] = tracks["n_rows"] >= RECO_MIN_HITS

reco_tracks = tracks[tracks["is_reco_candidate"]].copy()
raw_fragments = tracks[~tracks["is_reco_candidate"]].copy()

print("\n=== Raw fragments / orphan-hit candidates ===")
print("n candidates:", len(raw_fragments))
print("n events:", raw_fragments["event_id"].nunique())
print(raw_fragments[["purity_reco", "eff_true", "n_rows", "n_shared", "n_reco_hits", "n_true_hits"]].describe())

# Exclusive bins after n_rows >= RECO_MIN_HITS
conditions = [
    (
        (reco_tracks["PID"] != WEIRD_PID) &
        (reco_tracks["purity_reco"] >= 0.9),
        "high-purity SM/helical"
    ),
    (
        (reco_tracks["PID"] == WEIRD_PID) &
        (reco_tracks["purity_reco"] >= 0.9) &
        (reco_tracks["eff_true"] >= 0.5),
        "high-purity PID15 non-helical"
    ),
    (
        (reco_tracks["purity_reco"] >= 0.5) &
        (reco_tracks["purity_reco"] < 0.9),
        "merged/contaminated: 0.5<=purity<0.9"
    ),
    (
        reco_tracks["purity_reco"] < 0.5,
        "traditional fake: purity<0.5"
    ),
]

reco_tracks["exclusive_category"] = "other / edge case"

already_assigned = pd.Series(False, index=reco_tracks.index)

for mask, name in conditions:
    assign = mask & (~already_assigned)
    reco_tracks.loc[assign, "exclusive_category"] = name
    already_assigned |= assign

# Useful non-exclusive tags
reco_tracks["tag_PID15_dominant"] = reco_tracks["PID"] == WEIRD_PID
reco_tracks["tag_multi_contributor"] = reco_tracks["n_truth_contributors"] >= 2
reco_tracks["tag_extra_hits"] = reco_tracks["n_reco_hits"] > reco_tracks["n_shared"]
reco_tracks["tag_near_threshold_merged"] = (
    (reco_tracks["purity_reco"] >= 0.5) &
    (reco_tracks["purity_reco"] < 0.6)
)

rows = []
important_rows = []
for name, sub in reco_tracks.groupby("exclusive_category"):
    rows.append({
        "exclusive_category": name,
        "n_candidates": len(sub),
        "n_events": sub["event_id"].nunique(),
        "median_purity": sub["purity_reco"].median(),
        "median_eff": sub["eff_true"].median(),
        "median_hits": sub["n_rows"].median(),
        "median_extra_hit_frac": sub["extra_hit_frac"].median(),
        "median_n_truth_contributors": sub["n_truth_contributors"].median(),
        "frac_PID15_dominant": sub["tag_PID15_dominant"].mean(),
        "frac_multi_contributor": sub["tag_multi_contributor"].mean(),
        "frac_extra_hits": sub["tag_extra_hits"].mean(),
        "frac_near_threshold_merged": sub["tag_near_threshold_merged"].mean(),
    })
    important_rows.append({
        "exclusive_category": name,
        "n_candidates": len(sub),
        "n_events": sub["event_id"].nunique(),
        "median_purity": sub["purity_reco"].median(),
        "median_eff": sub["eff_true"].median(),
        "median_hits": sub["n_rows"].median(),
        "median_n_truth_contributors": sub["n_truth_contributors"].median(),
    })

exclusive_summary = pd.DataFrame(rows).sort_values("exclusive_category")
important_exclusive_summary = pd.DataFrame(important_rows).sort_values("exclusive_category")
category_order = [
    r"high-purity SM/helical",
    r"high-purity PID15 non-helical",
    r"merged/contaminated: 0.5<=purity<0.9",
    r"traditional fake: purity<0.5",
    r"other / edge case",
]

exclusive_summary = (
    exclusive_summary
    .set_index("exclusive_category")
    .reindex(category_order)
    .reset_index()
)
important_exclusive_summary = (
    important_exclusive_summary
    .set_index("exclusive_category")
    .reindex(category_order)
    .reset_index()
)

exclusive_summary["n_candidates"] = exclusive_summary["n_candidates"].fillna(0).astype(int)
exclusive_summary["n_events"] = exclusive_summary["n_events"].fillna(0).astype(int)

important_exclusive_summary["n_candidates"] = important_exclusive_summary["n_candidates"].fillna(0).astype(int)
important_exclusive_summary["n_events"] = important_exclusive_summary["n_events"].fillna(0).astype(int)

print("\n=== Exclusive reconstructed-track categories, n_rows >= 20 ===")
print(exclusive_summary.to_string(index=False))
print("\nLaTeX:")
print(exclusive_summary.to_latex(index=False, escape=True))

print("\n=== Important reconstructed-track categories, n_rows >= 20 ===")
print(important_exclusive_summary.to_string(index=False))
print("\nLaTeX:")
print(important_exclusive_summary.to_latex(index=False, escape=True))

exclusive_summary.to_csv(
    "levi_finder_output_exclusive_reco_categories.csv",
    index=False,
)

reco_tracks.to_csv(
    "levi_finder_output_tracklevel_with_exclusive_categories.csv",
    index=False,
)

# Optional: inspect the interesting merged/glued cases Makayla pointed out
merged_near_threshold = reco_tracks[
    reco_tracks["tag_near_threshold_merged"]
].copy()

print("\n=== Near-threshold merged/contaminated candidates: 0.5<=purity<0.6, n_rows>=20 ===")
print("n candidates:", len(merged_near_threshold))
print("n events:", merged_near_threshold["event_id"].nunique())

print(
    merged_near_threshold[[
        "event_id", "track_id", "PID", "particle_id",
        "purity_reco", "eff_true", "n_rows",
        "n_shared", "n_reco_hits", "n_true_hits",
        "n_truth_contributors", "extra_hit_frac",
    ]]
    .sort_values(["n_rows", "purity_reco"], ascending=[False, True])
    .to_string(index=False)
)

merged_near_threshold.to_csv(
    "levi_near_threshold_merged_candidates_0p5_0p6.csv",
    index=False,
)

# ------------------------------------------------------------
# Levi-style closure accounting from paper definition
# Candidate: N_hits >= 15
# Reconstructible truth particle: N_true_hits >= 22
# Double-majority match:
#   reco purity > 0.5 and truth efficiency > 0.5
# Fake:
#   candidate with N_hits >= 15 and no double-majority matched truth particle
# ------------------------------------------------------------

LEVI_CANDIDATE_MIN_HITS = 15
LEVI_RECONSTRUCTIBLE_TRUE_HITS = 22

tracks["levi_candidate"] = tracks["n_rows"] >= LEVI_CANDIDATE_MIN_HITS
tracks["levi_truth_reconstructible"] = tracks["n_true_hits"] >= LEVI_RECONSTRUCTIBLE_TRUE_HITS

tracks["levi_double_majority_match"] = (
    tracks["levi_candidate"] &
    tracks["levi_truth_reconstructible"] &
    (tracks["purity_reco"] > 0.5) &
    (tracks["eff_true"] > 0.5)
)

# First define special non-fake failure modes
tracks["gnn_merged_artifact"] = (
    tracks["levi_candidate"] &
    (tracks["purity_reco"] >= 0.5) &
    (tracks["purity_reco"] < 0.9) &
    (tracks["n_truth_contributors"] >= 2)
)

tracks["partial_PID15"] = (
    tracks["levi_candidate"] &
    (tracks["PID"] == WEIRD_PID) &
    (tracks["purity_reco"] >= 0.9) &
    (tracks["eff_true"] < 0.5)
)

# Raw failure of double-majority
tracks["fails_double_majority"] = (
    tracks["levi_candidate"] &
    (~tracks["levi_double_majority_match"])
)

# Fake-like only after removing the categories Makayla said not to count as fakes
tracks["levi_style_fake_cleaned"] = (
    tracks["fails_double_majority"] &
    (~tracks["partial_PID15"]) &
    (~tracks["gnn_merged_artifact"])
)

tracks["levi_category"] = "below 15-hit candidate threshold"

tracks.loc[
    tracks["levi_double_majority_match"] & (tracks["PID"] != WEIRD_PID),
    "levi_category"
] = "double-majority matched SM/helical"

tracks.loc[
    tracks["levi_double_majority_match"] & (tracks["PID"] == WEIRD_PID),
    "levi_category"
] = "double-majority matched PID15 non-helical"

tracks.loc[
    tracks["partial_PID15"],
    "levi_category"
] = "partial PID15 non-helical"

tracks.loc[
    tracks["gnn_merged_artifact"],
    "levi_category"
] = "GNN merged/glued artifact"

tracks.loc[
    tracks["levi_style_fake_cleaned"],
    "levi_category"
] = "Levi-style fake after removing partial PID15/merged"
# GNN-specific artifacts / special categories: report separately, do not count as standard fakes

levi_rows = []

for name, sub in tracks.groupby("levi_category"):
    levi_rows.append({
        "levi_category": name,
        "n_candidates": len(sub),
        "n_events": sub["event_id"].nunique(),
        "median_purity": sub["purity_reco"].median(),
        "median_eff": sub["eff_true"].median(),
        "median_hits": sub["n_rows"].median(),
        "median_n_true_hits": sub["n_true_hits"].median(),
        "median_n_truth_contributors": sub["n_truth_contributors"].median(),
        "frac_PID15_dominant": (sub["PID"] == WEIRD_PID).mean(),
        "frac_PID15_dominant": (sub["PID"] == WEIRD_PID).mean(),
        "median_n_truth_contributors": sub["n_truth_contributors"].median(),
    })

levi_summary = pd.DataFrame(levi_rows)

levi_category_order = [
    "double-majority matched SM/helical",
    "double-majority matched PID15 non-helical",
    "partial PID15 non-helical",
    "GNN merged/glued artifact",
    "Levi-style fake after removing partial PID15/merged",
    "below 15-hit candidate threshold",
]

levi_summary = (
    levi_summary
    .set_index("levi_category")
    .reindex(levi_category_order)
    .reset_index()
)

levi_summary["n_candidates"] = levi_summary["n_candidates"].fillna(0).astype(int)
levi_summary["n_events"] = levi_summary["n_events"].fillna(0).astype(int)

print("\n=== Levi-style double-majority closure table ===")
print(levi_summary.to_string(index=False))

print("\nLaTeX:")
print(levi_summary.to_latex(index=False, escape=True))

levi_summary.to_csv(
    "levi_style_double_majority_closure_table.csv",
    index=False,
)

tracks.to_csv(
    "levi_tracklevel_with_levi_style_labels.csv",
    index=False,
)

print("\n=== GNN merged artifact tag, Levi candidate hits >= 15 ===")
gnn_merged = tracks[tracks["gnn_merged_artifact"]].copy()
print("n candidates:", len(gnn_merged))
print("n events:", gnn_merged["event_id"].nunique())
print(gnn_merged[[
    "purity_reco", "eff_true", "n_rows", "n_shared",
    "n_reco_hits", "n_true_hits", "n_truth_contributors"
]].describe())

print("\n=== Partial PID15 tag, Levi candidate hits >= 15 ===")
partial_pid15 = tracks[tracks["partial_PID15"]].copy()
print("n candidates:", len(partial_pid15))
print("n events:", partial_pid15["event_id"].nunique())
print(partial_pid15[[
    "purity_reco", "eff_true", "n_rows", "n_shared",
    "n_reco_hits", "n_true_hits", "n_truth_contributors"
]].describe())

# ------------------------------------------------------------
# Truth-contributor PID composition for merged/glued candidates
# Answers: do these merged candidates include a weird PID15 track?
# ------------------------------------------------------------

merged_keys = merged_near_threshold[["event_id", "track_id"]].copy()

merged_hit_rows = df.merge(
    merged_keys,
    on=["event_id", "track_id"],
    how="inner",
)

# One row per truth contributor inside each reconstructed candidate
merged_contributors = (
    merged_hit_rows
    .groupby(["event_id", "track_id", "particle_id", "PID"])
    .agg(
        n_hits_from_contributor=("hit_id", "size"),
        n_shared_value=("n_shared", "max"),
        n_reco_hits=("n_reco_hits", "max"),
        purity_reco=("purity_reco", "max"),
        eff_true=("eff_true", "max"),
    )
    .reset_index()
    .sort_values(
        ["event_id", "track_id", "n_hits_from_contributor"],
        ascending=[True, True, False],
    )
)

merged_contributors.to_csv(
    "levi_near_threshold_merged_truth_contributors_0p5_0p6.csv",
    index=False,
)

# Candidate-level summary: which reco candidates include PID15 anywhere?
merged_pid_summary = (
    merged_contributors
    .groupby(["event_id", "track_id"])
    .agg(
        n_truth_contributors=("particle_id", "nunique"),
        contributor_PIDs=("PID", lambda x: tuple(sorted(pd.unique(x)))),
        contributor_particle_ids=("particle_id", lambda x: tuple(pd.unique(x))),
        contributor_hit_counts=("n_hits_from_contributor", lambda x: tuple(x)),
        includes_PID15=("PID", lambda x: (x == WEIRD_PID).any()),
        n_PID15_contributors=("PID", lambda x: int((x == WEIRD_PID).sum())),
    )
    .reset_index()
)

merged_pid_summary = merged_pid_summary.merge(
    merged_near_threshold[[
        "event_id", "track_id", "PID", "particle_id",
        "purity_reco", "eff_true", "n_rows",
        "n_shared", "n_reco_hits", "n_true_hits",
        "extra_hit_frac",
    ]].rename(columns={
        "PID": "dominant_PID",
        "particle_id": "dominant_particle_id",
    }),
    on=["event_id", "track_id"],
    how="left",
)

print("\n=== PID composition of near-threshold merged/contaminated candidates ===")
print("num candidates:", len(merged_pid_summary))
print("num including PID15 anywhere:", merged_pid_summary["includes_PID15"].sum())
print("fraction including PID15 anywhere:", merged_pid_summary["includes_PID15"].mean())

print("\nContributor PID tuple counts:")
print(
    merged_pid_summary["contributor_PIDs"]
    .value_counts(dropna=False)
    .head(30)
)

print("\nDominant PID counts:")
print(
    merged_pid_summary["dominant_PID"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nExamples including PID15:")
print(
    merged_pid_summary[merged_pid_summary["includes_PID15"]]
    .sort_values(["n_rows", "purity_reco"], ascending=[False, True])
    .head(20)
    .to_string(index=False)
)

print("\nExamples not including PID15:")
print(
    merged_pid_summary[~merged_pid_summary["includes_PID15"]]
    .sort_values(["n_rows", "purity_reco"], ascending=[False, True])
    .head(20)
    .to_string(index=False)
)

merged_pid_summary.to_csv(
    "levi_near_threshold_merged_pid_summary_0p5_0p6.csv",
    index=False,
)

# ------------------------------------------------------------
# Event displays for the clearest glued/merged candidates
# ------------------------------------------------------------

DISPLAY_DIR = Path("levi_glued_track_event_displays")
DISPLAY_DIR.mkdir(parents=True, exist_ok=True)

display_candidates = (
    merged_near_threshold
    .sort_values(
        ["n_rows", "n_truth_contributors", "purity_reco"],
        ascending=[False, False, True],
    )
    .head(5)
)

print("\n=== Making event displays for glued/merged candidates ===")
print(display_candidates[[
    "event_id", "track_id", "PID", "particle_id",
    "purity_reco", "eff_true", "n_rows",
    "n_shared", "n_reco_hits", "n_true_hits",
    "n_truth_contributors", "extra_hit_frac",
]].to_string(index=False))

for _, row in display_candidates.iterrows():
    make_glued_event_display(
        df,
        event_id=row["event_id"],
        track_id=row["track_id"],
        out_dir=DISPLAY_DIR,
        show=False,
    )

# Also make event displays for the rare merged candidates that include PID15
pid15_display_candidates = (
    merged_pid_summary[merged_pid_summary["includes_PID15"]]
    .sort_values(
        ["n_rows", "purity_reco"],
        ascending=[False, True],
    )
)

print("\n=== Making event displays for merged candidates that include PID15 ===")
print(pid15_display_candidates[[
    "event_id", "track_id",
    "contributor_PIDs", "contributor_particle_ids", "contributor_hit_counts",
    "dominant_PID", "dominant_particle_id",
    "purity_reco", "eff_true", "n_rows",
    "n_shared", "n_reco_hits", "n_true_hits",
    "extra_hit_frac",
]].to_string(index=False))

for _, row in pid15_display_candidates.iterrows():
    make_glued_event_display(
        df,
        event_id=row["event_id"],
        track_id=row["track_id"],
        out_dir=DISPLAY_DIR / "includes_PID15",
        show=False,
    )

WEIRD_PATH = Path(
    "~/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/"
    "Levi_reco_nonhelical_PID15_purity_ge_0p9_eff_ge_0p5_minhits_20/"
).expanduser()

WEIRD_PATH.mkdir(parents=True, exist_ok=True)

for old in WEIRD_PATH.glob("reco_nonhelical_event*_track*-hits.csv"):
    old.unlink()

weird_good.to_csv(WEIRD_PATH / "selected_reco_nonhelical_tracklevel.csv", index=False)

selected_weird_keys = weird_good[["event_id", "track_id"]]

weird_hits = df.merge(
    selected_weird_keys,
    on=["event_id", "track_id"],
    how="inner"
)

weird_hits = weird_hits.sort_values(["event_id", "track_id", "r"])

sigma_xyz = 0.01
eps = 1e-12

for (event_id, track_id), track_df in weird_hits.groupby(["event_id", "track_id"]):
    track_df = track_df.sort_values("r").copy()

    track_df["sigma_r"] = sigma_xyz
    track_df["sigma_phi"] = sigma_xyz / np.maximum(track_df["r"].to_numpy(float), eps)
    track_df["sigma_z"] = sigma_xyz

    filename = f"reco_nonhelical_event{int(event_id)}_track{int(track_id)}-hits.csv"
    track_df.to_csv(WEIRD_PATH / filename, index=False)

print(f"Saved {weird_hits.groupby(['event_id', 'track_id']).ngroups} reconstructed non-helical tracks to:")
print(WEIRD_PATH)

print(tracks[["event_id", "track_id", "n_shared", "n_reco_hits", "purity_reco", "purity_calc", "n_rows"]].head())
print(tracks["purity_reco"].describe())

print("num reco tracks:", len(tracks))
print("num true low-purity purity_reco < 0.5:", (tracks["purity_reco"] < 0.5).sum())
print("num impure purity_reco < 0.9:", (tracks["purity_reco"] < 0.9).sum())
print("num pure purity_reco == 1:", (tracks["purity_reco"] == 1.0).sum())

print("\nDominant-purity candidates by threshold and min hits:")
for purity_cut in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
    print(f"\npurity_reco < {purity_cut}:")
    for min_hits in [25, 20, 15, 10, 5]:
        sel = tracks[
            (tracks["purity_reco"] < purity_cut) &
            (tracks["n_reco_hits"] >= min_hits) &
            (tracks["n_rows"] >= min_hits)
        ]
        print(f"  min_hits >= {min_hits:2d}: {len(sel)}")

merged_candidates = tracks[
    (tracks["purity_reco"] >= 0.5) &
    (tracks["purity_reco"] < PURITY_CUT) &
    (tracks["n_reco_hits"] >= MIN_HITS) &
    (tracks["n_rows"] >= MIN_HITS)
].copy()

print("\nNear-threshold merged/contaminated sample:")
print(merged_candidates["purity_reco"].describe())
print("num candidates:", len(merged_candidates))

if len(merged_candidates) >= 100:
    fake_sample_100 = merged_candidates.sample(n=100, random_state=123)
else:
    fake_sample_100 = merged_candidates

print("sample size:", len(fake_sample_100))
print("sample max purity:", fake_sample_100["purity_reco"].max())

for old in PATH.glob("fake_reco_event*_track*-hits.csv"):
    old.unlink()

fake_sample_100.to_csv(PATH / "selected_fake_reco_tracks_tracklevel.csv", index=False)

selected_keys = fake_sample_100[["event_id", "track_id"]]

fake_hits_100 = df.merge(
    selected_keys,
    on=["event_id", "track_id"],
    how="inner"
)

fake_hits_100 = fake_hits_100.sort_values(["event_id", "track_id", "r"])

sigma_xyz = 0.01
eps = 1e-12

for (event_id, track_id), track_df in fake_hits_100.groupby(["event_id", "track_id"]):
    track_df = track_df.sort_values("r").copy()

    track_df["sigma_r"] = sigma_xyz
    track_df["sigma_phi"] = sigma_xyz / np.maximum(track_df["r"].to_numpy(float), eps)
    track_df["sigma_z"] = sigma_xyz

    filename = f"fake_reco_event{int(event_id)}_track{int(track_id)}-hits.csv"
    track_df.to_csv(PATH / filename, index=False)

print(f"Saved {fake_hits_100.groupby(['event_id', 'track_id']).ngroups} corrected track CSVs to:")
print(PATH)

rows = []

for f in sorted(PATH.glob("fake_reco_event*_track*-hits.csv")):
    h = pd.read_csv(f)
    rows.append({
        "file": f.name,
        "event_id": h["event_id"].iloc[0],
        "track_id": h["track_id"].iloc[0],
        "max_purity_reco": h["purity_reco"].max(),
        "max_n_shared": h["n_shared"].max(),
        "n_reco_hits": h["n_reco_hits"].iloc[0],
        "n_rows": len(h),
    })

file_meta = pd.DataFrame(rows)
try:
    print(file_meta["max_purity_reco"].describe())
    print("num files:", len(file_meta))
    print(f"num max purity >= {PURITY_CUT}:", (file_meta["max_purity_reco"] >= PURITY_CUT).sum())
    print(f"num max purity < {PURITY_CUT}:", (file_meta["max_purity_reco"] < PURITY_CUT).sum())
    print("num max purity < 0.5:", (file_meta["max_purity_reco"] < 0.5).sum())
except KeyError:
    print("max_purity_reco not contained in file_meta, continuing...")

SM_MIN_HITS = 20
SM_PURITY_CUT = 0.9
SM_SAMPLE_SIZE = 100  # use None to save all

sm_high_purity = tracks[
    (tracks["PID"] != WEIRD_PID) &
    (tracks["purity_reco"] >= SM_PURITY_CUT) &
    (tracks["n_rows"] >= SM_MIN_HITS)
].copy()

print("\n=== High-purity SM/helical sample ===")
print("num candidates:", len(sm_high_purity))
print("num events:", sm_high_purity["event_id"].nunique())
print(sm_high_purity["PID"].value_counts(dropna=False).sort_index())
print(sm_high_purity[["purity_reco", "eff_true", "n_rows", "n_shared", "n_reco_hits", "n_true_hits"]].describe())

if SM_SAMPLE_SIZE is not None and len(sm_high_purity) > SM_SAMPLE_SIZE:
    sm_high_purity_out = sm_high_purity.sample(n=SM_SAMPLE_SIZE, random_state=123)
else:
    sm_high_purity_out = sm_high_purity

SM_PATH = (
    "~/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/"
    "Levi_SM_high_purity_minhits_20/"
)

write_track_sample(
    df_hits=df,
    selected_tracks=sm_high_purity_out,
    out_path=SM_PATH,
    filename_prefix="reco_SM_highpurity",
    metadata_filename="selected_SM_high_purity_tracklevel.csv",
)

IMPURE09_CUT = 0.9
IMPURE09_MIN_HITS = 20
IMPURE09_SAMPLE_SIZE = 100  # use None to save all 160

impure_lt_0p9 = tracks[
    (tracks["purity_reco"] < IMPURE09_CUT) &
    (tracks["n_rows"] >= IMPURE09_MIN_HITS)
].copy()

print("\n=== Impure sample: purity < 0.9, n_rows >= 20 ===")
print("num candidates:", len(impure_lt_0p9))
print("num events:", impure_lt_0p9["event_id"].nunique())
print(impure_lt_0p9["PID"].value_counts(dropna=False).sort_index())
print(impure_lt_0p9[["purity_reco", "eff_true", "n_rows", "n_shared", "n_reco_hits", "n_true_hits"]].describe())

if IMPURE09_SAMPLE_SIZE is not None and len(impure_lt_0p9) > IMPURE09_SAMPLE_SIZE:
    impure_lt_0p9_out = impure_lt_0p9.sample(n=IMPURE09_SAMPLE_SIZE, random_state=123)
else:
    impure_lt_0p9_out = impure_lt_0p9

IMPURE09_PATH = (
    "~/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/"
    "Levi_impure_purity_lt_0p9_minhits_20/"
)

write_track_sample(
    df_hits=df,
    selected_tracks=impure_lt_0p9_out,
    out_path=IMPURE09_PATH,
    filename_prefix="reco_impure_purity_lt_0p9",
    metadata_filename="selected_impure_purity_lt_0p9_tracklevel.csv",
)
