import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
#  DETECTOR SPECIFICATION  (shared with the non-helical generator — do NOT
#  change these on one side only; the layer radii and hit resolution must be
#  identical for both classes or radius/noise alone becomes a class label).
# ============================================================================
min_r0  = 3.1
max_r0  = 53.0
nlayers = 25
sigma   = 0.01            # per-hit Gaussian resolution (cm), applied in x/y/z

DETECTOR_LENGTH = 320.0   # full length; |z| acceptance is +/- DETECTOR_LENGTH/2
DET_HALF_Z      = DETECTOR_LENGTH / 2.0

# ============================================================================
#  HIT-EXTRACTION OPTIONS  (the methodology change)
# ============================================================================
REACH_TOL      = 1e-9     # tolerance separating sub-ULP rounding from genuine no-crossing
HIT_EFFICIENCY = 1.0      # set < 1.0 to randomly drop hits (detector inefficiency).
                          # If you use this, apply the SAME value to the non-helical class.
MIN_HITS       = 1        # tracks reaching fewer than this many layers are skipped.
                          # Any stricter cut must also be applied to the non-helical class.

VALIDATION_MODE = True

if VALIDATION_MODE:
    OUTPUT_PATH = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/RyanDanielHelical/validation_10k.csv"
    VALIDATION_DIR = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/RyanDanielHelical/validation_plots"
    N_TRACKS = 10000
else:
    OUTPUT_PATH = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/RyanDanielHelical/10M_test.csv"
    VALIDATION_DIR = None
    N_TRACKS = 10000000

LAYER_RADII = np.linspace(min_r0, max_r0, nlayers)

# from: https://www-jlc.kek.jp/subg/offl/lib/docs/helix_manip/node3.html
# a track is parameterized by a 5-dim helix, expressed relative to an initial
# angle phi0 and swept out by varying phi with the other parameters fixed.
def track(phi, d0, phi0, pt, dz, tanl):
    alpha = 1 / 2.0  # constant: 1/(cB)
    q = 1
    kappa = q / pt
    rho = alpha / kappa                       # = pt/2 ; circle radius
    x = d0 * np.cos(phi0) + rho * (np.cos(phi0) - np.cos(phi0 + phi))
    y = d0 * np.sin(phi0) + rho * (np.sin(phi0) - np.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z


def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl):
    """
    phi at which the helix first reaches radius sqrt(target_radius_squared).

    Reachability is gated in radius-space upstream (in make_hits), so by the
    time we get here the arccos argument is guaranteed to lie in [-1, 1] up to
    floating-point rounding. The clip therefore only absorbs sub-ULP overshoot
    at a near-tangent crossing -- it no longer fabricates hits for layers the
    track never reaches (the old clamp's silent failure mode).
    """
    rho = 0.5 * pt
    cx = (d0 + rho) * np.cos(phi0)            # circle centre, distance (d0+rho) from origin
    cy = (d0 + rho) * np.sin(phi0)
    hyp = np.hypot(cx, cy)
    a = (hyp * hyp + rho * rho - target_radius_squared) / (2.0 * rho * hyp)
    a = np.clip(a, -1.0, 1.0)                 # rounding guard only
    return np.arccos(a) % (2 * np.pi)


def make_hits(params):
    """
    Return (layer_ids, xs, ys, zs) for the layers this track actually crosses.

    A layer is recorded only if:
      (1) the track physically reaches its radius:  d0 <= r0 <= d0 + 2*rho, and
      (2) the crossing falls within the detector's z acceptance, and
      (3) it survives the (optional) per-hit efficiency draw.
    Layers failing any of these are simply absent -> the track has a missing
    hit there, exactly like a non-helical track that doesn't reach that far.
    layer_id is written with every hit so the missing layers are unambiguous.
    """
    d0, phi0, pt, dz, tanl = params
    rho    = 0.5 * pt
    r_peri = d0                # closest approach  = |centre| - rho
    r_apo  = d0 + 2.0 * rho    # farthest reach    = |centre| + rho

    layer_ids, xs, ys, zs = [], [], [], []
    for layer_idx, r0 in enumerate(LAYER_RADII):
        # (1) radial reachability -- this is what allows missing (outer) hits
        if r0 < r_peri - REACH_TOL or r0 > r_apo + REACH_TOL:
            continue
        phi = find_phi_inverse(r0 * r0, *params)
        x0, y0, z0 = track(phi, *params)
        # (2) z acceptance (rarely triggers at tanl~0.6, matters if tanl widens)
        if abs(z0) > DET_HALF_Z:
            continue
        # (3) detector inefficiency -- keep identical to the non-helical class
        if HIT_EFFICIENCY < 1.0 and np.random.random() > HIT_EFFICIENCY:
            continue
        layer_ids.append(layer_idx + 1)       # 1-based, matches the h5 pipeline
        xs.append(x0 + np.random.normal(scale=sigma))
        ys.append(y0 + np.random.normal(scale=sigma))
        zs.append(z0 + np.random.normal(scale=sigma))
    return layer_ids, xs, ys, zs


def gen_tracks(n=1):
    tracks = []
    skipped = 0
    for i in range(n):
        if i % 10000 == 0:
            print("Track %d/%d" % (i, n), flush=True)
        # --- physicist-provided parameter ranges: unchanged ---
        d0   = np.fabs(np.random.normal(scale=0.03))
        phi  = np.random.uniform(low=0, high=2 * np.pi)
        pt   = np.random.lognormal(4, 0.5)
        dz   = np.random.normal(scale=0.5)
        tanl = np.random.normal(scale=0.6)
        params = (d0, phi, pt, dz, tanl)

        layer_ids, xs, ys, zs = make_hits(params)
        if len(layer_ids) < MIN_HITS:         # degenerate sub-detector track
            skipped += 1
            continue
        tracks.append([params, layer_ids, xs, ys, zs])
    if skipped:
        print("Skipped %d tracks with < %d hits" % (skipped, MIN_HITS), flush=True)
    return tracks

def summarize_tracks(tracks):
    """
    Build one row per generated track with simple detector-footprint diagnostics.
    """
    rows = []

    for idx, trk in enumerate(tracks):
        params, layer_ids, xs, ys, zs = trk
        d0, phi0, pt, dz, tanl = params

        layer_ids = np.asarray(layer_ids, dtype=int)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        zs = np.asarray(zs, dtype=float)

        if len(layer_ids) == 0:
            innermost_layer = np.nan
            outermost_layer = np.nan
            n_hits = 0
            frac_layers_hit = 0.0
        else:
            innermost_layer = int(np.min(layer_ids))
            outermost_layer = int(np.max(layer_ids))
            n_hits = len(layer_ids)
            frac_layers_hit = n_hits / nlayers

        rows.append({
            "track_idx": idx,
            "d0": d0,
            "phi0": phi0,
            "pt": pt,
            "dz": dz,
            "tanl": tanl,
            "n_hits": n_hits,
            "innermost_layer": innermost_layer,
            "outermost_layer": outermost_layer,
            "frac_layers_hit": frac_layers_hit,
            "min_r_hit": np.min(np.sqrt(xs**2 + ys**2)) if len(xs) else np.nan,
            "max_r_hit": np.max(np.sqrt(xs**2 + ys**2)) if len(xs) else np.nan,
            "min_abs_z": np.min(np.abs(zs)) if len(zs) else np.nan,
            "max_abs_z": np.max(np.abs(zs)) if len(zs) else np.nan,
            "hits_layer_1": int(1 in layer_ids),
            "hits_layer_2": int(2 in layer_ids),
            "hits_inner_2": int((1 in layer_ids) or (2 in layer_ids)),
        })

    return pd.DataFrame(rows)


def layer_occupancy(tracks):
    """
    Fraction of tracks with a hit in each detector layer.
    """
    counts = np.zeros(nlayers, dtype=int)

    for trk in tracks:
        _, layer_ids, _, _, _ = trk
        for lid in set(layer_ids):
            if 1 <= lid <= nlayers:
                counts[lid - 1] += 1

    return pd.DataFrame({
        "layer_id": np.arange(1, nlayers + 1),
        "radius": LAYER_RADII,
        "occupancy": counts / max(len(tracks), 1),
        "count": counts,
    })


def make_validation_plots(tracks, outdir):
    os.makedirs(outdir, exist_ok=True)

    summary = summarize_tracks(tracks)
    occ = layer_occupancy(tracks)

    summary.to_csv(os.path.join(outdir, "helical_validation_track_summary.csv"), index=False)
    print(f"Saved {outdir}/helical_validation_track_summary.csv")
    occ.to_csv(os.path.join(outdir, "helical_validation_layer_occupancy.csv"), index=False)
    print(f"Saved {outdir}/helical_validation_layer_occupancy.csv")
    print("\n=== Helical validation summary ===")
    print(summary[[
        "n_hits",
        "innermost_layer",
        "outermost_layer",
        "frac_layers_hit",
        "d0",
        "pt",
        "tanl",
        "max_abs_z",
    ]].describe())

    print("\n=== Inner layer hit fractions ===")
    print("Fraction hitting layer 1:", summary["hits_layer_1"].mean())
    print("Fraction hitting layer 2:", summary["hits_layer_2"].mean())
    print("Fraction hitting layer 1 or 2:", summary["hits_inner_2"].mean())

    print("\n=== Layer occupancy ===")
    print(occ.to_string(index=False))

    # Number of hits per track
    plt.figure(figsize=(7, 5))
    plt.hist(summary["n_hits"], bins=np.arange(0.5, nlayers + 1.5, 1), histtype="step", linewidth=2)
    plt.xlabel("Number of hits")
    plt.ylabel("Tracks")
    plt.title("Helical validation: number of hits")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_n_hits.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_n_hits.png")

    # Innermost layer reached
    plt.figure(figsize=(7, 5))
    plt.hist(summary["innermost_layer"].dropna(), bins=np.arange(0.5, nlayers + 1.5, 1), histtype="step", linewidth=2)
    plt.xlabel("Innermost layer hit")
    plt.ylabel("Tracks")
    plt.title("Helical validation: innermost layer hit")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_innermost_layer.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_innermost_layer.png")


    # Outermost layer reached
    plt.figure(figsize=(7, 5))
    plt.hist(summary["outermost_layer"].dropna(), bins=np.arange(0.5, nlayers + 1.5, 1), histtype="step", linewidth=2)
    plt.xlabel("Outermost layer hit")
    plt.ylabel("Tracks")
    plt.title("Helical validation: outermost layer hit")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_outermost_layer.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_outermost_layer.png")


    # Per-layer occupancy
    plt.figure(figsize=(7, 5))
    plt.plot(occ["layer_id"], occ["occupancy"], marker="o")
    plt.xlabel("Layer ID")
    plt.ylabel("Fraction of tracks with hit")
    plt.title("Helical validation: layer occupancy")
    plt.xticks(np.arange(1, nlayers + 1, 2))
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_layer_occupancy.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_layer_occupancy.png")

    # pt distribution
    plt.figure(figsize=(7, 5))
    plt.hist(summary["pt"], bins=50, histtype="step", linewidth=2)
    plt.xlabel(r"$p_T$")
    plt.ylabel("Tracks")
    plt.title("Helical validation: sampled pT")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_pt.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_pt.png")

    # d0 distribution
    plt.figure(figsize=(7, 5))
    plt.hist(summary["d0"], bins=50, histtype="step", linewidth=2)
    plt.xlabel(r"$d_0$")
    plt.ylabel("Tracks")
    plt.title("Helical validation: sampled d0")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_d0.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_d0.png")

    # n_hits vs pt
    plt.figure(figsize=(7, 5))
    plt.scatter(summary["pt"], summary["n_hits"], alpha=0.3, s=8)
    plt.xlabel(r"$p_T$")
    plt.ylabel("Number of hits")
    plt.title("Helical validation: hit count vs pT")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_nhits_vs_pt.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_nhits_vs_pt.png")

    # innermost layer vs d0
    plt.figure(figsize=(7, 5))
    plt.scatter(summary["d0"], summary["innermost_layer"], alpha=0.3, s=8)
    plt.xlabel(r"$d_0$")
    plt.ylabel("Innermost layer hit")
    plt.title("Helical validation: innermost layer vs d0")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "helical_innermost_vs_d0.png"), dpi=300)
    plt.close()
    os.system(f"open {outdir}/helical_innermost_vs_d0.png")


    print(f"\nWrote validation plots and CSVs to: {outdir}")

# generate tracks and output them
# format per track:
#   <d0>, <phi0>, <pt>, <dz>, <tanl>          (parameter header line)
#   <layer_id>, <x>, <y>, <z>                 (one line per layer actually hit)
#   ...                                        (variable count, 1..nlayers)
#   EOT
tracks = gen_tracks(n=N_TRACKS)

if VALIDATION_MODE:
    make_validation_plots(tracks, VALIDATION_DIR)

with open(OUTPUT_PATH, "w") as f:
    for trk in tracks:
        params, layer_ids, xs, ys, zs = trk
        f.write("%1.8f, %1.8f, %1.8f, %1.8f, %1.8f\n" % params)
        for lid, x, y, z in zip(layer_ids, xs, ys, zs):
            f.write("%d, %1.8f, %1.8f, %1.8f\n" % (lid, x, y, z))
        f.write("EOT\n\n")

print(f"Wrote tracks to: {OUTPUT_PATH}")
