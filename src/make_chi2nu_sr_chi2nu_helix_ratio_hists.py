import numpy as np
import matplotlib.pyplot as plt
from os import system
import os
import re
import json
import pandas as pd
from pathlib import Path

dataset_choice = "100_track_Finder_Output_Levi"

USE_K_FROM_HTML = True
K_HELIX = 5

# Important:
# These are the HTML reports moved by fit_tracks.py into your Whiteson directory.
HTML_DIR = "/Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson"

files = {
    "10_tracks_case":
    {
        "SM": "../data_files/chi2_ratio_values_v20260407_091434__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel_using_Phi.csv",
        "5-mode": "../data_files/chi2_ratio_values_v20260430_092453__train5_test5__layers25_len320p0__fd5-5__func3-3__noiseXY0p01_Z0p01.csv",
        "25-mode": "../data_files/chi2_ratio_values_v20260305_002410__train5_test5__layers25_len320p0__fd25-25__func3-3__noiseXY0p01_Z0p01.csv",
        "Noise": "../data_files/chi2_ratio_values_v20260430_193549__train5_test5__layers25_len320p0__fd5-5__func3-3__noiseXY0p01_Z0p01__randomNoise.csv",
    },
    "100_tracks_case":
    {
        "25-mode": "../data_files/chi2_ratio_values_v20260518_142036__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.csv",
        "5-mode": "../data_files/chi2_ratio_values_v20260518_131139__train50_test50__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.csv",
        "SM": "../data_files/chi2_ratio_values_v20260518_142850__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel_using_Phi.csv",
        "Noise": "../data_files/chi2_ratio_values_v20260519_110640__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__randomNoise.csv",
        "Levi purity<0.6": "../data_files/chi2_ratio_values_Levi_dominant_purity_lt_0p6_minhits_20.csv"
    },
    "100_track_Finder_Output_Levi":
    {
        "Levi purity<0.6": "../data_files/chi2_ratio_values_Levi_dominant_purity_lt_0p6_minhits_20.csv",
        "Levi non-helical": "../data_files/chi2_ratio_values_Levi_reco_nonhelical_PID15_purity_ge_0p9_eff_ge_0p5_minhits_20.csv",
        "Levi SM": "../data_files/chi2_ratio_values_Levi_SM_high_purity_minhits_20.csv",
        "Levi purity<0.9": "../data_files/chi2_ratio_values_Levi_impure_purity_lt_0p9_minhits_20.csv"
    }
}[dataset_choice]


html_files = {
    "100_tracks_case": {
        "25-mode": "v20260518_142036__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.html",
        "5-mode": "v20260518_131139__train50_test50__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.html",
        "SM": "v20260518_142850__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel_using_Phi.html",
        "Noise": "v20260519_110640__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__randomNoise.html",
        "Levi purity<0.6": "Levi_dominant_purity_lt_0p6_minhits_20.html"
    },
    "100_track_Finder_Output_Levi": {
        "Levi purity<0.6": "Levi_dominant_purity_lt_0p6_minhits_20.html",
        "Levi non-helical": "Levi_reco_nonhelical_PID15_purity_ge_0p9_eff_ge_0p5_minhits_20.html",
        "Levi SM": "Levi_SM_high_purity_minhits_20.html",
        "Levi purity<0.9": "Levi_impure_purity_lt_0p9_minhits_20.html"
    }
}[dataset_choice]


def extract_js_json_var(html_text, var_name):
    """
    Extract a JS const assignment like:

        const slides = [...];
        const eqSlides = [...];
        const eqStats = {...};

    This is safer than regexing until the first semicolon, because the JSON
    strings themselves contain LaTeX semicolons like \\;.
    """
    m = re.search(rf"const\s+{re.escape(var_name)}\s*=", html_text)
    if not m:
        raise ValueError(f"Could not find JS variable {var_name}")

    # Move to first non-whitespace character after '='
    i = m.end()
    while i < len(html_text) and html_text[i].isspace():
        i += 1

    if i >= len(html_text) or html_text[i] not in "[{":
        raise ValueError(
            f"Expected JSON array/object after const {var_name} =, "
            f"got {html_text[i:i+40]!r}"
        )

    opener = html_text[i]
    closer = "]" if opener == "[" else "}"

    start = i
    depth = 0
    in_string = False
    escape = False

    for j in range(start, len(html_text)):
        ch = html_text[j]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        # Not currently inside a JSON string
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                json_text = html_text[start:j+1]
                return json.loads(json_text)

    raise ValueError(f"Could not find end of JSON for variable {var_name}")


def extract_dataset_folder(html_text):
    """
    Reads:
        Dataset: <span class="mono">../tracks_for_ed/...</span>
    after placeholders have been filled.
    """
    m = re.search(
        r'Dataset:\s*<span class="mono">(.*?)</span>',
        html_text,
        flags=re.S,
    )
    if not m:
        return None
    s = m.group(1)
    s = (
        s.replace("&amp;", "&")
         .replace("&lt;", "<")
         .replace("&gt;", ">")
    )
    return s.strip()


def count_template_params_from_latex(eq_latex):
    """
    Count distinct template parameters in an equation like:
        $x(s) = a_{0} + a_{1}\sin(a_{2}s)$

    Handles common latex forms:
        a_{0}, a_{12}, a_0, a_12
    """
    params = set()

    for m in re.finditer(r"a_\{(\d+)\}", eq_latex):
        params.add(int(m.group(1)))

    for m in re.finditer(r"a_(\d+)", eq_latex):
        params.add(int(m.group(1)))

    return len(params)


def build_k_lookup_from_html(html_path):
    """
    Returns:
      slides
      dataset_folder
      k_by_slide_coord[(slide_index, "x"/"y"/"z")] = number of parameters
    """
    html_text = Path(html_path).read_text(encoding="utf-8")

    slides = extract_js_json_var(html_text, "slides")
    eq_slides = extract_js_json_var(html_text, "eqSlides")
    dataset_folder = extract_dataset_folder(html_text)

    k_by_slide_coord = {}
    eq_by_slide_coord = {}

    for eq_item in eq_slides:
        k = count_template_params_from_latex(eq_item["eq"])

        for use in eq_item["uses"]:
            slide_idx = int(use["slide"])

            # Newer HTML has explicit coord in uses.
            # Older labels look like "17-x".
            coord = use.get("coord", None)
            if coord is None:
                label = use.get("label", "")
                m = re.search(r"-(x|y|z)$", label)
                if not m:
                    raise ValueError(f"Could not infer coord from use={use}")
                coord = m.group(1)

            k_by_slide_coord[(slide_idx, coord)] = k
            eq_by_slide_coord[(slide_idx, coord)] = eq_item["eq"]

    return slides, dataset_folder, k_by_slide_coord, eq_by_slide_coord


def resolve_track_csv_path(dataset_folder, event_id):
    """
    Find the hit CSV corresponding to one HTML slide.

    Handles normal files like:
        event100000001-hits.csv

    and Levi fake files like:
        fake_reco_event65050_track174-hits.csv
        fake_reco_000_event65050_track174-hits.csv
    """
    folder = Path(dataset_folder)

    candidates = [
        folder / f"{event_id}.csv",
        folder / event_id,
        folder / f"{event_id}-hits.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Fallback: Levi fake files may have prefixes such as fake_reco_...
    # Example event_id from HTML:
    #   event65050_track174-hits
    # possible file:
    #   fake_reco_event65050_track174-hits.csv
    #   fake_reco_000_event65050_track174-hits.csv
    glob_patterns = [
        f"*{event_id}.csv",
        f"*{event_id}",
    ]

    matches = []
    for pat in glob_patterns:
        matches.extend(folder.glob(pat))

    # Remove duplicates while preserving deterministic order
    matches = sorted(set(matches))

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"[warning] multiple matches for event_id={event_id}:")
        for m in matches:
            print(f"  {m}")
        print(f"[warning] using first match: {matches[0]}")
        return matches[0]

    # Last-resort diagnostic: show a few files in the folder
    preview = sorted(folder.glob("*.csv"))[:10]
    raise FileNotFoundError(
        f"Could not find hit CSV for event_id={event_id} in {dataset_folder}.\n"
        f"Tried exact candidates: {candidates}\n"
        f"Tried glob patterns: {glob_patterns}\n"
        f"First CSVs in folder: {preview}"
    )


def count_hits_for_slide(dataset_folder, slide):
    path = resolve_track_csv_path(dataset_folder, slide["event_id"])
    return len(pd.read_csv(path))


def corrected_ratios_for_dataset(label, ratio_csv_path, html_path):
    """
    Old CSV stores:
      col0 = log_chi2_sr_over_helix
      col1 = chi2nu_sr
      col2 = chi2nu_helix

    In fit_tracks.py for USE_K=False:
      chi2nu_sr    = chi2_sr_total / (3*n_hits - 1)
      chi2nu_helix = chi2_helix_total / (3*n_hits)

    We invert those to recover approximate total chi2, then recompute:
      chi2nu_sr_K    = chi2_sr_total    / (3*n_hits - k_sr)
      chi2nu_helix_K = chi2_helix_total / (3*n_hits - 5)
    """
    data = np.loadtxt(ratio_csv_path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)

    old_log_ratio = data[:, 0]
    old_chi2nu_sr = data[:, 1]
    old_chi2nu_helix = data[:, 2]

    slides, dataset_folder, k_by_slide_coord, eq_by_slide_coord = build_k_lookup_from_html(html_path)

    if len(slides) != len(data):
        raise ValueError(
            f"{label}: HTML has {len(slides)} slides but CSV has {len(data)} rows."
        )

    new_rows = []

    for i, slide in enumerate(slides):
        n_hits = count_hits_for_slide(dataset_folder, slide)
        n_obs = 3 * n_hits

        kx = k_by_slide_coord.get((i, "x"), 0)
        ky = k_by_slide_coord.get((i, "y"), 0)
        kz = k_by_slide_coord.get((i, "z"), 0)
        k_sr = kx + ky + kz
        eq_x = eq_by_slide_coord.get((i, "x"), "")
        eq_y = eq_by_slide_coord.get((i, "y"), "")
        eq_z = eq_by_slide_coord.get((i, "z"), "")

        # Recover total chi2 from old saved averages.
        # This matches your current fit_tracks.py convention.
        chi2_sr_total = old_chi2nu_sr[i] * max(n_obs - 1, 1)
        chi2_helix_total = old_chi2nu_helix[i] * n_obs

        dof_sr = max(n_obs - k_sr, 1)
        dof_helix = max(n_obs - K_HELIX, 1)

        chi2nu_sr_k = chi2_sr_total / dof_sr
        chi2nu_helix_k = chi2_helix_total / dof_helix
        log_ratio_k = np.log((chi2nu_sr_k + 1e-300) / (chi2nu_helix_k + 1e-300))

        new_rows.append({
            "track_idx": slide["track_idx"],
            "event_id": slide["event_id"],
            "n_hits": n_hits,
            "n_obs": n_obs,
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "k_sr": k_sr,
            "k_helix": K_HELIX,
            "old_log_ratio": old_log_ratio[i],
            "old_chi2nu_sr": old_chi2nu_sr[i],
            "old_chi2nu_helix": old_chi2nu_helix[i],
            "chi2_sr_total_recovered": chi2_sr_total,
            "chi2_helix_total_recovered": chi2_helix_total,
            "chi2nu_sr_K": chi2nu_sr_k,
            "chi2nu_helix_K": chi2nu_helix_k,
            "log_ratio_K": log_ratio_k,
            "eq_x": eq_x,
            "eq_y": eq_y,
            "eq_z": eq_z,
        })

    return pd.DataFrame(new_rows)

def summarize_sr_tail(label, df, threshold=-10.0):
    """
    Compact diagnostics for any dataset, without needing truth metadata.
    Useful for SM and purity<0.9 tails.
    """
    d = df.copy()
    d["dof_sr_raw"] = d["n_obs"] - d["k_sr"]
    d["underconstrained"] = d["dof_sr_raw"] <= 0
    d["sr_like"] = d["log_ratio_K"] < threshold

    rows = []

    for region_name, mask in {
        "all": np.ones(len(d), dtype=bool),
        "constrained: N_obs > k_SR": d["dof_sr_raw"] > 0,
        "underconstrained: N_obs <= k_SR": d["dof_sr_raw"] <= 0,
        f"SR-like: log_ratio_K < {threshold}": d["sr_like"],
        f"not SR-like: log_ratio_K >= {threshold}": ~d["sr_like"],
    }.items():
        sub = d.loc[mask].copy()

        if len(sub) == 0:
            rows.append({
                "label": label,
                "region": region_name,
                "n": 0,
                "frac": 0.0,
                "median_log_ratio_K": np.nan,
                "mean_log_ratio_K": np.nan,
                "median_n_hits": np.nan,
                "median_k_sr": np.nan,
                "median_dof_sr_raw": np.nan,
                "frac_underconstrained": np.nan,
                "frac_sr_like": np.nan,
            })
            continue

        rows.append({
            "label": label,
            "region": region_name,
            "n": len(sub),
            "frac": len(sub) / len(d),
            "median_log_ratio_K": sub["log_ratio_K"].median(),
            "mean_log_ratio_K": sub["log_ratio_K"].mean(),
            "median_n_hits": sub["n_hits"].median(),
            "median_k_sr": sub["k_sr"].median(),
            "median_dof_sr_raw": sub["dof_sr_raw"].median(),
            "frac_underconstrained": sub["underconstrained"].mean(),
            "frac_sr_like": sub["sr_like"].mean(),
        })

    out = pd.DataFrame(rows)

    print(f"\n=== SR tail / constraint diagnostic: {label} ===")
    print(out.to_string(index=False))

    safe_label = (
        label.replace(" ", "_")
             .replace("<", "lt")
             .replace(">", "gt")
             .replace("/", "_")
    )

    out.to_csv(
        f"sr_tail_constraint_summary_{safe_label}_{dataset_choice}.csv",
        index=False,
    )

    tail_cols = [
        "event_id",
        "n_hits",
        "n_obs",
        "k_sr",
        "kx",
        "ky",
        "kz",
        "dof_sr_raw",
        "old_log_ratio",
        "log_ratio_K",
        "chi2nu_sr_K",
        "chi2nu_helix_K",
    ]

    print(f"\n--- Most SR-like examples: {label} ---")
    print(
        d[tail_cols]
        .sort_values("log_ratio_K")
        .head(15)
        .to_string(index=False)
    )

    d.sort_values("log_ratio_K").head(25).to_csv(
        f"most_SR_like_examples_{safe_label}_{dataset_choice}.csv",
        index=False,
    )
    
    # ------------------------------------------------------------
    # Template-size signatures by constraint status
    # ------------------------------------------------------------
    sig_cols = ["kx", "ky", "kz", "k_sr"]

    template_sig_counts = (
        d.assign(
            constraint_region=np.where(
                d["dof_sr_raw"] > 0,
                "constrained",
                "underconstrained",
            )
        )
        .groupby(["constraint_region"] + sig_cols)
        .size()
        .reset_index(name="count")
        .sort_values(["constraint_region", "count"], ascending=[True, False])
    )

    print(f"\n=== Template-size signatures by constraint status: {label} ===")
    print(template_sig_counts.head(5).to_string(index=False))

    template_sig_counts.to_csv(
        f"template_size_signatures_{safe_label}_{dataset_choice}.csv",
        index=False,
    )
    
    # Example equations for the most common signatures
    example_eqs = (
        d.sort_values("log_ratio_K")
        .groupby(sig_cols)
        .head(1)[sig_cols + ["eq_x", "eq_y", "eq_z", "log_ratio_K", "n_hits", "dof_sr_raw"]]
        .sort_values("k_sr")
    )

    print(f"\n=== Example equations by template-size signature: {label} ===")
    print(example_eqs.head(5).to_string(index=False))

    example_eqs.to_csv(
        f"example_equations_by_template_signature_{safe_label}_{dataset_choice}.csv",
        index=False,
    )

    # Also save the full per-track table with constraint labels.
    d.to_csv(
        f"per_track_constraint_diagnostics_{safe_label}_{dataset_choice}.csv",
        index=False,
    )

    return out

# -------------------------
# Load data, possibly corrected
# -------------------------
results = {}

for label, path in files.items():
    if USE_K_FROM_HTML:
        html_path = Path(HTML_DIR) / html_files[label]
        df = corrected_ratios_for_dataset(label, path, html_path)

        out_csv = f"corrected_chi2_ratio_values_{label}_{dataset_choice}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[{label}] wrote {out_csv}")
        print(df[["k_sr", "n_obs", "old_log_ratio", "log_ratio_K"]].describe())
        summarize_sr_tail(label, df, threshold=-10.0)

        # ------------------------------------------------------------
        # Extra diagnostics for the realistic Levi fake-track sample
        # ------------------------------------------------------------
        if label == "Levi purity<0.6":
            html_text_for_metadata = Path(html_path).read_text(encoding="utf-8")
            dataset_folder_for_metadata = extract_dataset_folder(html_text_for_metadata)
            # This threshold is just a first-pass split.
            # It separates the mildly SR-favored/noise-like group from the strongly SR-favored group.
            fake_split_threshold = -10.0

            smooth_fake = df[df["log_ratio_K"] < fake_split_threshold].copy()
            helixlike_fake = df[df["log_ratio_K"] >= fake_split_threshold].copy()
            
            # ------------------------------------------------------------
            # Merge PID / truth metadata from the original fake-track CSVs
            # ------------------------------------------------------------
            def add_metadata_from_hit_csvs(df_in):
                rows = []

                for _, row in df_in.iterrows():
                    hit_path = resolve_track_csv_path(dataset_folder_for_metadata, row["event_id"])
                    hits = pd.read_csv(hit_path)

                    meta = row.to_dict()

                    # Use dominant truth contributor, not first row.
                    if "n_shared" in hits.columns:
                        dom_idx = hits["n_shared"].astype(float).idxmax()
                    else:
                        dom_idx = hits.index[0]

                    for col in [
                        "PID",
                        "particle_id",
                        "purity_reco",
                        "eff_true",
                        "n_shared",
                        "n_reco_hits",
                        "n_true_hits",
                        "pt",
                        "event_id",
                        "track_id",
                    ]:
                        if col in hits.columns:
                            meta[col] = hits.loc[dom_idx, col]

                    meta["n_rows_csv"] = len(hits)

                    if "purity_reco" in hits.columns:
                        meta["max_purity_reco"] = hits["purity_reco"].max()

                    if "n_shared" in hits.columns:
                        meta["max_n_shared"] = hits["n_shared"].max()

                    rows.append(meta)

                return pd.DataFrame(rows)

            real_meta = add_metadata_from_hit_csvs(df)
            print("\n=== Metadata purity sanity check ===")
            print(real_meta["purity_reco"].describe())
            print("num purity_reco >= 0.6:", (real_meta["purity_reco"] >= 0.6).sum())
            print("max purity_reco:", real_meta["purity_reco"].max())

            smooth_meta = real_meta[real_meta["log_ratio_K"] < fake_split_threshold].copy()
            helixlike_meta = real_meta[real_meta["log_ratio_K"] >= fake_split_threshold].copy()

            print("\n=== Real Fakes PID counts: all ===")
            print(real_meta["PID"].value_counts(dropna=False))

            print("\n=== Real Fakes PID counts: smooth / SR-like ===")
            print(smooth_meta["PID"].value_counts(dropna=False))

            print("\n=== Real Fakes PID counts: helixlike-or-noise-like ===")
            print(helixlike_meta["PID"].value_counts(dropna=False))

            print("\n=== Real Fakes PID fractions: smooth / SR-like ===")
            print(smooth_meta["PID"].value_counts(normalize=True, dropna=False))

            print("\n=== Real Fakes PID fractions: helixlike-or-noise-like ===")
            print(helixlike_meta["PID"].value_counts(normalize=True, dropna=False))

            # Save metadata-enriched tables
            real_meta.to_csv(f"real_fakes_with_metadata_{dataset_choice}.csv", index=False)
            smooth_meta.to_csv(f"real_fakes_smooth_SR_like_with_metadata_{dataset_choice}.csv", index=False)
            helixlike_meta.to_csv(f"real_fakes_helixlike_or_noise_like_with_metadata_{dataset_choice}.csv", index=False)

            print("\n=== Real Fakes split by corrected SR/helix ratio ===")
            print(f"threshold log_ratio_K < {fake_split_threshold}")
            print("smooth / SR-like fake count:", len(smooth_fake))
            print("helixlike-or-noise-like fake count:", len(helixlike_fake))

            cols = ["event_id", "n_hits", "k_sr", "kx", "ky", "kz",
                    "old_log_ratio", "log_ratio_K",
                    "chi2nu_sr_K", "chi2nu_helix_K"]

            print("\n--- Smooth / SR-like fake examples ---")
            print(
                smooth_fake[cols]
                .sort_values("log_ratio_K")
                .head(20)
                .to_string(index=False)
            )

            print("\n--- Helixlike-or-noise-like fake examples ---")
            print(
                helixlike_fake[cols]
                .sort_values("log_ratio_K", ascending=False)
                .head(20)
                .to_string(index=False)
            )

            # Save the split tables too, so you can inspect them later.
            smooth_fake.to_csv(
                f"real_fakes_smooth_SR_like_{dataset_choice}.csv",
                index=False,
            )
            helixlike_fake.to_csv(
                f"real_fakes_helixlike_or_noise_like_{dataset_choice}.csv",
                index=False,
            )
            
            plt.figure(figsize=(8, 5))

            pid_counts = pd.DataFrame({
            "smooth_SR_like": smooth_meta["PID"].value_counts(),
            "helixlike_or_noise_like": helixlike_meta["PID"].value_counts(),
            }).fillna(0)

            pid_counts.plot(kind="bar", figsize=(8, 5))
            plt.xlabel("PID")
            plt.ylabel("Tracks")
            plt.title("Real fake PID composition by SR/helix-ratio category")
            plt.tight_layout()
            plt.savefig(f"real_fakes_pid_split_{dataset_choice}.png", dpi=300)
            system(f"open real_fakes_pid_split_{dataset_choice}.png")
            plt.close()
            
            # ------------------------------------------------------------
            # Check whether the real-fake split is driven by SR underconstraint
            # ------------------------------------------------------------
            real_meta["dof_sr_raw"] = real_meta["n_obs"] - real_meta["k_sr"]
            real_meta["dof_helix"] = real_meta["n_obs"] - real_meta["k_helix"]
            real_meta["sr_overparameterized"] = real_meta["dof_sr_raw"] <= 0

            smooth_meta = real_meta[real_meta["log_ratio_K"] < fake_split_threshold].copy()
            helixlike_meta = real_meta[real_meta["log_ratio_K"] >= fake_split_threshold].copy()

            print("\n=== Real Fakes SR DOF diagnostic: all ===")
            print(
                real_meta[["n_hits", "n_obs", "k_sr", "dof_sr_raw", "log_ratio_K"]]
                .describe()
            )

            print("\n=== SR overparameterized counts ===")
            print("all:", real_meta["sr_overparameterized"].sum(), "/", len(real_meta))
            print(
                "smooth / SR-like:",
                smooth_meta["sr_overparameterized"].sum(),
                "/",
                len(smooth_meta),
            )
            print(
                "helixlike-or-noise-like:",
                helixlike_meta["sr_overparameterized"].sum(),
                "/",
                len(helixlike_meta),
            )

            print("\n=== Mean metadata by SR/helix category ===")
            compare_cols = [
                "n_hits",
                "n_obs",
                "k_sr",
                "dof_sr_raw",
                "purity_reco",
                "eff_true",
                "n_shared",
                "n_reco_hits",
                "n_true_hits",
                "pt",
                "log_ratio_K",
            ]

            category_summary = (
                real_meta
                .assign(category=np.where(
                    real_meta["log_ratio_K"] < fake_split_threshold,
                    "smooth_SR_like",
                    "helixlike_or_noise_like",
                ))
                .groupby("category")[compare_cols]
                .agg(["count", "mean", "median", "std", "min", "max"])
            )

            print(category_summary)

            real_meta.to_csv(
                f"real_fakes_with_metadata_and_dof_{dataset_choice}.csv",
                index=False,
            )
            # ------------------------------------------------------------
            # Replot real fakes after requiring SR to be constrained
            # ------------------------------------------------------------
            constrained_real = real_meta[real_meta["dof_sr_raw"] > 0].copy()
            underconstrained_real = real_meta[real_meta["dof_sr_raw"] <= 0].copy()

            print("\n=== Real Fakes constrained-only summary ===")
            print("constrained count:", len(constrained_real))
            print("underconstrained count:", len(underconstrained_real))

            print("\nConstrained real fakes log_ratio_K:")
            print(constrained_real["log_ratio_K"].describe())

            print("\nUnderconstrained real fakes log_ratio_K:")
            print(underconstrained_real["log_ratio_K"].describe())

            plt.figure(figsize=(7, 5))
            plt.hist(
                underconstrained_real["log_ratio_K"],
                bins=10,
                histtype="step",
                linewidth=2,
                label=f"underconstrained: N_obs - k_SR <= 0, n={len(underconstrained_real)}",
            )
            plt.hist(
                constrained_real["log_ratio_K"],
                bins=10,
                histtype="step",
                linewidth=2,
                label=f"constrained: N_obs - k_SR > 0, n={len(constrained_real)}",
            )
            plt.axvline(fake_split_threshold, linestyle="--", linewidth=1)
            plt.axvline(0.0, linestyle="--", linewidth=1)
            plt.xlabel(r"$\log(\chi^2_{\nu,\mathrm{SR},K}/\chi^2_{\nu,\mathrm{helix},K})$")
            plt.ylabel("Tracks")
            plt.title("Real fakes split by SR template constraint")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"real_fakes_constrained_vs_underconstrained_{dataset_choice}.png", dpi=300)
            system(f"open real_fakes_constrained_vs_underconstrained_{dataset_choice}.png")
            plt.close()
            
            # ------------------------------------------------------------
            # Compact real-fake SR metric failure-mode summary
            # ------------------------------------------------------------
            real_meta["dof_sr_raw"] = real_meta["n_obs"] - real_meta["k_sr"]
            real_meta["sr_constraint_region"] = np.where(
                real_meta["dof_sr_raw"] > 0,
                "constrained: N_obs > k_SR",
                "underconstrained: N_obs <= k_SR",
            )
            real_meta["sr_like"] = real_meta["log_ratio_K"] < fake_split_threshold

            summary_cols = [
                "n_hits",
                "n_obs",
                "k_sr",
                "dof_sr_raw",
                "purity_reco",
                "eff_true",
                "n_shared",
                "n_reco_hits",
                "n_true_hits",
                "pt",
                "log_ratio_K",
            ]

            metric_summary = (
                real_meta
                .groupby("sr_constraint_region")
                .agg(
                    count=("log_ratio_K", "size"),
                    mean_n_hits=("n_hits", "mean"),
                    median_n_hits=("n_hits", "median"),
                    mean_n_obs=("n_obs", "mean"),
                    mean_k_sr=("k_sr", "mean"),
                    mean_dof_sr_raw=("dof_sr_raw", "mean"),
                    mean_purity_reco=("purity_reco", "mean"),
                    median_purity_reco=("purity_reco", "median"),
                    mean_eff_true=("eff_true", "mean"),
                    mean_log_ratio_K=("log_ratio_K", "mean"),
                    median_log_ratio_K=("log_ratio_K", "median"),
                    frac_sr_like=("sr_like", "mean"),
                )
                .reset_index()
            )

            print("\n=== Real fakes: SR metric vs constraint region ===")
            print(metric_summary.to_string(index=False))

            metric_summary.to_csv(
                f"real_fakes_metric_constraint_summary_{dataset_choice}.csv",
                index=False,
            )
            
            real_meta = real_meta.copy()

            real_meta["dof_sr_raw"] = real_meta["n_obs"] - real_meta["k_sr"]
            real_meta["sr_like"] = real_meta["log_ratio_K"] < fake_split_threshold
            real_meta["sr_constrained"] = real_meta["dof_sr_raw"] > 0

            cuts = [
                ("all Levi purity<0.6", np.ones(len(real_meta), dtype=bool)),
                ("n_hits >= 25", real_meta["n_hits"] >= 25),
                ("n_hits >= 30", real_meta["n_hits"] >= 30),
                ("n_hits >= 35", real_meta["n_hits"] >= 35),
                ("n_hits >= 40", real_meta["n_hits"] >= 40),
                ("N_obs > k_SR", real_meta["dof_sr_raw"] > 0),
                ("n_hits >= 25 and N_obs > k_SR",
                 (real_meta["n_hits"] >= 25) & (real_meta["dof_sr_raw"] > 0)),
            ]

            summary_rows = []

            n_total = len(real_meta)

            for cut_name, mask in cuts:
                sub = real_meta.loc[mask].copy()

                if len(sub) == 0:
                    summary_rows.append({
                        "cut": cut_name,
                        "n_tracks": 0,
                        "frac_remaining": 0.0,
                        "mean_n_hits": np.nan,
                        "median_n_hits": np.nan,
                        "mean_dof_sr_raw": np.nan,
                        "median_dof_sr_raw": np.nan,
                        "mean_log_ratio_K": np.nan,
                        "median_log_ratio_K": np.nan,
                        "frac_sr_like": np.nan,
                        "frac_underconstrained": np.nan,
                    })
                    continue

                summary_rows.append({
                    "cut": cut_name,
                    "n_tracks": len(sub),
                    "frac_remaining": len(sub) / n_total,
                    "mean_n_hits": sub["n_hits"].mean(),
                    "median_n_hits": sub["n_hits"].median(),
                    "mean_dof_sr_raw": sub["dof_sr_raw"].mean(),
                    "median_dof_sr_raw": sub["dof_sr_raw"].median(),
                    "mean_log_ratio_K": sub["log_ratio_K"].mean(),
                    "median_log_ratio_K": sub["log_ratio_K"].median(),
                    "frac_sr_like": sub["sr_like"].mean(),
                    "frac_underconstrained": (sub["dof_sr_raw"] <= 0).mean(),
                })

            quality_cut_summary = pd.DataFrame(summary_rows)

            print("\n=== Real fakes: quality-cut summary ===")
            print(quality_cut_summary.to_string(index=False))

            quality_cut_summary.to_csv(
                "real_fakes_quality_cut_summary.csv",
                index=False,
            )
            
            # ------------------------------------------------------------
            # Plot: log_ratio_K vs n_hits, colored by SR constraint region
            # ------------------------------------------------------------

            plt.figure(figsize=(7, 5))

            constrained = real_meta["dof_sr_raw"] > 0
            underconstrained = ~constrained

            plt.scatter(
                real_meta.loc[underconstrained, "n_hits"],
                real_meta.loc[underconstrained, "log_ratio_K"],
                alpha=0.7,
                label=f"underconstrained: N_obs <= k_SR, n={underconstrained.sum()}",
            )

            plt.scatter(
                real_meta.loc[constrained, "n_hits"],
                real_meta.loc[constrained, "log_ratio_K"],
                alpha=0.7,
                label=f"constrained: N_obs > k_SR, n={constrained.sum()}",
            )

            plt.axhline(fake_split_threshold, linestyle="--", linewidth=1)
            plt.axhline(0.0, linestyle="--", linewidth=1)

            plt.xlabel("n_hits")
            plt.ylabel(r"$\log(\chi^2_{\nu,\mathrm{SR},K}/\chi^2_{\nu,\mathrm{helix},K})$")
            plt.title("Levi purity<0.6 candidates: SR/helix ratio vs hit count")
            plt.legend()
            plt.tight_layout()
            plt.savefig("real_fakes_log_ratio_vs_n_hits.png", dpi=300)
            plt.close()
            system("open real_fakes_log_ratio_vs_n_hits.png")

            # ------------------------------------------------------------
            # Plot: histograms after selected quality cuts
            # ------------------------------------------------------------

            hist_cuts = [
                ("all low purity", np.ones(len(real_meta), dtype=bool)),
                ("n_hits >= 25", real_meta["n_hits"] >= 25),
                ("n_hits >= 40", real_meta["n_hits"] >= 40),
                ("N_obs > k_SR", real_meta["dof_sr_raw"] > 0),
            ]

            plt.figure(figsize=(7, 5))

            bins = np.linspace(
                real_meta["log_ratio_K"].min(),
                real_meta["log_ratio_K"].max(),
                25,
            )

            for cut_name, mask in hist_cuts:
                sub = real_meta.loc[mask]
                plt.hist(
                    sub["log_ratio_K"],
                    bins=bins,
                    histtype="step",
                    linewidth=2,
                    label=f"{cut_name}, n={len(sub)}",
                )

            plt.axvline(fake_split_threshold, linestyle="--", linewidth=1)
            plt.axvline(0.0, linestyle="--", linewidth=1)

            plt.xlabel(r"$\log(\chi^2_{\nu,\mathrm{SR},K}/\chi^2_{\nu,\mathrm{helix},K})$")
            plt.ylabel("Tracks")
            plt.title("Levi purity<0.6 candidates: SR/helix ratio after quality cuts")
            plt.legend()
            plt.tight_layout()
            plt.savefig("real_fakes_quality_cut_histograms.png", dpi=300)
            plt.close()
            system("open real_fakes_quality_cut_histograms.png")
            
            # ------------------------------------------------------------
            # Real-fake population breakdown by SR-like / constraint status
            # ------------------------------------------------------------

            real_meta = real_meta.copy()

            fake_split_threshold = -10.0

            real_meta["dof_sr_raw"] = real_meta["n_obs"] - real_meta["k_sr"]
            real_meta["sr_like"] = real_meta["log_ratio_K"] < fake_split_threshold
            real_meta["underconstrained"] = real_meta["dof_sr_raw"] <= 0

            def fake_population(row):
                if row["underconstrained"] and row["sr_like"]:
                    return "SR-like underconstrained"
                elif row["underconstrained"] and not row["sr_like"]:
                    return "not-SR-like underconstrained"
                else:
                    return "constrained"

            real_meta["fake_population"] = real_meta.apply(fake_population, axis=1)

            cols_to_summarize = [
                "n_hits",
                "n_obs",
                "k_sr",
                "dof_sr_raw",
                "purity_reco",
                "eff_true",
                "n_shared",
                "n_reco_hits",
                "n_true_hits",
                "pt",
                "log_ratio_K",
            ]

            available_cols = [c for c in cols_to_summarize if c in real_meta.columns]

            fake_pop_summary = (
                real_meta
                .groupby("fake_population")
                .agg(
                    count=("log_ratio_K", "size"),
                    mean_n_hits=("n_hits", "mean"),
                    median_n_hits=("n_hits", "median"),
                    mean_dof_sr_raw=("dof_sr_raw", "mean"),
                    median_dof_sr_raw=("dof_sr_raw", "median"),
                    mean_purity_reco=("purity_reco", "mean"),
                    median_purity_reco=("purity_reco", "median"),
                    mean_eff_true=("eff_true", "mean"),
                    median_eff_true=("eff_true", "median"),
                    mean_n_shared=("n_shared", "mean"),
                    median_n_shared=("n_shared", "median"),
                    mean_n_reco_hits=("n_reco_hits", "mean"),
                    median_n_reco_hits=("n_reco_hits", "median"),
                    mean_n_true_hits=("n_true_hits", "mean"),
                    median_n_true_hits=("n_true_hits", "median"),
                    mean_pt=("pt", "mean"),
                    median_pt=("pt", "median"),
                    mean_log_ratio_K=("log_ratio_K", "mean"),
                    median_log_ratio_K=("log_ratio_K", "median"),
                )
                .reset_index()
            )

            print("\n=== Real fakes: population breakdown ===")
            print(fake_pop_summary.to_string(index=False))

            fake_pop_summary.to_csv(
                "real_fakes_population_breakdown.csv",
                index=False,
            )
            # ------------------------------------------------------------
            # PID composition by fake population
            # ------------------------------------------------------------

            if "PID" in real_meta.columns:
                pid_counts = (
                    real_meta
                    .groupby(["fake_population", "PID"])
                    .size()
                    .reset_index(name="count")
                )

                pid_counts["frac_within_population"] = (
                    pid_counts["count"]
                    / pid_counts.groupby("fake_population")["count"].transform("sum")
                )

                print("\n=== Real fakes: PID composition by population ===")
                print(
                    pid_counts
                    .sort_values(["fake_population", "count"], ascending=[True, False])
                    .to_string(index=False)
                )

                pid_counts.to_csv(
                    "real_fakes_pid_by_population.csv",
                    index=False,
                )

                vals = df["log_ratio_K"].to_numpy(float)
                chi2_sr = df["chi2nu_sr_K"].to_numpy(float)
                chi2_helix = df["chi2nu_helix_K"].to_numpy(float)
            # ------------------------------------------------------------
            # Plot: purity vs efficiency, colored by fake population
            # ------------------------------------------------------------

            plt.figure(figsize=(7, 5))

            for pop, sub in real_meta.groupby("fake_population"):
                plt.scatter(
                    sub["purity_reco"],
                    sub["eff_true"],
                    alpha=0.75,
                    label=f"{pop}, n={len(sub)}",
                )

            plt.xlabel("purity_reco")
            plt.ylabel("eff_true")
            plt.title("Levi purity<0.6 candidates: purity vs efficiency by SR/fake population")
            plt.legend()
            plt.tight_layout()
            plt.savefig("real_fakes_purity_vs_efficiency_by_population.png", dpi=300)
            plt.close()
            system("open real_fakes_purity_vs_efficiency_by_population.png")
            plt.figure(figsize=(7, 5))

            for pop, sub in real_meta.groupby("fake_population"):
                plt.scatter(
                    sub["n_hits"],
                    sub["purity_reco"],
                    alpha=0.75,
                    label=f"{pop}, n={len(sub)}",
                )

            plt.axvline(25, linestyle="--", linewidth=1)
            plt.axvline(30, linestyle="--", linewidth=1)

            plt.xlabel("n_hits")
            plt.ylabel("purity_reco")
            plt.title("Levi purity<0.6 candidates: purity vs hit count by SR/fake population")
            plt.legend()
            plt.tight_layout()
            plt.savefig("real_fakes_purity_vs_nhits_by_population.png", dpi=300)
            plt.close()
            system("open real_fakes_purity_vs_nhits_by_population.png")
                        
        else:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            data = np.atleast_2d(data)
            vals = data[:, 0]
            chi2_sr = data[:, 1]
            chi2_helix = data[:, 2]

        mask = (
            np.isfinite(vals)
            & np.isfinite(chi2_sr)
            & np.isfinite(chi2_helix)
            & (chi2_sr > 0)
            & (chi2_helix > 0)
        )

        results[label] = {
            "vals": vals[mask],
            "chi2_sr": chi2_sr[mask],
            "chi2_helix": chi2_helix[mask],
        }
            

# -------------------------
# Histogram
# -------------------------
plt.figure(figsize=(8, 5))

for label, d in results.items():
    vals = d["vals"]
    mean = np.mean(vals)
    median = np.median(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    plt.hist(
        vals,
        bins=10,
        histtype="step",
        linewidth=2,
        label=f"{label}: mean={mean:.2f}, med={median:.2f}, sd={sd:.2f}",
    )

plt.axvline(0, linestyle="--", linewidth=1)

if USE_K_FROM_HTML:
    xlabel = r"$\log(\chi^2_{\nu,\mathrm{SR},K}/\chi^2_{\nu,\mathrm{helix},K})$"
    title = "SR vs helix reduced chi-square ratio with parameter DOF correction"
    suffix = "with_K_from_html"
else:
    xlabel = r"$\log(\bar{\chi}^2_{\mathrm{SR}}/\bar{\chi}^2_{\mathrm{helix}})$"
    title = "SR vs helix goodness-of-fit ratio"
    suffix = "old"

plt.xlabel(xlabel)
plt.ylabel("Tracks")
plt.title(title)
plt.legend()
plt.tight_layout()
plt.savefig(f"combined_chi2_ratio_populations_{dataset_choice}_{suffix}.png", dpi=300)
system(f"open combined_chi2_ratio_populations_{dataset_choice}_{suffix}.png")
plt.close()


# -------------------------
# Scatter
# -------------------------
plt.figure(figsize=(6, 6))

all_sr = []
all_helix = []

for label, d in results.items():
    chi2_sr = d["chi2_sr"]
    chi2_helix = d["chi2_helix"]

    all_sr.append(chi2_sr)
    all_helix.append(chi2_helix)

    plt.scatter(chi2_helix, chi2_sr, label=label, alpha=0.75)

all_sr = np.concatenate(all_sr)
all_helix = np.concatenate(all_helix)

lo = min(all_sr.min(), all_helix.min())
hi = max(all_sr.max(), all_helix.max())

plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

plt.xscale("log")
plt.yscale("log")

if USE_K_FROM_HTML:
    plt.xlabel(r"$\chi^2_{\nu,\mathrm{helix},K}$")
    plt.ylabel(r"$\chi^2_{\nu,\mathrm{SR},K}$")
    plt.title("SR vs helix reduced chi-square with parameter DOF correction")
else:
    plt.xlabel(r"$\bar{\chi}^2_{\mathrm{helix}}$")
    plt.ylabel(r"$\bar{\chi}^2_{\mathrm{SR}}$")
    plt.title("SR vs helix per-observation chi-square")

plt.legend()
plt.tight_layout()
plt.savefig(f"combined_chi2_scatter_populations_loglog_{dataset_choice}_{suffix}.png", dpi=300)
system(f"open combined_chi2_scatter_populations_loglog_{dataset_choice}_{suffix}.png")
plt.close()
