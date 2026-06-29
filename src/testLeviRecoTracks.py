import pandas as pd
import numpy as np

df = pd.read_csv("~/Downloads/Train_SM+Schwartz_Set_19_Test_SM+Schwartz_Set_10.csv")

tracks = (
    df.groupby(["event_id", "track_id"])
      .agg(
          particle_id=("particle_id", "first"),
          n_shared=("n_shared", "first"),
          n_reco_hits=("n_reco_hits", "first"),
          n_true_hits=("n_true_hits", "first"),
          purity_reco=("purity_reco", "first"),
          eff_true=("eff_true", "first"),
          PID=("PID", "first"),
          n_rows=("hit_id", "size"),
      )
      .reset_index()
)

print(tracks.head())
print(tracks["purity_reco"].describe())
print(tracks["purity_reco"].value_counts().sort_index().tail())

print("num reco tracks:", len(tracks))
print("num fake-ish purity < 0.5:", (tracks["purity_reco"] < 0.5).sum())
print("num impure purity < 0.9:", (tracks["purity_reco"] < 0.9).sum())
print("num pure purity == 1:", (tracks["purity_reco"] == 1.0).sum())

# ------------------------------------------------------------
# Check how many low-purity reconstructed candidates are usable
# for SR fits after requiring a minimum number of hits.
# ------------------------------------------------------------

print("\nFake-ish candidates with purity_reco < 0.5:")
for min_hits in [25, 20, 15, 10, 5]:
    sel = tracks[
        (tracks["purity_reco"] < 0.5) &
        (tracks["n_reco_hits"] >= min_hits) &
        (tracks["n_rows"] >= min_hits)
    ]
    print(f"  min_hits >= {min_hits:2d}: {len(sel)} tracks")

print("\nImpure candidates with purity_reco < 0.9:")
for min_hits in [25, 20, 15, 10, 5]:
    sel = tracks[
        (tracks["purity_reco"] < 0.9) &
        (tracks["n_reco_hits"] >= min_hits) &
        (tracks["n_rows"] >= min_hits)
    ]
    print(f"  min_hits >= {min_hits:2d}: {len(sel)} tracks")

# ------------------------------------------------------------
# Recommended fake/noise selection for the paper:
# purity_reco < 0.5 and n_reco_hits >= 20, if enough survive.
# ------------------------------------------------------------

fake_candidates = tracks[
    (tracks["purity_reco"] < 0.5) &
    (tracks["n_reco_hits"] >= 20) &
    (tracks["n_rows"] >= 20)
].copy()

print("\nRecommended fake-candidate sample:")
print(fake_candidates.head())
print(fake_candidates["n_reco_hits"].describe())
print("\nPID counts:")
print(fake_candidates["PID"].value_counts())

if len(fake_candidates) >= 100:
    fake_sample_100 = fake_candidates.sample(n=100, random_state=123)
    print("\nSampled 100 fake candidates.")
else:
    fake_sample_100 = fake_candidates
    print(f"\nOnly {len(fake_candidates)} fake candidates available; using all of them.")

# Save the track-level list of selected fake candidates.
PATH = "~/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/True_Fakes_Levi_Train_SM+Schwartz_Set_19_Test_SM+Schwartz_Set_10/"
fake_sample_100.to_csv(f"{PATH}selected_fake_reco_tracks_tracklevel.csv", index=False)

# ------------------------------------------------------------
# Optional: recover the hit-level rows for those selected tracks.
# This is what you probably want to feed into your SR/helix fitter.
# ------------------------------------------------------------

selected_keys = fake_sample_100[["event_id", "track_id"]]

fake_hits_100 = df.merge(
    selected_keys,
    on=["event_id", "track_id"],
    how="inner"
)

from pathlib import Path

outdir = Path(PATH)
outdir.mkdir(parents=True, exist_ok=True)

fake_hits_100 = fake_hits_100.sort_values(["event_id", "track_id", "r"])
sigma_xyz = 0.01
eps = 1e-12

for (event_id, track_id), track_df in fake_hits_100.groupby(["event_id", "track_id"]):
    track_df = track_df.sort_values("r").copy()

    filename = f"fake_reco_event{event_id}_track{track_id}-hits.csv"
    # Add detector-resolution proxy expected by load_tracks.py.
    # This corresponds approximately to sigma_x = sigma_y = sigma_z = 0.01 cm.
    track_df["sigma_r"] = sigma_xyz
    track_df["sigma_phi"] = sigma_xyz / np.maximum(track_df["r"].to_numpy(float), eps)
    track_df["sigma_z"] = sigma_xyz

    filename = f"fake_reco_event{event_id}_track{track_id}-hits.csv"
    track_df.to_csv(outdir / filename, index=False)

print(f"Saved {fake_hits_100.groupby(['event_id', 'track_id']).ngroups} track CSVs to:")
print(outdir)
