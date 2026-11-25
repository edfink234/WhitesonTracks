#!/usr/bin/env python3

import numpy as np
import pandas as pd

SPLIT_CSV = (
    "../data_files/"
    "split_helix_ratio_values_mu20_finder_fakes_100__split_helix.csv"
)

SR_CSV = (
    "../data_files/"
    "chi2_ratio_values_mu20_finder_fakes_100.csv"
)

OUT_CSV = (
    "../data_files/"
    "chi2_ratio_values_mu20_finder_fakes_100_popB.csv"
)

LOG_FACTOR_TWO = np.log(2.0)


split_df = pd.read_csv(SPLIT_CSV)
sr_df = pd.read_csv(SR_CSV)

if len(split_df) != len(sr_df):
    raise RuntimeError(
        f"Row-count mismatch: "
        f"split={len(split_df)}, SR={len(sr_df)}"
    )


# Population B:
# split changes chi2 by less than a factor of 2
vals = split_df[
    "log_chi2_split_over_helix"
].to_numpy(float)

mask_B = (
    np.isfinite(vals)
    &
    (np.abs(vals) <= LOG_FACTOR_TWO)
)


sr_B = sr_df.loc[mask_B].copy()

print(f"Total fake rows: {len(sr_df)}")
print(f"Population B rows: {len(sr_B)}")

sr_B.to_csv(
    OUT_CSV,
    index=False,
)

print(f"Saved {OUT_CSV}")
