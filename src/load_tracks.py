import numpy as np
import pandas as pd
import glob
import os
from scipy.optimize import least_squares

def cylindrical_to_cartesian(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z

def compute_arc_length(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    s = np.insert(np.cumsum(ds), 0, 0.0)
    return s

def load_single_track(csv_path, param="arc"):
    df = pd.read_csv(csv_path)

    r   = df["r"].values
    phi = df["phi"].values
    z   = df["z"].values

    x, y, z = cylindrical_to_cartesian(r, phi, z)
    s = None
    if param == "index":
        s = np.arange(len(x), dtype=float) # <-- hit index parameterization
    elif param == "phi":
        # Reconstruct the intrinsic helix phase about the fitted circle center,
        # not the stored CSV phi = atan2(y, x) about the origin.
        xc0, yc0 = np.mean(x), np.mean(y)
        R0 = np.median(np.sqrt((x - xc0)**2 + (y - yc0)**2))

        def circle_resid(p):
            xc, yc, R = p
            return np.sqrt((x - xc)**2 + (y - yc)**2) - R

        res_xy = least_squares(circle_resid, [xc0, yc0, R0], loss="soft_l1", f_scale=1.0)
        xc_fit, yc_fit, R_fit = res_xy.x

        s = np.unwrap(np.arctan2(y - yc_fit, x - xc_fit)).astype(float)
        s = s - s[0]

        # Keep orientation consistent along the track
        if len(s) > 1 and s[-1] < 0:
            s = -s
    else:
        s = compute_arc_length(x, y, z)
        if s[-1] > 0:
            s = s / s[-1]
    # Optional sigmas
    sig_x = sig_y = sig_z = None
    if ("sigma_r" in df.columns) and ("sigma_phi" in df.columns) and ("sigma_z" in df.columns):
        sigma_r = df["sigma_r"].values
        sigma_phi = df["sigma_phi"].values
        sig_z = df["sigma_z"].values

        sig_x = np.sqrt((np.cos(phi)*sigma_r)**2 + ((-r*np.sin(phi))*sigma_phi)**2)
        sig_y = np.sqrt((np.sin(phi)*sigma_r)**2 + (( r*np.cos(phi))*sigma_phi)**2)

    return s, x, y, z, sig_x, sig_y, sig_z

def load_many_tracks(folder_path, max_tracks=None, min_hits=6, param = "arc"):
    print(f"folder_path = {folder_path}")
    files = sorted(glob.glob(os.path.join(folder_path, "*-hits.csv")))
    print(f"Found {len(files)} tracks.")

    S, X, Y, Z, F = [], [], [], [], []
    SIG_X, SIG_Y, SIG_Z = [], [], []

    count = 0
    for f in files:
        print(f"Loading file {f}")
        try:
            # load_single_track returns (s, x, y, z, sig_x, sig_y, sig_z, meta)
            s, x, y, z, sig_x, sig_y, sig_z = load_single_track(f, param=param)

            if len(s) < min_hits:
                continue  # too short → useless for SR

            # NOTE: don't renormalize s here — load_single_track already normalized t/phi/index
            S.append(s)
            X.append(x)
            Y.append(y)
            Z.append(z)
            SIG_X.append(sig_x)
            SIG_Y.append(sig_y)
            SIG_Z.append(sig_z)

            F.append(f[f.index("event"):-4])

            count += 1
            if max_tracks and count >= max_tracks:
                break

        except Exception as e:
            print(f"Skipping {f}: {e}")

    print(f"Loaded {len(S)} usable tracks.")
    # return sig lists as extra outputs
    return S, X, Y, Z, SIG_X, SIG_Y, SIG_Z, F

