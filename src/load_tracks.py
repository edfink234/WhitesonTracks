import numpy as np
import pandas as pd
import glob
import os

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

def load_many_tracks(folder_path, max_tracks=None, min_hits=6):
    files = sorted(glob.glob(os.path.join(folder_path, "*-hits.csv")))
    print(f"Found {len(files)} tracks.")

    S, X, Y, Z, F = [], [], [], [], []
    SIG_X, SIG_Y, SIG_Z = [], [], []

    count = 0
    for f in files:
        print(f"Loading file {f}")
        try:
            # load_single_track returns (s, x, y, z, sig_x, sig_y, sig_z, meta)
            s, x, y, z, sig_x, sig_y, sig_z = load_single_track(f, param="arc")

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

