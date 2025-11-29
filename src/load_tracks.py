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


def load_single_track(csv_path):
    df = pd.read_csv(csv_path)

    r   = df["r"].values
    phi = df["phi"].values
    z   = df["z"].values

    x, y, z = cylindrical_to_cartesian(r, phi, z)
    s = compute_arc_length(x, y, z)

    return s, x, y, z


def load_many_tracks(folder_path, max_tracks=None, min_hits=6):
    files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    print(f"Found {len(files)} tracks.")

    S, X, Y, Z = [], [], [], []

    count = 0
    for f in files:
        print(f"Loading file {f}")
        try:
            s, x, y, z = load_single_track(f)

            if len(s) < min_hits:
                continue  # too short → useless for SR

            # Normalize arc length so s in [0, 1]
            s = s / s[-1]

            S.append(s)
            X.append(x)
            Y.append(y)
            Z.append(z)

            count += 1
            if max_tracks and count >= max_tracks:
                break

        except Exception as e:
            print(f"Skipping {f}: {e}")

    print(f"Loaded {len(S)} usable tracks.")
    return S, X, Y, Z
