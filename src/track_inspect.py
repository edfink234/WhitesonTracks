import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import make_tracks_60 as mt
from os import system

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

# Add the directory containing make_tracks_60.py
sys.path.append("/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/src")
# Plot or show
show = False
open = not show and True
# Load your hits file
hits_path = "/Users/edwardfinkelstein/SDSU_UCI/WhitesonResearch/TrackProject/tracks_for_ed/v20260209_125904__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01/event100000010-hits.csv"
file_pref = hits_path[hits_path.index("event"):-9]
csv_file = hits_path[hits_path.index("tracks_for_ed")+14:hits_path.index("__train")]
df = pd.read_csv(hits_path)
print(df)
plt.scatter(df["r"], df["z"])
plt.xlabel("r")
plt.ylabel("z")
plt.title(f"{file_pref} $r$ vs $z$ (from CSV, {csv_file})")
if show:
    plt.show()
else:
    plt.savefig(filename:="event100000010-hits_r_z.png", dpi=5*96)
plt.close()
if open:
    system(f"open {filename}")

# Set globals expected by Levi's plotting function
mt.plot_title = f"{file_pref} hits (from CSV, {csv_file})"
mt.plotting_save_file = f"{file_pref}_hits_3d.png"

mt.make_track_plot([df], num_plotted_samples=1, show = show)

print(f"Saved: {mt.plotting_save_file}" if not show else "Showed")
plt.close()
if open:
    system(f"open {mt.plotting_save_file}")

r   = df["r"].to_numpy(dtype=float)
phi = df["phi"].to_numpy(dtype=float)
z_cyl = df["z"].to_numpy(dtype=float)

x, y, z = cylindrical_to_cartesian(r, phi, z_cyl)

s = compute_arc_length(x, y, z)
if s[-1] > 0:
    s = s / s[-1]  # normalize to [0, 1]

out_png = f"{file_pref}_xyz_vs_s.png"
title = f"{file_pref}: x(s), y(s), z(s) with normalized arclength s (from CSV, {csv_file})"

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
fig.suptitle(title)

axes[0].plot(s, x, marker="o", linewidth=1)
axes[0].set_ylabel("x")

axes[1].plot(s, y, marker="o", linewidth=1)
axes[1].set_ylabel("y")

axes[2].plot(s, z, marker="o", linewidth=1)
axes[2].set_ylabel("z")
axes[2].set_xlabel("s (normalized arc-length)")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

if show:
    plt.show()
    print("Showed")
else:
    plt.savefig(out_png, dpi=5*96, bbox_inches="tight")
    print(f"Saved: {out_png}")

if open:
    system(f"open {out_png}")
