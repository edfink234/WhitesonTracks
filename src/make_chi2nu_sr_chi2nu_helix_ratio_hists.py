import numpy as np
import matplotlib.pyplot as plt
from os import system

dataset_choice = "100_tracks_case"
files = {
    "10_tracks_case":
    {
        "SM": "../data_files/chi2_ratio_values_v20260407_091434__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel_using_Phi.csv",
        "5-mode": "../data_files/chi2_ratio_values_v20260430_092453__train5_test5__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.csv",
        "25-mode": "../data_files/chi2_ratio_values_v20260305_002410__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.csv",
        "Noise": "../data_files/chi2_ratio_values_v20260430_193549__train5_test5__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01__randomNoise.csv",
    },
    "100_tracks_case":
    {
        "25-mode": "../data_files/chi2_ratio_values_v20260518_142036__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.csv",
        "5-mode": "../data_files/chi2_ratio_values_v20260518_131139__train50_test50__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.csv",
        "SM": "../data_files/chi2_ratio_values_v20260518_142850__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel_using_Phi.csv",
        "Noise": "../data_files/chi2_ratio_values_v20260519_110640__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__randomNoise.csv"
    }
}[dataset_choice]

plt.figure(figsize=(8,5))

for label, path in files.items():
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    vals = np.atleast_2d(data)[:, 0]
    vals = vals[np.isfinite(vals)]

    mean = np.mean(vals)
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    plt.hist(vals, bins=10, histtype="step", linewidth=2,
             label=f"{label}: mean={mean:.2f}, sd={sd:.2f}")

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel(r"$\log(\chi^2_{\nu,\mathrm{SR}}/\chi^2_{\nu,\mathrm{helix}})$")
plt.ylabel("Tracks")
plt.title("SR vs helix goodness-of-fit ratio")
plt.legend()
plt.tight_layout()
plt.savefig(f"combined_chi2_ratio_populations_{dataset_choice}.png", dpi=300)
system(f"open combined_chi2_ratio_populations_{dataset_choice}.png")

plt.close()

plt.figure(figsize=(6,6))

all_sr = []
all_helix = []

for label, path in files.items():
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)

    chi2_sr = data[:, 1]
    chi2_helix = data[:, 2]

    mask = (
        np.isfinite(chi2_sr)
        & np.isfinite(chi2_helix)
        & (chi2_sr > 0)
        & (chi2_helix > 0)
    )

    chi2_sr = chi2_sr[mask]
    chi2_helix = chi2_helix[mask]

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
plt.xlabel(r"$\chi^2_{\nu,\mathrm{helix}}$")
plt.ylabel(r"$\chi^2_{\nu,\mathrm{SR}}$")
plt.title("SR vs helix reduced chi-square")
plt.legend()
plt.tight_layout()
plt.savefig(f"combined_chi2_scatter_populations_loglog_{dataset_choice}.png", dpi=300)
system(f"open combined_chi2_scatter_populations_loglog_{dataset_choice}.png")
