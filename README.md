Analysis framework accompanying *Fitting and Filtering Unanticipated Particle Tracks*.

This repository contains the code and data products used to generate track samples, fit symbolic templates to tracks using a separately maintained symbolic-regression engine, evaluate classification performance, and reproduce the figures presented in the paper.

The symbolic-regression engine itself is **not included in this repository**. It is available separately:

* Paper: https://arxiv.org/abs/2410.08137
* Source: https://github.com/edfink234/Alpha-Zero-Symbolic-Regression/tree/PrefixPostfixSymbolicDifferentiator

---

## Repository Structure

```text
TrackProject/
├── src/
├── data_files/
├── plot_4_of_tracks/
├── stubborn_track_csvs/
├── pngs/
├── tracks_for_ed/
├── EquationTemplatesChosenSR.pdf
└── README.md
```

### `src/`

Primary analysis scripts.

| File                                         | Description                                                           |
| -------------------------------------------- | --------------------------------------------------------------------- |
| `make_tracks_60.py`                          | Generates synthetic Fourier, helical, and random-noise track samples. |
| `fit_tracks.py`                              | Main symbolic-template fitting and evaluation pipeline.               |
| `load_tracks.py`                             | Utility functions for reading generated tracks.                       |
| `template_dag_jaccard.py`                    | Computes template DAG similarities and Jaccard similarity matrices.   |
| `make_chi2nu_sr_chi2nu_helix_ratio_hists.py` | Produces SR-vs-helix comparison plots.                                |
| `track_inspect.py`                           | Utilities for inspecting individual tracks and fits.                  |
| `track_templates.pkl`                        | Serialized symbolic template library.                                 |

### `data_files/`

Intermediate and final data products used in the paper.

Includes:

* SR-vs-helix fit-quality ratios
* Template fitting timing studies
* DAG similarity matrices
* Figure-generation CSV files

### `plot_4_of_tracks/`

Representative track visualizations.

### `stubborn_track_csvs/`

Tracks and template fits that were useful during debugging and symbolic-continuation studies.

### `pngs/`

Additional generated figures and visualizations.

---

## Reproducing Main Results

### Generate Tracks

```bash
python make_tracks_60.py
```

### Fit Tracks

```bash
python fit_tracks.py
```

### Compute Template Similarities

```bash
python template_dag_jaccard.py
```

### Produce SR-vs-Helix Comparison Figures

```bash
python make_chi2nu_sr_chi2nu_helix_ratio_hists.py
```

---

## Main Figures

The following figures appearing in the paper are generated from the contents of `data_files/`:

* SR vs Helix fit-quality ratio distributions
* SR vs Helix scatter plots
* DAG similarity heatmaps
* Cumulative template coverage plots

---

## Notes

The symbolic template library was obtained through symbolic continuation and consists of:

* 35 templates originating from PySR initialization searches (`maxsize=9`)
* 16 templates originating from the joint-optimization gradient-boosting symbolic-regression framework

for a total library size of 51 templates.

The templates themselves are stored in:

```text
src/track_templates.pkl
```

---

## Citation

If using this repository, please cite:

```bibtex
@article{finkelstein2024generalizedfixeddepthprefixpostfix,
  title={Generalized Fixed-Depth Prefix and Postfix Symbolic Regression Grammars},
  author={Finkelstein, Edward},
  year={2024},
  eprint={2410.08137},
  archivePrefix={arXiv},
  primaryClass={cs.SC}
}
```
