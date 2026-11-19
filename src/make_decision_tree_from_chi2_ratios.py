#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)


SPLIT_FILES = {
    "SM":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_SM_100__split_helix.csv",

    "PID15":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_PID15_100__split_helix.csv",

    "Background-only fakes":
        "../data_files/"
        "split_helix_ratio_values_mu20_finder_fakes_100__split_helix.csv",
}

SR_FILES = {
    "SM":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_SM_100.csv",

    "PID15":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_PID15_100.csv",

    "Background-only fakes":
        "../data_files/"
        "chi2_ratio_values_mu20_finder_fakes_100.csv",
}


CLASS_NAMES = [
    "SM",
    "PID15",
    "Background-only fakes",
]


# ------------------------------------------------------------
# Build feature matrix
#
# feature 0:
#   log(chi2_split / chi2_helix)
#
# feature 1:
#   log(chi2_SR / chi2_helix)
# ------------------------------------------------------------

X_all = []
helix_all = []
y_all = []

for class_idx, label in enumerate(CLASS_NAMES):

    split_df = pd.read_csv(SPLIT_FILES[label])
    sr_df = pd.read_csv(SR_FILES[label])

    if len(split_df) != len(sr_df):
        raise RuntimeError(
            f"{label}: split has {len(split_df)} rows, "
            f"SR has {len(sr_df)} rows."
        )

    split_ratio = split_df[
        "log_chi2_split_over_helix"
    ].to_numpy(float)

    sr_ratio = sr_df[
        "log_chi2_sr_over_helix"
    ].to_numpy(float)
    
    helix_chi2_split = split_df[
        "chi2nu_helix"
    ].to_numpy(float)

    helix_chi2_sr = sr_df[
        "chi2nu_helix"
    ].to_numpy(float)

    # These should be the same helix fit for the same tracks.
    if not np.allclose(
        helix_chi2_split,
        helix_chi2_sr,
        rtol=1e-8,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{label}: helix chi2 values differ "
            "between split and SR CSVs."
        )

    helix_chi2 = helix_chi2_split

    mask = (
        np.isfinite(split_ratio)
        & np.isfinite(sr_ratio)
        & np.isfinite(helix_chi2)
        & (helix_chi2 > 0)
    )

    X_class = np.column_stack([
        split_ratio[mask],
        sr_ratio[mask],
    ])

    y_class = np.full(
        len(X_class),
        class_idx,
        dtype=int,
    )

    X_all.append(X_class)
    helix_all.append(
        np.log(helix_chi2[mask])
    )
    y_all.append(y_class)


X = np.vstack(X_all)
y = np.concatenate(y_all)
X_helix = np.concatenate(
    helix_all
).reshape(-1, 1)
X_helix_split = np.column_stack([
    X_helix[:, 0],
    X[:, 0],
])
X_helix_split_sr = np.column_stack([
    X_helix[:, 0],
    X[:, 0],
    X[:, 1],
])

print("X shape:", X.shape)
print("class counts:", np.bincount(y))


# ------------------------------------------------------------
# Shallow, interpretable decision tree
# ------------------------------------------------------------

clf = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=0,
)


# ------------------------------------------------------------
# Cross-validated predictions
#
# Every track is predicted by a model that did NOT train on it.
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

# ------------------------------------------------------------
# Feature ablation study
# ------------------------------------------------------------

FEATURE_SETS = {
    "helix only": X_helix,

    "split/helix only":
        X[:, [0]],

    "SR/helix only":
        X[:, [1]],
        
    "helix + split":
        X_helix_split,

    "split + SR":
        X,
    
    "helix + split + SR":
        X_helix_split_sr,
}

print("\n" + "=" * 70)
print("FEATURE ABLATION")
print("=" * 70)

ablation_results = {}

for name, X_use in FEATURE_SETS.items():

    clf_ablation = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=0,
    )

    pred = cross_val_predict(
        clf_ablation,
        X_use,
        y,
        cv=cv,
    )

    cm_ablation = confusion_matrix(
        y,
        pred,
        labels=[0, 1, 2],
    )

    acc = accuracy_score(
        y,
        pred,
    )

    bal_acc = balanced_accuracy_score(
        y,
        pred,
    )

    correct = int(
        np.trace(cm_ablation)
    )

    ablation_results[name] = {
        "correct": correct,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cm": cm_ablation,
    }

    print(f"\n=== {name} ===")
    print(cm_ablation)

    print(
        f"correct: {correct}/{len(y)}"
    )

    print(
        f"accuracy: "
        f"{acc:.4f}"
    )

    print(
        f"balanced accuracy: "
        f"{bal_acc:.4f}"
    )

y_pred = cross_val_predict(
    clf,
    X,
    y,
    cv=cv,
)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1, 2],
)

print("\n=== CROSS-VALIDATED CONFUSION MATRIX ===")
print(cm)

print("\nCorrect:", np.trace(cm), "/", np.sum(cm))

print(
    "Accuracy:",
    accuracy_score(y, y_pred),
)

print(
    "Balanced accuracy:",
    balanced_accuracy_score(y, y_pred),
)

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        y,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
)


# ------------------------------------------------------------
# Plot confusion matrix
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

disp.plot(
    ax=ax,
    cmap=None,
    colorbar=False,
)

plt.title(
    "5-fold cross-validated classification\n"
    "using SR and split-helix discriminants"
)

plt.tight_layout()

out_cm = (
    "../pngs/"
    "decision_tree_confusion_matrix_mu20.png"
)

plt.savefig(
    out_cm,
    dpi=300,
)

plt.close()

print(f"\nSaved {out_cm}")


# ------------------------------------------------------------
# Fit one final tree on all 300 points ONLY so that we can
# inspect the learned thresholds.
#
# Do NOT use this training accuracy as the performance result.
# ------------------------------------------------------------

clf.fit(X, y)

print("\n=== FINAL TREE THRESHOLDS ===")

print(
    export_text(
        clf,
        feature_names=[
            "log_split_over_helix",
            "log_SR_over_helix",
        ],
    )
)


# ------------------------------------------------------------
# Plot final tree
# ------------------------------------------------------------

plt.figure(
    figsize=(14, 8)
)

plot_tree(
    clf,
    feature_names=[
        "log(split/helix)",
        "log(SR/helix)",
    ],
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    impurity=False,
    proportion=False,
)

plt.tight_layout()

out_tree = (
    "../pngs/"
    "decision_tree_thresholds_mu20.png"
)

plt.savefig(
    out_tree,
    dpi=300,
)

plt.close()

print(f"Saved {out_tree}")

print("\n=== ABLATION SUMMARY ===")

for name, r in ablation_results.items():
    print(
        f"{name:20s} "
        f"{r['correct']:3d}/{len(y)}  "
        f"acc={r['accuracy']:.3f}  "
        f"bal_acc={r['balanced_accuracy']:.3f}"
    )
