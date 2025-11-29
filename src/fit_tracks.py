from load_tracks import load_many_tracks
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from pysr import PySRRegressor
from os import system

round_floats = lambda expr, ndigits: expr.xreplace({f: sp.Float(round(float(f), ndigits)) for f in expr.atoms(sp.Float)})

def sympy_to_latex_with_s(expr):
    """Return a LaTeX string with rounded floats and x0→s replacement."""
    # Round floats first
    expr = round_floats(expr, 3)

    # Convert to LaTeX
    expr_latex = sp.latex(expr)

    # Replace PySR variable name with s
    expr_latex = expr_latex.replace("x_{0}", "s").replace("x0", "s")

    return expr_latex

# Fix random seed
np.random.seed(0)

track_folder = "../tracks_for_ed"
S, X, Y, Z = load_many_tracks(track_folder, max_tracks=1)

s_all = np.concatenate(S)
x_all = np.concatenate(X)
y_all = np.concatenate(Y)
z_all = np.concatenate(Z)

model_kwargs = dict(
    niterations=100,
    binary_operators=["+", "-", "*"],
    unary_operators=["sin", "cos", "exp"],
    maxsize=10,
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    random_state=0,
    deterministic=True,
    procs=0,
    multithreading=False
)

model_x = PySRRegressor(**model_kwargs)
model_y = PySRRegressor(**model_kwargs)
model_z = PySRRegressor(**model_kwargs)

s_all_reshaped = s_all.reshape(-1, 1)

model_x.fit(s_all_reshaped, x_all)
eq_x = r"$x(s) = " + sympy_to_latex_with_s(model_x.sympy()) + "$"

model_y.fit(s_all_reshaped, y_all)
eq_y = r"$y(s) = " + sympy_to_latex_with_s(model_y.sympy()) + "$"

model_z.fit(s_all_reshaped, z_all)
eq_z = r"$z(s) = " + sympy_to_latex_with_s(model_z.sympy()) + "$"

print("x(s):", eq_x)
print("y(s):", eq_y)
print("z(s):", eq_z)

# R^2 for each fit
r2_x = model_x.score(s_all_reshaped, x_all)
r2_y = model_y.score(s_all_reshaped, y_all)
r2_z = model_z.score(s_all_reshaped, z_all)

# Nice dense grid for smooth curves
s_plot = np.linspace(s_all.min(), s_all.max(), 500).reshape(-1, 1)

x_pred = model_x.predict(s_plot)
y_pred = model_y.predict(s_plot)
z_pred = model_z.predict(s_plot)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# x(s)
axes[0].scatter(s_all, x_all, s=10, alpha=0.6, label="Data")
axes[0].plot(s_plot.ravel(), x_pred, linewidth=2,
             label=fr"Fit: $R^2={r2_x:.3f}$" + "\n" + eq_x)
axes[0].set_ylabel("$x$")
axes[0].legend(fontsize=8)

# y(s)
axes[1].scatter(s_all, y_all, s=10, alpha=0.6, label="Data")
axes[1].plot(s_plot.ravel(), y_pred, linewidth=2,
             label=fr"Fit: $R^2={r2_y:.3f}$" + "\n" + eq_y)
axes[1].set_ylabel("$y$")
axes[1].legend(fontsize=8)

# z(s)
axes[2].scatter(s_all, z_all, s=10, alpha=0.6, label="Data")
axes[2].plot(s_plot.ravel(), z_pred, linewidth=2,
             label=fr"Fit: $R^2={r2_z:.3f}$" + "\n" + eq_z)
axes[2].set_ylabel("$z$")
axes[2].set_xlabel("$s$")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("SR_track_event100000001-hit.pdf")
system("open SR_track_event100000001-hit.pdf")
