from load_tracks import load_many_tracks
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from os import system
from scipy.optimize import least_squares
from collections import defaultdict
import pickle
import os
from os import system
import json
import re
import pandas as pd

def parse_event_number(f_tag: str) -> int:
    """
    Accepts strings like:
      'event100000001-hits'
      'event100000001'
      '.../event100000001-hits.csv'
    Returns the full event number as int, e.g. 100000001.
    """
    m = re.search(r"event(\d+)", str(f_tag))
    if not m:
        raise ValueError(f"Could not parse event number from: {f_tag}")
    return int(m.group(1))

round_floats = lambda expr, ndigits: expr.xreplace({f: sp.Float(round(float(f), ndigits)) for f in expr.atoms(sp.Float)})
sech = lambda x: 1/np.cosh(x)

def template_signature(template_or_expr, round_digits=8):
    """
    Return a canonical structural signature string for a sympy expression or parametric template.
    Uses rounding to reduce floating-point fractional differences, then srepr(simplified_expr).
    """
    if isinstance(template_or_expr, dict):
        expr = template_or_expr["expr"]
    else:
        expr = template_or_expr

    # Round floats (helper defined earlier)
    expr_rounded = round_floats(expr, round_digits)

    try:
        expr_simpl = sp.simplify(expr_rounded)
    except Exception:
        expr_simpl = expr_rounded

    # srepr gives a structural representation (stable for equivalence checks)
    return sp.srepr(expr_simpl)

def sci_to_latex(sci_str):
    coeff, exp = sci_str.lower().split('e')
    return rf"{coeff} \times 10^{{{int(exp)}}}"

def regression_metrics(y_true, y_pred, *, sigma=None, ddof=0, eps=1e-12):
    """
    Returns dict with R2, MSE, MAE, chi2, chi2_red.

    chi2:
      - If sigma is provided: sum(((y - yhat)/sigma)**2)
      - Else: Pearson-style: sum((y - yhat)^2 / max(|yhat|, eps))
        (common fallback when per-point uncertainties are unknown)

    chi2_red uses (N - ddof) in denominator if possible.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape

    resid = y_true - y_pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    MSE = np.mean(resid**2)
    MAE = np.mean(np.abs(resid))

    if sigma is not None:
        sigma = np.asarray(sigma).ravel()
        if sigma.shape != y_true.shape:
            raise ValueError(f"sigma shape {sigma.shape} must match y shape {y_true.shape}")
        chi2 = np.sum((resid / sigma)**2)
    else:
        denom = np.maximum(np.abs(y_pred), eps)
        chi2 = np.sum((resid**2) / denom)

    dof = max(len(y_true) - int(ddof), 1)
    chi2_red = chi2 / dof

    return {"R2": R2, "MSE": MSE, "MAE": MAE, "chi2": chi2, "chi2_red": chi2_red}


def r2_score_1d(y_true, y_pred):
    """
    Compute R^2 = 1 - SS_res / SS_tot for 1D numpy arrays.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    assert y_true.shape == y_pred.shape

    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    # Handle the degenerate case y_true = constant
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - ss_res / ss_tot
    
def load_sigmas_for_event(track_folder: str, event_num: int):
    """
    Read event{event_num}-hits.csv and return (sig_x, sig_y, sig_z, n_hits).

    Backwards compatible:
      - If sigma columns are missing, returns (None, None, None, n_hits).
      - If file missing, raises FileNotFoundError.
    """
    hits_path = os.path.join(track_folder, f"event{int(event_num)}-hits.csv")
    if not os.path.exists(hits_path):
        raise FileNotFoundError(f"Hits CSV not found: {hits_path}")

    df = pd.read_csv(hits_path)

    # Always need r, phi to compute sigma_x/y (if sigmas exist)
    if ("r" not in df.columns) or ("phi" not in df.columns):
        raise ValueError(f"Required columns 'r' and 'phi' missing from {hits_path}")

    n_hits = len(df)

    # If no sigma columns (old noiseless datasets), just return None sigmas.
    required_sigma_cols = {"sigma_r", "sigma_phi", "sigma_z"}
    if not required_sigma_cols.issubset(df.columns):
        return None, None, None, n_hits

    r   = df["r"].to_numpy(dtype=float)
    phi = df["phi"].to_numpy(dtype=float)
    sr  = df["sigma_r"].to_numpy(dtype=float)
    sph = df["sigma_phi"].to_numpy(dtype=float)
    sz  = df["sigma_z"].to_numpy(dtype=float)

    c = np.cos(phi)
    s = np.sin(phi)

    # Error propagation: x = r cos φ, y = r sin φ
    sig_x = np.sqrt((c * sr)**2 + ((r * s) * sph)**2)
    sig_y = np.sqrt((s * sr)**2 + ((r * c) * sph)**2)
    sig_z = sz

    return sig_x, sig_y, sig_z, n_hits

def fit_until_both_conditions(s, y, model_kwargs, R2_THRESHOLD, min_iters=100, step=50, max_iters=np.inf, stop_flag_path="STOP_PYSR"):
    """
    Continue training PySR until BOTH:
      - R^2 >= R2_THRESHOLD
      - total iterations >= min_iters
    Training stops if max_iters is reached.
    """

    model = PySRRegressor(**model_kwargs)
    total_iters = 0
    best_R2 = 0
    R2 = best_R2

    while True:
        # Run more iterations *without resetting the model*
        model.niterations = total_iters + step
        model.fit(s, y)
        total_iters += step

        # Compute R^2
        R2 = model.score(s, y)
        if R2 > best_R2:
            best_R2 = R2
            print(f"New best R2 = {R2:.3f}")

        # --- stopping conditions ---

        # (1) Desired accuracy + min_iters satisfied
        if total_iters >= min_iters and best_R2 >= R2_THRESHOLD:
            print(f"[fit_until_both_conditions] Reached R^2 >= {R2_THRESHOLD} with {total_iters} iterations.")
            break

        # (2) Hard max iterations
        if total_iters >= max_iters:
            print(f"[fit_until_both_conditions] Reached max_iters={max_iters} with best R^2={best_R2:.3f}.")
            break

        # (3) External stop flag
        if os.path.exists(stop_flag_path):
            print(f"[fit_until_both_conditions] Stop flag '{stop_flag_path}' detected; aborting this fit.")
            # Remove the flag so future dimensions/tracks can run fresh
            os.remove(stop_flag_path)
            break

    system("rm *hall_of_fame*")
    return model, best_R2, total_iters

def sympy_to_latex_with_s(expr):
    """Return a LaTeX string with rounded floats and x0→s replacement."""
    # Round floats first
    expr = round_floats(expr, 3)

    # Convert to LaTeX
    expr_latex = sp.latex(expr)

    # Replace PySR variable name with s
    expr_latex = expr_latex.replace("x_{0}", "s").replace("x0", "s")

    return expr_latex

def template_to_latex_with_s(template_or_expr):
    expr = None
    """Convert a parametric template (dict) or a sympy expr to LaTeX, replacing x0→s."""
    if isinstance(template_or_expr, dict):
        expr = template_or_expr["expr"]
    else:
        expr = template_or_expr

    import sympy as sp
    expr_latex = sp.latex(expr)
    expr_latex = expr_latex.replace("x_{0}", "s").replace("x0", "s")
    return expr_latex

def make_parametric_template(expr, s_name="x0"):
    """
    Take a SymPy expression with numeric floats and one input symbol (x0),
    and return a parametric template where each float is replaced by a_i.
    """
    # Identify numeric constants (SymPy Floats)
    floats = list(expr.atoms(sp.Float))
    floats_sorted = sorted(floats, key=float)  # for reproducibility

    param_syms = []
    replacements = {}

    for i, f in enumerate(floats_sorted):
        a_i = sp.Symbol(f"a{i}")
        param_syms.append(a_i)
        replacements[f] = a_i

    expr_param = expr.xreplace(replacements)

    # Identify the s symbol (input variable) – assume it's named s_name, e.g. "x0"
    s_sym_candidates = [sym for sym in expr_param.free_symbols if sym.name == s_name]
    if len(s_sym_candidates) != 1:
        raise ValueError(f"Expected exactly one symbol named {s_name} in expression.")
    s_sym = s_sym_candidates[0]

    init_params = np.array([float(f) for f in floats_sorted], dtype=float)

    return {
        "expr": expr_param,
        "s_sym": s_sym,
        "param_syms": param_syms,
        "init_params": init_params,
    }

def fit_template_to_data(template, s_data, y_data, *, sigma=None):
    """
    Optimize parameters and return:
      (metrics_dict, best_params, expr_fitted, y_pred)

    metrics_dict has: R2, MSE, MAE, chi2, chi2_red
    """
    expr_param = template["expr"]
    s_sym      = template["s_sym"]
    param_syms = template["param_syms"]
    p0         = template["init_params"]

    s_data = np.asarray(s_data).ravel()
    y_data = np.asarray(y_data).ravel()

    # Build f(s, *a)
    f = sp.lambdify((s_sym, *param_syms), expr_param, "numpy")
    
    BIG = 1e6  # penalty magnitude for invalid parameter regions
    
    def residuals(p):
        with np.errstate(all="ignore"):
            y_pred = f(s_data, *p)

        # Force finite residuals even if model blows up
        if (not np.all(np.isfinite(y_pred))) or np.any(np.iscomplex(y_pred)):
            return np.full_like(y_data, BIG, dtype=float)

        r = (y_pred - y_data).astype(float)

        if sigma is not None:
            sig = np.asarray(sigma).ravel()
            if sig.shape != y_data.shape:
                raise ValueError(f"sigma shape {sig.shape} must match y shape {y_data.shape}")
            r = r / sig

        # Guard against sigma producing inf/nan too
        if not np.all(np.isfinite(r)):
            return np.full_like(y_data, BIG, dtype=float)

        return r

    # Nonlinear least squares (robust loss helps too)
    res = least_squares(residuals, p0, loss="soft_l1")

    p_opt = res.x
    y_pred = f(s_data, *p_opt)

    # ddof: number of fitted parameters (for reduced chi2)
    metrics = regression_metrics(y_data, y_pred, sigma=sigma, ddof=len(p_opt))

    subs_dict = {sym: val for sym, val in zip(param_syms, p_opt)}
    expr_fitted = expr_param.subs(subs_dict)

    return metrics, p_opt, expr_fitted, y_pred

def summarize_families(coord_name, templates, families, r2_list):
    """
    Print text summary + emit LaTeX table and LaTeX equation list
    for a given coordinate (x, y, or z).
    """
    # Group R^2 by family
    stats = defaultdict(lambda: {"count": 0, "r2": []})
    for i, fam in enumerate(families):
        stats[fam]["count"] += 1
        stats[fam]["r2"].append(r2_list[i])

    print(f"\n=== Summary for {coord_name}(s) ===")
    print(f"Number of template families: {len(templates)}")

    for fam_id in sorted(stats.keys()):
        arr = np.array(stats[fam_id]["r2"])
        mean_r2 = arr.mean()
        std_r2 = arr.std()
        print(
            f"  Family {fam_id}: {stats[fam_id]['count']} tracks, "
            f"mean R^2 = {mean_r2:.3f} ± {std_r2:.3f}"
        )

    # ---- LaTeX table ----
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{c c c}")
    lines.append(r"\hline")
    lines.append(r"Family & \#~Tracks & Mean $R^2$ \\")
    lines.append(r"\hline")
    for fam_id in sorted(stats.keys()):
        arr = np.array(stats[fam_id]["r2"])
        mean_r2 = arr.mean()
        lines.append(fr"{fam_id} & {stats[fam_id]['count']} & {mean_r2:.3f} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(fr"\caption{{Summary of {coord_name}(s) template families.}}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    # ---- LaTeX equations for template forms ----
    eq_lines = []
    eq_lines.append(r"\begin{align*}")
    for fam_id, template in enumerate(templates):
        eq_latex = template_to_latex_with_s(template)
        eq_lines.append(
            fr"{coord_name}_{{\text{{family}}\,{fam_id}}}(s) &= {eq_latex} \\"
        )
    eq_lines.append(r"\end{align*}")

    latex_eqns = "\n".join(eq_lines)

    return latex_table, latex_eqns

def summarize_3d_families(families_x, families_y, families_z, r2_x_all, r2_y_all, r2_z_all):
    """
    Group tracks by 3D family triples (Fx, Fy, Fz),
    print text summary, and return LaTeX table string.
    """
    assert len(families_x) == len(families_y) == len(families_z) == len(r2_x_all) == len(r2_y_all) == len(r2_z_all), \
        "All family and R^2 arrays must have the same length."

    n_tracks = len(families_x)

    # Collect stats per triple
    triple_stats = defaultdict(lambda: {
        "count": 0,
        "r2x": [],
        "r2y": [],
        "r2z": [],
        "indices": []  # track indices (optional, for debugging)
    })

    for i in range(n_tracks):
        triple = (families_x[i], families_y[i], families_z[i])
        triple_stats[triple]["count"]  += 1
        triple_stats[triple]["r2x"].append(r2_x_all[i])
        triple_stats[triple]["r2y"].append(r2_y_all[i])
        triple_stats[triple]["r2z"].append(r2_z_all[i])
        triple_stats[triple]["indices"].append(i)

    # Sort triples in a deterministic way
    unique_triples = sorted(triple_stats.keys())

    # Assign a simple integer ID to each triple for nicer reporting
    triple_id_map = {triple: k for k, triple in enumerate(unique_triples)}

    # -------- LaTeX table --------
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{c c c c c c c c}")
    lines.append(r"\hline")
    lines.append(r"Family & $F_x$ & $F_y$ & $F_z$ & \#~Tracks & Mean $R_x^2$ & Mean $R_y^2$ & Mean $R_z^2$ \\")
    lines.append(r"\hline")

    for triple in unique_triples:
        fid = triple_id_map[triple]
        data = triple_stats[triple]
        r2x_arr = np.array(data["r2x"])
        r2y_arr = np.array(data["r2y"])
        r2z_arr = np.array(data["r2z"])

        mean_r2x = r2x_arr.mean()
        mean_r2y = r2y_arr.mean()
        mean_r2z = r2z_arr.mean()

        lines.append(
            fr"{fid} & {triple[0]} & {triple[1]} & {triple[2]} & "
            fr"{data['count']} & {mean_r2x:.3f} & {mean_r2y:.3f} & {mean_r2z:.3f} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary of 3D track families, with indices $(F_x, F_y, F_z)$ "
                 r"referring to the 1D template families for $x(s)$, $y(s)$, and $z(s)$.}")
    lines.append(r"\end{table}")

    latex_table_3d = "\n".join(lines)

    return latex_table_3d, triple_id_map

if __name__ == '__main__':
    # Fix random seed
    np.random.seed(0)
    np.sech = lambda x: 1/np.cosh(x)
    OPEN_PNGS = False
    OPEN_HTML = True
    create_dataset = False
    TEMPLATE_PATH = "track_templates.pkl"
    track_folder = ["../tracks_for_ed/v20260122_163839__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01/", "../tracks_for_ed/v1_noiseless_69000/"][1]
    R2_THRESHOLD = 0.997
    RunPySR = False   # <-- set False to disable PySR discovery entirely
    MaxPySRIters = 100
    num_tracks = 100
    
    loaded = {}
    x_templates, y_templates, z_templates = [], [], []
    
    if os.path.exists(TEMPLATE_PATH):
        print("Loading existing template library...")
        with open(TEMPLATE_PATH, "rb") as f:
            loaded = pickle.load(f)

        x_templates = loaded["x_templates"]
        y_templates = loaded["y_templates"]
        z_templates = loaded["z_templates"]

    else:
        print("No template file found — starting with empty template lists.")
    
    if not loaded:
        # ------------------------------------------------------------------
        # Seed templates from previously discovered families (second run)
        # ------------------------------------------------------------------
        s = sp.Symbol("s")  # we'll use 's' as the parameter and pass s_name="s"
        
        # ---------- Family 0 ----------
        expr0_x = -360.982 * (0.589 - sp.tanh(s)) * sp.tanh(s)
        expr0_y = 57.471 * s * sp.sin(sp.exp(s))
        expr0_z = 1.234 * sp.exp(sp.exp(sp.sin(8.9 * sp.sqrt(sp.exp(-s)))))

        # ---------- Family 1 ----------
        expr1_x = -151.324 * (0.888 - sp.tanh(s)) * sp.tanh(s)
        expr1_y = -54.3 * s * sp.tanh(s + s) + 0.828
        expr1_z = (-57.035 * s - 85.784) * sp.tanh(s)

        # ---------- Family 2 ----------
        expr2_x = -370.298 * (0.689 - sp.tanh(s)) * sp.tanh(s)
        expr2_y = 40.612 * s * sp.cos(6.177 * s) + 6.676
        expr2_z = -28.139 * s * sp.exp(sp.sin(11.717 * sp.sqrt(s)))

        # ---------- Family 3 ----------
        expr3_x = -sp.sinh(2.394 * s + 1.852)
        expr3_y = 22.687 * sp.exp(sp.sin(s + sp.sin(s))) - 19.796
        expr3_z = (33.104 - 36.305 * s) * sp.tanh(s)

        # ---------- Family 4 ----------
        expr4_x = 4.877 * sp.sinh(sp.sin(s + sp.exp(s) + 5.836))
        expr4_y = 48.752 * s * sp.tanh(s + s) + 7.586
        expr4_z = -8.01 * sp.sin(sp.sin(2.691 * s)) - 1.697

        # Build parametric templates (replace floats with a0, a1, ...)
        first_run_x_templates = [
            make_parametric_template(expr0_x, s_name="s"),
            make_parametric_template(expr1_x, s_name="s"),
            make_parametric_template(expr2_x, s_name="s"),
            make_parametric_template(expr3_x, s_name="s"),
            make_parametric_template(expr4_x, s_name="s"),
        ]

        first_run_y_templates = [
            make_parametric_template(expr0_y, s_name="s"),
            make_parametric_template(expr1_y, s_name="s"),
            make_parametric_template(expr2_y, s_name="s"),
            make_parametric_template(expr3_y, s_name="s"),
            make_parametric_template(expr4_y, s_name="s"),
        ]

        first_run_z_templates = [
            make_parametric_template(expr0_z, s_name="s"),
            make_parametric_template(expr1_z, s_name="s"),
            make_parametric_template(expr2_z, s_name="s"),
            make_parametric_template(expr3_z, s_name="s"),
            make_parametric_template(expr4_z, s_name="s"),
        ]

        x_templates.extend(first_run_x_templates)
        y_templates.extend(first_run_y_templates)
        z_templates.extend(first_run_z_templates)

        # ----- Family 0 -----
        expr0_x = 2.628 - 37.513*sp.sin(4.603*s)
        expr0_y = 22.332*sp.exp(sp.sin(2.207*s)) - 24.108
        expr0_z = 7.739*sp.exp(sp.sin(5.757*s)) - 1.917

        # ----- Family 1 -----
        expr1_x = -23.895*sp.sin(2.88*s) - 6.149
        expr1_y = 53.5 - 49.255*sp.exp(sp.sin(0.846*s))
        expr1_z = -109.225*s*sp.cos(sp.sin(s) - 0.721)

        # ----- Family 2 -----
        expr2_x = 3.977*sp.sin(3.74*s) + 1.945
        expr2_y = 32.506*sp.exp(sp.sin(1.173*s)) - 29.209
        expr2_z = s*(19.851*s - 23.241) - 1.718

        # ----- Family 3 -----
        expr3_x = -39.697*sp.sin(3.83*s) - 3.123
        expr3_y = 17.313 - 31.212*sp.sin(9.241*sp.cos(s))
        expr3_z = -23.95*s*sp.exp(-sp.sin(9.372*s))

        # ----- Family 4 -----
        expr4_x = -sp.exp(3.494*s) - 3.929
        expr4_y = 22.84*sp.exp(sp.sin(1.954*s)) - 19.851
        expr4_z = 45.462*s*sp.cos(sp.sin(s) + 13.347)

        # Build parametric templates (replace floats by a0, a1, ...)
        second_run_x_templates = [
            make_parametric_template(expr0_x, s_name="s"),
            make_parametric_template(expr1_x, s_name="s"),
            make_parametric_template(expr2_x, s_name="s"),
            make_parametric_template(expr3_x, s_name="s"),
            make_parametric_template(expr4_x, s_name="s"),
        ]

        second_run_y_templates = [
            make_parametric_template(expr0_y, s_name="s"),
            make_parametric_template(expr1_y, s_name="s"),
            make_parametric_template(expr2_y, s_name="s"),
            make_parametric_template(expr3_y, s_name="s"),
            make_parametric_template(expr4_y, s_name="s"),
        ]

        second_run_z_templates = [
            make_parametric_template(expr0_z, s_name="s"),
            make_parametric_template(expr1_z, s_name="s"),
            make_parametric_template(expr2_z, s_name="s"),
            make_parametric_template(expr3_z, s_name="s"),
            make_parametric_template(expr4_z, s_name="s"),
        ]
        
        x_templates.extend(second_run_x_templates)
        y_templates.extend(second_run_y_templates)
        z_templates.extend(second_run_z_templates)
        
        
        
        
        z_templates.append(make_parametric_template((((((0.508292 ** (-5.130599 + sp.cos((s * 7.306336)))) * (-0.622545 + sp.sech((-2.263896 - (-11.663861 * s))))) + sp.cos((4 ** (2.492631 - s)))) + ((0.987333 + s) ** -72.499292)) + sp.cos((sp.sin(s) / sp.exp(-3.515159)))), s_name="s"))
#        z_templates.append(make_parametric_template(-34.520199*sp.sech(s**(-1.45342) - 3.00504026595319), s_name="s"))
    track_infos = []  # store per-track info for HTML
    x_eqns, y_eqns, z_eqns = [], [], []   # store final numeric expressions used
    families_x, families_y, families_z = [], [], []  # indices of template used
    r2_x_all, r2_y_all, r2_z_all = [], [], [] # R^2 values for each track and coordinate

    S, X, Y, Z, F = load_many_tracks(track_folder, max_tracks=num_tracks)
    assert((len(S), len(X), len(Y), len(Z), len(F)) == (num_tracks, num_tracks, num_tracks, num_tracks, num_tracks))
#    print(f"len(S), len(X), len(Y), len(Z), len(F) = {(len(S), len(X), len(Y), len(Z), len(F))}")
    
    def create_stubborn_track_dataset(X, Y, file_path = "", latex_eqn = "", show = False, plot_func = None):
        stubborn_track_dataset = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
        assert(stubborn_track_dataset.shape[0] == X.shape[0])
        assert(stubborn_track_dataset.shape[0] == Y.shape[0])
        assert(stubborn_track_dataset.shape[1] == 2)
        if file_path:
            np.savetxt(file_path, stubborn_track_dataset, delimiter = ',')
            print(f"Data saved to {file_path}")
        else:
            print(stubborn_track_dataset)
        if show:
            plt.scatter(stubborn_track_dataset[:,0], stubborn_track_dataset[:, 1], label = "Data")
            if plot_func:
                s_min = stubborn_track_dataset[:,0].min()
                s_max = stubborn_track_dataset[:,0].max()
                s_vals = np.linspace(s_min, s_max, 1000)
                y_true = stubborn_track_dataset[:,1]
                y_pred = plot_func(stubborn_track_dataset[:,0])
                R2 = r2_score_1d(y_true, y_pred)
                plt.plot(s_vals, plot_func(s_vals), label = fr"Fit: $R^2={R2:.3f}$" + "\n" + latex_eqn)
                plt.xlabel("s")
                plt.ylabel("z")
                plt.legend()
            plt.show()
        exit()
    
    if create_dataset:
        base_path = "../stubborn_track_csvs"
        file_path = f"{base_path}/event100000003-hits_Z.csv"
        file_path = ""
#        plot_func = lambda x0: (-34.520199 * np.sech((np.sqrt(9.030267) - (x0 ** -1.453420))))
#        plot_func_eqn = r"$z(s) = - 34.52 \operatorname{sech}{\left(s^{-1.45} - 3.01 \right)}$"
        plot_func = lambda s: (((((0.508292 ** (-5.130599 + np.cos((s * 7.306336)))) * (-0.622545 + np.sech((-2.263896 - (-11.663861 * s))))) + np.cos((4 ** (2.492631 - s)))) + ((0.987333 + s) ** -72.499292)) + np.cos((np.sin(s) / np.exp(-3.515159))))
        plot_func_eqn = r"$z(s) = {0.51}^{\cos{\left(7.31 s \right)} - 5.13} \left(\operatorname{sech}{\left(11.66 s - 2.26 \right)} - 0.62\right) + \left(s + 0.99\right)^{-72.5} + \cos{\left(4^{2.49 - s} \right)} + \cos{\left(33.62 \sin{\left(s \right)} \right)}$"
        create_stubborn_track_dataset(S[2], Z[2], file_path = file_path, show  = True, plot_func = plot_func, latex_eqn = plot_func_eqn)
        

    model_kwargs = dict(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "tanh", "sqrt", "log", "sech(x) = 1/cosh(x)"],
        extra_sympy_mappings={"sech": lambda x: 1/sp.cosh(x)},
        maxsize=9,
        model_selection="accuracy",
        elementwise_loss="loss(x, y) = (x - y)^2",
        random_state=0,
        deterministic=True,
        procs=0,
        multithreading=False,
        delete_tempfiles=True,
        warm_start=True
    )
    from pysr import PySRRegressor

    for track_idx, (s_all, x_all, y_all, z_all, f_all) in enumerate(zip(S, X, Y, Z, F)):
        s_all_reshaped = s_all.reshape(-1, 1)

        # --- Helper inner function to handle one coordinate --- #
        def fit_dimension(s_data, y_data, templates, eqn_list, families_list, coord_name, *, sigma=None):
            print(f"\n--- Fitting Track {track_idx}, coordinate {coord_name} ---")

            best_metrics = None
            best_expr = None
            best_template_index = None

            # 1) Try existing templates
            if templates:
                for idx, template in enumerate(templates):
                    try:
                        metrics, p_opt, expr_fitted, _ = fit_template_to_data(template, s_data, y_data, sigma=sigma)
                    except ValueError as e:
                        # Often: "array must not contain infs or NaNs" from pathological templates
                        print(f"[Track {track_idx} {coord_name}] Skipping template {idx} due to ValueError: {e}")
                        continue

                    if (best_metrics is None) or (metrics["R2"] > best_metrics["R2"]):
                        best_metrics = metrics
                        best_expr = expr_fitted
                        best_template_index = idx

            # 2) If good enough, use best template
            if templates and best_metrics is not None and best_metrics["R2"] >= R2_THRESHOLD:
                print(f"[Track {track_idx} {coord_name}] Used existing template {best_template_index} "
                      f"with R^2={best_metrics['R2']:.3f}")
                eqn_list.append(best_expr)
                families_list.append(best_template_index)
                return best_expr, best_metrics

            # 3) Otherwise, run PySR to discover new form
            best_R2 = -np.inf if best_metrics is None else best_metrics["R2"]
            if not RunPySR:
                print(f"[Track {track_idx} {coord_name}] best_R2={best_R2:.3f} < {R2_THRESHOLD} "
                      f"but RunPySR=False, so skipping PySR and using best available template.")
                if best_expr is None:
                    raise RuntimeError(
                        f"No templates available (or all failed) for Track {track_idx} {coord_name}, "
                        f"and RunPySR=False. Can't produce a fit."
                    )
                eqn_list.append(best_expr)
                families_list.append(best_template_index)
                return best_expr, best_metrics
            print(f"best_R2 of {best_R2} is less than {R2_THRESHOLD}, running PYSR now.")

            model, R2_final, iters = fit_until_both_conditions(
                s_data.reshape(-1, 1),
                y_data,
                model_kwargs,
                R2_THRESHOLD,
                min_iters=100,
                step=50,
                max_iters=MaxPySRIters,
            )

            expr = model.sympy()

            template = make_parametric_template(expr, s_name="x0")
            templates.append(template)
            new_template_index = len(templates) - 1

            metrics_new, _, expr_fitted_new, _ = fit_template_to_data(template, s_data, y_data, sigma=sigma)
            
            # If templates existed and one of them was better, keep it instead of PySR
            if best_metrics is not None and best_metrics["R2"] >= metrics_new["R2"]:
                templates.pop()  # remove the appended PySR template
                print(f"[Track {track_idx} {coord_name}] PySR R^2={metrics_new['R2']:.3f} "
                      f"but best template R^2={best_metrics['R2']:.3f}; keeping template {best_template_index}.")
                eqn_list.append(best_expr)
                families_list.append(best_template_index)
                return best_expr, best_metrics

            # Otherwise accept PySR
            eqn_list.append(expr_fitted_new)
            families_list.append(new_template_index)
            return expr_fitted_new, metrics_new

        # --- Fit x, y, z this track (use per-hit sigmas) --- #
        event_num = parse_event_number(f_all)           # e.g. 100000001
        sig_x, sig_y, sig_z, n_hits = load_sigmas_for_event(track_folder, event_num)
        
        if sig_x is None:
            print(f"[info] event {event_num}: no sigma columns found → unweighted fits")
        else:
            print(f"[info] event {event_num}: sigma columns found → weighted fits")

        # sanity: lengths must match loaded arrays
        if len(s_all) != n_hits:
            # defensive: if load_many_tracks filtered hits differently, warn and fall back to unweighted
            print(f"[warning] mismatch in hit counts for event {f_all}: s_len={len(s_all)}, csv_hits={n_hits}. Using unweighted fit.")
            expr_x, m_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x")
            expr_y, m_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y")
            expr_z, m_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z")
        else:
            # If sigmas are None (old datasets), this is just unweighted.
            expr_x, m_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x", sigma=sig_x)
            expr_y, m_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y", sigma=sig_y)
            expr_z, m_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z", sigma=sig_z)

        # Store R^2 values for summaries
        r2_x_all.append(m_x["R2"])
        r2_y_all.append(m_y["R2"])
        r2_z_all.append(m_z["R2"])

        # LaTeX strings
        eq_x = r"$x(s) = " + sympy_to_latex_with_s(expr_x) + "$"
        eq_y = r"$y(s) = " + sympy_to_latex_with_s(expr_y) + "$"
        eq_z = r"$z(s) = " + sympy_to_latex_with_s(expr_z) + "$"

        # Plotting using the fitted expressions
        # (build numpy callables from expr_x, expr_y, expr_z)
        # Helper: build a numpy-callable from an expression with exactly one free symbol
        def expr_to_callable(expr):
            free_syms = list(expr.free_symbols)
            if len(free_syms) != 1:
                raise ValueError(f"Expected exactly 1 free symbol in expression, got {free_syms}")
            s_sym = free_syms[0]
            return sp.lambdify(s_sym, expr, "numpy")

        fx = expr_to_callable(expr_x)
        fy = expr_to_callable(expr_y)
        fz = expr_to_callable(expr_z)

        s_plot = np.linspace(s_all.min(), s_all.max(), 500)
        x_pred = fx(s_plot)
        y_pred = fy(s_plot)
        z_pred = fz(s_plot)

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        # x(s)
        label_x = (fr"Fit: $R^2={m_x['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_x['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_x['MAE']:.3e}$, "
           fr"$\chi^2={m_x['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_x['chi2_red']:.3e}$" + "\n" + eq_x)
        axes[0].scatter(s_all, x_all, s=10, alpha=0.6, label="Data")
        axes[0].plot(s_plot, x_pred, linewidth=2, label=label_x)
        axes[0].set_ylabel("$x$")
        axes[0].legend(fontsize=8)

        # y(s)
        label_y = (fr"Fit: $R^2={m_y['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_y['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_y['MAE']:.3e}$, "
           fr"$\chi^2={m_y['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_y['chi2_red']:.3e}$" + "\n" + eq_y)
        axes[1].scatter(s_all, y_all, s=10, alpha=0.6, label="Data")
        axes[1].plot(s_plot, y_pred, linewidth=2, label=label_y)
        axes[1].set_ylabel("$y$")
        axes[1].legend(fontsize=8)

        # z(s)
        label_z = (fr"Fit: $R^2={m_z['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_z['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_z['MAE']:.3e}$, "
           fr"$\chi^2={m_z['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_z['chi2_red']:.3e}$" + "\n" + eq_z)
        axes[2].scatter(s_all, z_all, s=10, alpha=0.6, label="Data")
        axes[2].plot(s_plot, z_pred, linewidth=2, label=label_z)
        axes[2].set_ylabel("$z$")
        axes[2].set_xlabel("$s$")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"../pngs/SR_track_{f_all}.png", dpi=5*96)
        system(f"open ../pngs/SR_track_{f_all}.png") if OPEN_PNGS else None
        
        png_rel = f"../pngs/SR_track_{f_all}.png"  # matches your save path

        track_infos.append({
            "track_idx": track_idx,
            "event_id": str(f_all),
            "png": png_rel,
            "Fx": families_x[-1],
            "Fy": families_y[-1],
            "Fz": families_z[-1],
            "mx": m_x, "my": m_y, "mz": m_z,
            "eqx": eq_x, "eqy": eq_y, "eqz": eq_z,
        })


    # Build explicit LaTeX equations per 3D family
    def latex_equations_for_3d_families(triple_id_map,
                                        families_x, families_y, families_z,
                                        x_eqns, y_eqns, z_eqns):
        # Invert triple_id_map: family_id -> triple
        family_to_triple = {fid: triple for triple, fid in triple_id_map.items()}

        # For each family, pick the first track that belongs to it as representative
        family_to_rep_idx = {}
        for i, triple in enumerate(zip(families_x, families_y, families_z)):
            fid = triple_id_map[triple]
            if fid not in family_to_rep_idx:
                family_to_rep_idx[fid] = i

        # Build LaTeX align* block
        lines = []
        lines.append(r"\begin{align*}")
        for fid in sorted(family_to_rep_idx.keys()):
            i = family_to_rep_idx[fid]

            eqx = sympy_to_latex_with_s(x_eqns[i])
            eqy = sympy_to_latex_with_s(y_eqns[i])
            eqz = sympy_to_latex_with_s(z_eqns[i])

            lines.append(fr"\text{{Family }} {fid}: &\ x(s) = {eqx}, \\")
            lines.append(fr"                 &\ y(s) = {eqy}, \\")
            lines.append(fr"                 &\ z(s) = {eqz} \\[0.5em]")
        lines.append(r"\end{align*}")

        return "\n".join(lines)

    print("\n"*10)

    latex_3d_table, triple_id_map = summarize_3d_families(
        families_x, families_y, families_z,
        r2_x_all, r2_y_all, r2_z_all
    )
    latex_3d_eqns = latex_equations_for_3d_families(
        triple_id_map,
        families_x, families_y, families_z,
        x_eqns, y_eqns, z_eqns,
    )

    print(latex_3d_table)
    print(latex_3d_eqns)
    
    save_obj = {
        "x_templates": x_templates,
        "y_templates": y_templates,
        "z_templates": z_templates,
    }

    with open("track_templates.pkl", "wb") as f:
        pickle.dump(save_obj, f)


    # -----------------------------
    # HTML report (minimal)
    # -----------------------------
    def html_escape(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

    # Group tracks by 3D family id
    fam_to_tracks = defaultdict(list)
    for info in track_infos:
        triple = (info["Fx"], info["Fy"], info["Fz"])
        fid = triple_id_map[triple]
        info["fid"] = fid
        fam_to_tracks[fid].append(info)

    # Sort tracks in each family (optional)
    for fid in fam_to_tracks:
        fam_to_tracks[fid].sort(key=lambda d: d["track_idx"])

    html_parts = []
    html_parts.append(r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Track Fit Report (Scroller)</title>

  <!-- MathJax for LaTeX rendering -->
  <script>
    window.MathJax = {
      tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
      svg: { fontCache: "global" }   // SVG output; no webfont fetches
    };
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-svg.js"></script>

  <!-- TeX-y text fonts -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/latin-modern-roman/index.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/latin-modern-math/index.css">

  <style>
    body {
      font-family: "Latin Modern Roman", "Computer Modern", "CMU Serif", serif;
      margin: 24px;
      background: #fafafa;
    }
    h1 {
      margin: 0 0 8px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .subtle { color: #555; margin-top: 0; }

    .toolbar {
      display: flex; align-items: center; gap: 12px;
      position: sticky; top: 0; background: #fafafa;
      padding: 12px 0; z-index: 2;
      border-bottom: 1px solid #eee;
    }
    button {
      padding: 7px 14px;
      border-radius: 999px;
      border: 1px solid #ddd;
      background: #fff;
      cursor: pointer;
      font-family: inherit;
      font-size: 14px;
    }
    button:hover { box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .counter { color: #444; font-size: 14px; }

    .card {
      border-radius: 16px;
      padding: 16px 20px 20px;
      background: #fff;
      max-width: 980px;
      margin: 0 auto; 
      box-shadow: 0 6px 18px rgba(0,0,0,0.04);
      border: 1px solid #eee;
    }
    
    .eqJump {
      text-decoration: none;
      border-bottom: 1px dotted #888;
      color: inherit;
    }
    .eqJump:hover {
      border-bottom-style: solid;
    }

    
    .meta { font-size: 13px; color: #444; margin: 6px 0 0; }
    .meta b { font-weight: 600; }

    .metrics {
      text-align: center;
      margin: 14px 0 8px;
    }
    .metrics mjx-container {
      font-size: 1.05em;
    }

    .eq {
      font-size: 14px;
      margin-top: 14px;
      text-align: center;
      line-height: 1.6;
    }
    .eq mjx-container {
      font-size: 1.02em;
    }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
    }
    img {
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid #eee;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <h1>Track Fit Report</h1>
  <p class="subtle">Use ← / → or the buttons to scroll through tracks (grouped by family).</p>

  <div class="toolbar">
    <button id="prevBtn">← Prev</button>
    <button id="nextBtn">Next →</button>
    <span class="counter" id="counter"></span>
  </div>

  <div class="card" id="card"></div>

  <script>
    // Data injected from Python below
    const slides = __SLIDES_JSON__;
    const eqSlides = __EQS_JSON__;
    const eqStats  = __EQ_STATS_JSON__;

    function escapeHtml(s) {
      return s.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
    }

    function render(i) {
      const t = slides[i];

      // NOTE: eqx/eqy/eqz are already "$...$" strings.
      const eqBlock = `${t.eqx}<br/>${t.eqy}<br/>${t.eqz}`;

      const html = `
      <div class="meta">
        <b>Family ${t.fid}</b>
        <span class="mono"> (F_x=${t.Fx}, F_y=${t.Fy}, F_z=${t.Fz})</span>
      </div>

      <div id="eqViewer"></div>

      <div class="meta">
        <b>Track ${t.track_idx + 1}</b> (${escapeHtml(t.event_id)})
      </div>

      <div class="meta metrics">${t.metrics_line}</div>

      <a href="${escapeHtml(t.png)}" target="_blank" rel="noopener">
        <img id="trackImg" src="${escapeHtml(t.img_src)}" alt="Track ${t.track_idx} plot"/>
      </a>

      <div class="eq">${eqBlock}</div>
    `;

    
        document.getElementById("card").innerHTML = html;

        document.getElementById("counter").textContent =
        `Item ${i+1} / ${slides.length}`;
        
        renderEqViewer();

      // Re-typeset MathJax after DOM update
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise();
      }
    }

    let idx = 0;

    function goTo(newIndex) {
      // wrap around
      idx = (newIndex + slides.length) % slides.length;
      render(idx);
    }
    
    // -----------------------
    // Equation viewer (global)
    // -----------------------
    let eqIdx = 0;

    function renderEqViewer() {
      const container = document.getElementById("eqViewer");
      if (!container) return;

      const e = eqSlides[eqIdx];

      // build the list of clickable uses
      const usesHtml = e.uses.map(u =>
        `<a href="#" class="eqJump" data-slide="${u.slide}">${escapeHtml(u.label)}</a>`
      ).join(", ");

      container.innerHTML = `
        <div class="meta" style="margin-top:10px;">
          <b>${eqStats.unique}</b> unique equation templates used out of
          <b>${eqStats.total_components}</b> track-components.
        </div>

        <div style="display:flex; align-items:center; gap:10px; margin-top:10px;">
          <button id="eqPrev">←</button>
          <button id="eqNext">→</button>
          <span class="counter">Equation ${eqIdx + 1} / ${eqSlides.length}</span>
        </div>

        <div class="eq" style="margin-top:10px;">
          <div><b>Equation ${eqIdx + 1}:</b></div>
          <div style="margin-top:6px;">${e.eq}</div>
          <div class="meta" style="margin-top:10px;">
            ${usesHtml}
          </div>
        </div>
      `;

      // hook up arrow buttons with wrap
      document.getElementById("eqPrev").onclick = () => {
        eqIdx = (eqIdx - 1 + eqSlides.length) % eqSlides.length;
        renderEqViewer();
      };
      document.getElementById("eqNext").onclick = () => {
        eqIdx = (eqIdx + 1) % eqSlides.length;
        renderEqViewer();
      };

      // hook up each jump link
      for (const a of container.querySelectorAll(".eqJump")) {
        a.addEventListener("click", (ev) => {
          ev.preventDefault();
          const target = parseInt(a.dataset.slide, 10);
          goTo(target);

          // after the card re-renders, scroll the image into view
          setTimeout(() => {
            const img = document.getElementById("trackImg");
            if (img) img.scrollIntoView({ behavior: "smooth", block: "start" });
          }, 0);
        });
      }

      // typeset just-updated math
      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise();
      }
    }


    document.getElementById("prevBtn").addEventListener("click", () => {
      goTo(idx - 1);
    });
    document.getElementById("nextBtn").addEventListener("click", () => {
      goTo(idx + 1);
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft")  { goTo(idx - 1); }
      if (e.key === "ArrowRight") { goTo(idx + 1); }
    });

    render(idx);

  </script>
</body>
</html>
""")

    slides_payload = []
    eq_map = defaultdict(list)  # eq_string -> list of uses
    slide_index = 0
    for fid in sorted(fam_to_tracks.keys()):
        tracks = fam_to_tracks[fid]
        Fx, Fy, Fz = tracks[0]["Fx"], tracks[0]["Fy"], tracks[0]["Fz"]
        
        for t in tracks:
            mx, my, mz = t["mx"], t["my"], t["mz"]
            # Equations are already like: "$x(s)=...$"
            eq_block = "<br/>".join([
                t["eqx"],
                t["eqy"],
                t["eqz"],
            ])
            
            chi_x = sci_to_latex(f"{mx['chi2_red']:.3e}")
            chi_y = sci_to_latex(f"{my['chi2_red']:.3e}")
            chi_z = sci_to_latex(f"{mz['chi2_red']:.3e}")

            metrics_line = (
                r"\begin{align*}"
                rf"&R^2:\, x={mx['R2']:.3f},\;"
                rf"y={my['R2']:.3f},\;"
                rf"z={mz['R2']:.3f}"
                rf"\\"
                rf"&\chi^2_\nu:\;"
                r"\mathrm{x}="f"{chi_x},\;"
                r"\mathrm{y}="f"{chi_y},\;"
                r"\mathrm{z}="f"{chi_z}"
                r"\end{align*}"
            )
            
            # --- record equation usage (track number is 1-based for display) ---
            track_num = t["track_idx"] + 1
            
            # compute signature keys for the templates (store representative family idx)
            sig_x = template_signature(x_templates[t["Fx"]])
            sig_y = template_signature(y_templates[t["Fy"]])
            sig_z = template_signature(z_templates[t["Fz"]])
            # store mapping from (coord, sig) -> list of uses and remember a representative fam idx
            eq_map[("x", sig_x)].append({"slide": slide_index, "label": f"{track_num}-x", "rep_fam": t["Fx"]})
            eq_map[("y", sig_y)].append({"slide": slide_index, "label": f"{track_num}-y", "rep_fam": t["Fy"]})
            eq_map[("z", sig_z)].append({"slide": slide_index, "label": f"{track_num}-z", "rep_fam": t["Fz"]})

            
            slides_payload.append({
                "fid": t["fid"],
                "Fx": t["Fx"], "Fy": t["Fy"], "Fz": t["Fz"],
                "track_idx": t["track_idx"],
                "event_id": t["event_id"],
                "png": t["png"].replace("../",""),                       # link target (as you used before)
                "img_src": t["png"].replace("../",""), # displayed img path (as you used before)
                "metrics_line": metrics_line,
                "eqx": t["eqx"], "eqy": t["eqy"], "eqz": t["eqz"],
            })
            
            slide_index += 1
    
    # Preserve “first equation in template list” ordering:
    # iterate through slides_payload in order and take first time we see an eq
    def template_latex_for(coord, fam_idx):
        # Display the parametric template form (a0, a1, ...) for that family
        if coord == "x":
            return r"$x(s) = " + template_to_latex_with_s(x_templates[fam_idx]) + r"$"
        if coord == "y":
            return r"$y(s) = " + template_to_latex_with_s(y_templates[fam_idx]) + r"$"
        if coord == "z":
            return r"$z(s) = " + template_to_latex_with_s(z_templates[fam_idx]) + r"$"
        raise ValueError(coord)

    seen = set()
    eq_slides = []

    for sp_item in slides_payload:
        keys = [("x", sp_item["Fx"]), ("y", sp_item["Fy"]), ("z", sp_item["Fz"])]
        # For each coordinate, pick the signature for the family index
        for coord, fam_idx in [("x", sp_item["Fx"]), ("y", sp_item["Fy"]), ("z", sp_item["Fz"])]:
            if coord == "x":
                sig = template_signature(x_templates[fam_idx])
                rep_fam = fam_idx
            elif coord == "y":
                sig = template_signature(y_templates[fam_idx])
                rep_fam = fam_idx
            else:
                sig = template_signature(z_templates[fam_idx])
                rep_fam = fam_idx

            key = (coord, sig)
            if key in seen:
                continue
            seen.add(key)

            # Use the representative family index from the first recorded use if available
            uses = eq_map.get(key, [])
            rep_idx = uses[0]["rep_fam"] if uses else rep_fam

            # Render latex for the representative family index
            if coord == "x":
                eq_text = r"$x(s) = " + template_to_latex_with_s(x_templates[rep_idx]) + r"$"
            elif coord == "y":
                eq_text = r"$y(s) = " + template_to_latex_with_s(y_templates[rep_idx]) + r"$"
            else:
                eq_text = r"$z(s) = " + template_to_latex_with_s(z_templates[rep_idx]) + r"$"

            eq_slides.append({
                "key": [coord, sig],
                "eq": eq_text,
                "uses": eq_map.get(key, []),
            })


    eq_stats = {
        "unique": len(eq_slides),
        "total_components": 3 * len(slides_payload),
    }

    
    slides_json = json.dumps(slides_payload)
    eqs_json = json.dumps(eq_slides)
    stats_json = json.dumps(eq_stats)
    html_text = "".join(html_parts)
    html_text = html_text.replace("__SLIDES_JSON__", slides_json)
    html_text = html_text.replace("__EQS_JSON__", eqs_json)
    html_text = html_text.replace("__EQ_STATS_JSON__", stats_json)
    
    out_html = "track_results.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"[HTML] Wrote report to: {out_html}")
    system("mv track_results.html /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/")
    if OPEN_HTML:
        system("open /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/track_results.html")
