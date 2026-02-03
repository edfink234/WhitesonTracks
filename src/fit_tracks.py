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
import math

def _fmt_float_latex(x: float, sig: int = 3, sci_cut: float = 1e-3) -> str:
    """Return latex for a float. Use scientific notation if |x| < sci_cut and x != 0."""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax < sci_cut:
        # scientific notation with sig significant digits
        exp = int(math.floor(math.log10(ax)))
        coeff = x / (10**exp)
        coeff_str = f"{coeff:.{sig-1}f}".rstrip("0").rstrip(".")
        return rf"{coeff_str}\times 10^{{{exp}}}"
    else:
        # fixed with sig decimal places (not sig figs)
        s = f"{x:.{sig}f}".rstrip("0").rstrip(".")
        return s

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
        (common fallback when per-point uncertainties are unknown; \(\sigma ^{2}=\mu \approx \^{y}\))

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

def fit_circle_xy(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    xc0, yc0 = np.mean(x), np.mean(y)
    R0 = np.median(np.sqrt((x-xc0)**2 + (y-yc0)**2))
    p0 = np.array([xc0, yc0, R0], float)

    def resid(p):
        xc, yc, R = p
        return np.sqrt((x-xc)**2 + (y-yc)**2) - R

    res = least_squares(resid, p0, loss="linear")
    return res.x  # xc, yc, R

def fit_z_vs_phase(phi_c, z, *, sig_z=None):
    phi_c = np.asarray(phi_c).ravel()
    z = np.asarray(z).ravel()

    # z ≈ z0 + a*(phi - phi_ref)  <=>  z = b + a*phi  where b = z0 - a*phi_ref
    A = np.vstack([np.ones_like(phi_c), phi_c]).T

    if sig_z is None:
        b, a = np.linalg.lstsq(A, z, rcond=None)[0]
    else:
        w = 1.0 / (np.asarray(sig_z).ravel() + 1e-12)
        Aw = A * w[:, None]
        zw = z * w
        b, a = np.linalg.lstsq(Aw, zw, rcond=None)[0]

    # choose phi_ref = first phi so z0 is "at first hit"
    phi_ref = phi_c[0]
    z0 = b + a*phi_ref
    return z0, a, phi_ref

def helix_xyz_phi(phi, p):
    # p = [xc, yc, R, z0, a, phi_ref]
    xc, yc, R, z0, a, phi_ref = p
    x = xc + R * np.cos(phi)
    y = yc + R * np.sin(phi)
    z = z0 + a * (phi - phi_ref)
    return x, y, z

def fit_standard_model_helix_from_phi(phi, x, y, z, *, sig_x=None, sig_y=None, sig_z=None):
    phi = np.asarray(phi).ravel()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    n = len(phi)

    # initial guesses (robust)
    xc0, yc0 = np.mean(x), np.mean(y)
    R0 = np.median(np.sqrt((x - xc0)**2 + (y - yc0)**2))
    z00 = np.mean(z)
    a0 = (z[-1] - z[0]) / (phi[-1] - phi[0] + 1e-12)
    phi_ref0 = phi[0]
    p0 = np.array([xc0, yc0, R0, z00, a0, phi_ref0], dtype=float)

    BIG = 1e6
    def resid(p):
        xp, yp, zp = helix_xyz_phi(phi, p)
        rx = xp - x
        ry = yp - y
        rz = zp - z
        if sig_x is not None: rx = rx / (np.asarray(sig_x).ravel() + 1e-12)
        if sig_y is not None: ry = ry / (np.asarray(sig_y).ravel() + 1e-12)
        if sig_z is not None: rz = rz / (np.asarray(sig_z).ravel() + 1e-12)
        r = np.concatenate([rx, ry, rz])
        if not np.all(np.isfinite(r)): return np.full(3*n, BIG, dtype=float)
        return r

    # Use linear loss so residuals correspond to chi2
    res = least_squares(resid, p0, loss="linear", xtol=1e-10, ftol=1e-10)
    p_opt = res.x

    r = resid(p_opt)
    chi2 = float(np.sum(r**2))
    ddof = len(p_opt)
    dof = max(len(r) - ddof, 1)
    chi2_red = chi2 / dof

    # simple residual diagnostics
    xp, yp, zp = helix_xyz_phi(phi, p_opt)
    res_xyz = np.concatenate([xp - x, yp - y, zp - z])
    diagnostics = {
        "chi2": chi2, "chi2_red": chi2_red,
        "res_mean": np.mean(res_xyz), "res_std": np.std(res_xyz),
        "p_opt": p_opt, "success": res.success, "message": res.message
    }
    return p_opt, diagnostics
# -------------------------------------------------------------------


def helix_xyz(s, p):
    """
    Simple 3D helix parametrized by s:
      x = xc + R cos(omega s + phi0)
      y = yc + R sin(omega s + phi0)
      z = z0 + k s
    p = [xc, yc, R, omega, phi0, z0, k]
    """
    xc, yc, R, omega, phi0, z0, k = p
    th = omega * s + phi0
    x = xc + R * np.cos(th)
    y = yc + R * np.sin(th)
    z = z0 + k * s
    return x, y, z

def fit_standard_model_helix(s, x, y, z, *, sig_x=None, sig_y=None, sig_z=None):
    """
    Fit helix to all coords at once using least_squares.
    Returns: (p_opt, metrics_xyz)
      metrics_xyz: dict with chi2, chi2_red, plus per-dim R2/MSE/MAE if you want.
    """

    s = np.asarray(s).ravel()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    n = len(s)

    # --- initial guess (cheap + robust) ---
    xc0, yc0 = np.mean(x), np.mean(y)
    r0 = np.sqrt((x - xc0)**2 + (y - yc0)**2)
    R0 = np.median(r0) if np.isfinite(np.median(r0)) else 1.0
    omega0 = 2*np.pi / (s.max() - s.min() + 1e-9)   # ~one turn across the s-range
    phi0 = np.arctan2(np.mean(y - yc0), np.mean(x - xc0))
    z00 = np.mean(z)
    k0 = (z[-1] - z[0]) / (s[-1] - s[0] + 1e-9)

    p0 = np.array([xc0, yc0, R0, omega0, phi0, z00, k0], dtype=float)

    BIG = 1e6

    def resid(p):
        with np.errstate(all="ignore"):
            xp, yp, zp = helix_xyz(s, p)

        if (not np.all(np.isfinite(xp))) or (not np.all(np.isfinite(yp))) or (not np.all(np.isfinite(zp))):
            return np.full(3*n, BIG, dtype=float)

        rx = xp - x
        ry = yp - y
        rz = zp - z

        if sig_x is not None:
            rx = rx / np.asarray(sig_x).ravel()
        if sig_y is not None:
            ry = ry / np.asarray(sig_y).ravel()
        if sig_z is not None:
            rz = rz / np.asarray(sig_z).ravel()

        r = np.concatenate([rx, ry, rz])
        if not np.all(np.isfinite(r)):
            return np.full(3*n, BIG, dtype=float)
        return r

    res = least_squares(resid, p0, loss="soft_l1")
    p_opt = res.x

    # compute chi2 from weighted residuals (or unweighted if no sigmas)
    r = resid(p_opt)
    chi2 = np.sum(r**2)
    ddof = len(p_opt)
    dof = max(len(r) - ddof, 1)
    chi2_red = chi2 / dof

    # optional: per-dim R2 for sanity / label tie-breaks
    xp, yp, zp = helix_xyz(s, p_opt)
    mx = regression_metrics(x, xp, sigma=sig_x, ddof=ddof if sig_x is not None else 0)
    my = regression_metrics(y, yp, sigma=sig_y, ddof=ddof if sig_y is not None else 0)
    mz = regression_metrics(z, zp, sigma=sig_z, ddof=ddof if sig_z is not None else 0)

    metrics_xyz = {
        "chi2": chi2,
        "chi2_red": chi2_red,
        "mx": mx, "my": my, "mz": mz,
    }
    return p_opt, metrics_xyz

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

def sympy_expr_to_pysr_guess(expr, *, pysr_var="x0"):
    """
    Convert a SymPy expression (with numeric constants) into a PySR guess string.
    Assumes your independent variable is the only free symbol in expr.
    """
    expr = sp.sympify(expr)

    free = list(expr.free_symbols)
    if len(free) != 1:
        raise ValueError(f"Expected 1 free symbol in expr for PySR guess; got {free}")

    # rename that symbol to x0 so PySR understands it
    expr = expr.xreplace({free[0]: sp.Symbol(pysr_var)})

    s = sp.sstr(expr).replace("**", "^")
    # --- avoid unary minus at the start (PySR parser can choke if '-' not in unary ops) ---
    s_strip = s.lstrip()
    if s_strip.startswith("-"):
        # turn "-(...)" or "-x0*..." into "0-(...)" which uses binary '-'
        s = "0" + s

    # also normalize "+ -something" into "- something" (binary minus)
    s = s.replace("+ -", "- ")
    return s

def _chi2_red(y, yhat, sigma, k_params=0):
    r = (y - yhat) / sigma
    chi2 = float(np.sum(r*r))
    dof = max(1, len(y) - k_params)
    return chi2 / dof

def _estimate_num_params(model):
    """
    Best-effort estimate of number of fitted constants.
    For PySR, the hall-of-fame includes 'n_params' sometimes, but not always.
    We fall back to counting 'C' constants in sympy or just 0.
    """
    try:
        # PySR typically provides get_best() / sympy() depending on version
        # safest: model.get_best() returns a dict with 'sympy_format'
        best = model.get_best()
        if isinstance(best, dict):
            # recent PySR often includes 'num_params' or similar
            for key in ("num_params", "n_params", "nconstants", "complexity"):
                if key in best and isinstance(best[key], (int, np.integer)):
                    # NOTE: 'complexity' is NOT params, so only use if you know your schema
                    pass
            if "n_params" in best:
                return int(best["n_params"])
        # fallback: 0
    except Exception:
        pass
    return 0

def fit_until_both_conditions(
    s, y, model_kwargs,
    R2_THRESHOLD,
    min_iters=100, step=50, max_iters=np.inf,
    stop_flag_path="STOP_PYSR",
    sigma=None,
    CHI2_TOL=0.2,   # e.g. stop if |chi2_red-1| <= 0.2
    require_both=True  # if True: weighted -> require chi2 AND R2; else just chi2
):
    model = PySRRegressor(**model_kwargs)
    total_iters = 0
    best_R2 = -np.inf
    best_chi2 = np.inf

    y = np.asarray(y).ravel()
    s = np.asarray(s)

    weighted = sigma is not None
    if weighted:
        sigma = np.asarray(sigma).ravel()
        assert sigma.shape == y.shape

    while True:
        model.niterations = total_iters + step
        # NOTE: PySR supports sample weights via `weights=` in many versions.
        # If your version supports it, use: model.fit(s, y, weights=1/sigma**2)
        model.fit(s, y)

        total_iters += step

        # Predictions
        yhat = model.predict(s)

        # R² (unweighted unless you implement weighted R²)
        R2 = model.score(s, y)
        if R2 > best_R2:
            best_R2 = R2
            print(f"New best R2 = {best_R2:.3f}")

        # chi2_red (only if weighted)
        if weighted:
            k = _estimate_num_params(model)  # best-effort
            chi2_red = _chi2_red(y, yhat, sigma, k_params=k)
            if abs(chi2_red - 1.0) < abs(best_chi2 - 1.0):
                best_chi2 = chi2_red
                print(f"New best chi2_red = {best_chi2:.3f}")

        # stopping
        if total_iters >= min_iters:
            if weighted:
                chi2_ok = abs(best_chi2 - 1.0) <= CHI2_TOL
                r2_ok = best_R2 >= R2_THRESHOLD
                if (chi2_ok and r2_ok) if require_both else chi2_ok:
                    print(f"[fit_until_both_conditions] Stop: iters={total_iters}, best_R2={best_R2:.3f}, best_chi2_red={best_chi2:.3f}")
                    break
            else:
                if best_R2 >= R2_THRESHOLD:
                    print(f"[fit_until_both_conditions] Reached R^2 >= {R2_THRESHOLD} with {total_iters} iterations.")
                    break

        if total_iters >= max_iters:
            if weighted:
                print(f"[fit_until_both_conditions] Reached max_iters={max_iters} with best_R2={best_R2:.3f}, best_chi2_red={best_chi2:.3f}.")
            else:
                print(f"[fit_until_both_conditions] Reached max_iters={max_iters} with best_R2={best_R2:.3f}.")
            break

        if os.path.exists(stop_flag_path):
            print(f"[fit_until_both_conditions] Stop flag '{stop_flag_path}' detected; aborting this fit.")
            os.remove(stop_flag_path)
            break

    system("rm -f *hall_of_fame*")
    return model, best_R2, total_iters


def sympy_to_latex_with_s(expr, *, ndigits=3, sci_cut=1e-3):
    """LaTeX with x0→s replacement and scientific notation for tiny floats."""
    # Convert to latex first
    expr_latex = sp.latex(expr)

    # Replace variable name
    expr_latex = expr_latex.replace("x_{0}", "s").replace("x0", "s")

    # Replace numeric literals in latex string.
    # This regex targets numbers like -1.79122404979219e-5 or 0.2822120814
    import re
    number_re = re.compile(r'(?<![A-Za-z])([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)(?![A-Za-z])')

    def repl(m):
        token = m.group(1)
        try:
            x = float(token)
        except Exception:
            return token
        return _fmt_float_latex(x, sig=ndigits, sci_cut=sci_cut)

    expr_latex = number_re.sub(repl, expr_latex)
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

def make_parametric_template(expr, s_name="s"):
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

    # If there are no parameters, don't call least_squares
    if len(param_syms) == 0:
        with np.errstate(all="ignore"):
            y_pred = f(s_data)   # <-- call f with ONLY s_data

        # guard like you do elsewhere
        if (not np.all(np.isfinite(y_pred))) or np.any(np.iscomplex(y_pred)):
            y_pred = np.full_like(y_data, np.nan, dtype=float)

        metrics = regression_metrics(y_data, y_pred, sigma=sigma, ddof=0)
        return metrics, np.array([]), expr_param, y_pred

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
    np.random.seed(0)  # Fix random seed
    np.sech = lambda x: 1/np.cosh(x)
    OPEN_PNGS = False
    OPEN_HTML = True
    create_dataset = True
    TEMPLATE_PATH = "track_templates.pkl"
    ADD_FUNC_TO_TEMPLATES = True
    R2_THRESHOLD = 0.997
    RunPySR = True   # Whether to enable (True) or disable (False) PySR discovery
    MaxPySRIters = 100
    num_tracks = 5
    loaded = {}
    x_templates, y_templates, z_templates = [], [], []
    
    track_dataset_idx = 0
    out_html = [
        "v20260122_163839__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.html",
        "v1_noiseless_69000.html",
        "v20260202_142140__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel.html"
    ]
    track_folder = [f"../tracks_for_ed/{i[:-5]}" for i in out_html]
    out_html = out_html[track_dataset_idx]
    track_folder = track_folder[track_dataset_idx]
    dataset_labels = [
        "v20260122_163839 train/test (noise XY=0.01, Z=0.01)",
        "v1_noiseless_69000 (no sigma columns)",
        "v20260202_142140 train/test (noise XY=0.01, Z=0.01) Standard Model"
    ]
    dataset_label = dataset_labels[track_dataset_idx]
    is_standard_model_dataset = ("standardModel" in out_html)
    
    outfile_str = out_html[:-5]
    
    if os.path.exists(TEMPLATE_PATH):
        print("Loading existing template library...")
        with open(TEMPLATE_PATH, "rb") as f:
            loaded = pickle.load(f)

        x_templates = loaded["x_templates"]
        y_templates = loaded["y_templates"]
        z_templates = loaded["z_templates"]

    else:
        print("No template file found — starting with empty template lists.")
    
#        z_templates.append(make_parametric_template(-34.520199*sp.sech(s**(-1.45342) - 3.00504026595319), s_name="s"))
    track_infos = []  # store per-track info for HTML
    x_eqns, y_eqns, z_eqns = [], [], []   # store final numeric expressions used
    families_x, families_y, families_z = [], [], []  # indices of template used
    r2_x_all, r2_y_all, r2_z_all = [], [], [] # R^2 values for each track and coordinate

    # now load sigma arrays too
    S, X, Y, Z, SIG_X_LIST, SIG_Y_LIST, SIG_Z_LIST, F = load_many_tracks(track_folder, max_tracks=num_tracks)
    num_tracks = len(S)
    assert((len(S), len(X), len(Y), len(Z), len(F)) == (num_tracks, num_tracks, num_tracks, num_tracks, num_tracks))
#    print(f"len(S), len(X), len(Y), len(Z), len(F) = {(len(S), len(X), len(Y), len(Z), len(F))}")
#    print(SIG_X_LIST[0], SIG_Y_LIST[0], SIG_Z_LIST[0]); exit()
    
    def create_stubborn_track_dataset(
        X, Y,
        file_path="",
        latex_eqn="",
        show=False,
        plot_func=None,          # callable for plotting-only mode
        fitPlotFunc=False,
        fit_expr=None,           # SymPy expr (with float seeds) for fitting mode
        s_name="s",
        sigma=None,
        coord="x",
        ylim = None
    ):
        stubborn_track_dataset = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
        assert stubborn_track_dataset.shape[0] == X.shape[0]
        assert stubborn_track_dataset.shape[0] == Y.shape[0]
        assert stubborn_track_dataset.shape[1] == 2

        if file_path:
            np.savetxt(file_path, stubborn_track_dataset, delimiter=',')
            print(f"Data saved to {file_path}")
        else:
            print(stubborn_track_dataset)

        if show:
            s_data = stubborn_track_dataset[:, 0].astype(float)
            y_true = stubborn_track_dataset[:, 1].astype(float)

            plt.scatter(s_data, y_true, label="Data")

            s_vals = np.linspace(s_data.min(), s_data.max(), 1000)

            if fitPlotFunc:
                if plot_func is None:
                    raise ValueError("fitPlotFunc=True requires plot_func to be a SymPy expression with float seeds.")

                template = make_parametric_template(plot_func, s_name=s_name)
                metrics, p_opt, expr_fitted, y_pred = fit_template_to_data(
                    template, s_data, y_true, sigma=sigma
                )
                s_latex = sp.Symbol("s")  # for printing
                expr_for_latex = round_floats(expr_fitted.xreplace({template["s_sym"]: s_latex}), 3)
                
                latex_eqn = f"${sp.latex(expr_for_latex)}$"

                # plot fitted curve
                f_plot = sp.lambdify((template["s_sym"],), expr_fitted, "numpy")

                if ADD_FUNC_TO_TEMPLATES:
                    def _template_signature(t):
                        # t is a template dict with key "expr"
                        return sp.srepr(t["expr"])

                    def _load_template_db(path):
                        if not os.path.exists(path):
                            return {"x_templates": [], "y_templates": [], "z_templates": []}
                        with open(path, "rb") as f:
                            db = pickle.load(f)

                        # Backwards/defensive normalization
                        if isinstance(db, list):
                            # old format: treat as x_templates by default
                            db = {"x_templates": db, "y_templates": [], "z_templates": []}
                        elif isinstance(db, dict):
                            db.setdefault("x_templates", [])
                            db.setdefault("y_templates", [])
                            db.setdefault("z_templates", [])
                        else:
                            raise TypeError(f"Unexpected templates db type: {type(db)}")
                        return db

                    def _save_template_db(path, db):
                        with open(path, "wb") as f:
                            pickle.dump(db, f)

                    coord_key = f"{coord.lower()}_templates"   # requires coord in {"x","y","z"}
                    db = _load_template_db(TEMPLATE_PATH)

                    bucket = db[coord_key]

                    new_sig = _template_signature(template)
                    existing_sigs = {_template_signature(t) for t in bucket}

                    if new_sig not in existing_sigs:
                        bucket.append(template)
                        _save_template_db(TEMPLATE_PATH, db)
                        print(f"[template] Added new template to {coord_key} → {TEMPLATE_PATH}")
                    else:
                        print(f"[template] Template already exists in {coord_key}, not adding.")
                plt.plot(
                    s_vals, f_plot(s_vals),
                    label=fr"Fit (template): $R^2={metrics['R2']:.3f}$" + "\n" + latex_eqn,
                )
                if ylim:
                    plt.ylim(*ylim)

                print("Fitted params:", p_opt)
                print("Fitted expr:", expr_fitted)
                print("Metrics:", metrics)

            else:
                if plot_func is not None:
                    y_pred = plot_func(s_data)
                    R2 = r2_score_1d(y_true, y_pred)
                    plt.plot(
                        s_vals, plot_func(s_vals),
                        label=fr"Fit: $R^2={R2:.3f}$" + ("\n" + latex_eqn if latex_eqn else ""),
                    )
                    if ylim:
                        plt.ylim(*ylim)

            plt.xlabel("s")
            plt.ylabel("z")
            plt.legend()
            plt.show()
        exit()
    
    if create_dataset:
        base_path = "../stubborn_track_csvs"
        track_number = 3
        assert(track_number <= num_tracks)
        file_path = f"{base_path}/{out_html[:-5]}event10000000{track_number}-hits_Z.csv"
        coord = file_path[-5].lower()
        s = sp.Symbol("s")
        print(len(S))
        fitPlotFunc = False
        np.asin = np.arcsin; np.acos = np.arccos;
        plot_func = s/(sp.cos(sp.sqrt(s)) - 2.2661800709136) + 2.345027 if fitPlotFunc else lambda s: s/(np.cos(np.sqrt(s)) - 2.2661800709136) + 2.345027
        plot_func_eqn = r"$z(s) = \frac{s}{\cos{\left(\sqrt{s} \right)} - 2.27} + 2.35$".replace("x_{0}","s")
#        plot_func = None
#        plot_func_eqn = None
        print(*zip(S[track_number-1][:], Z[track_number-1][:]), sep = '\n')
        create_stubborn_track_dataset(S[track_number-1][:], Z[track_number-1][:], file_path = file_path, show  = True, plot_func = plot_func, latex_eqn = plot_func_eqn, fitPlotFunc = fitPlotFunc, coord = coord, ylim = (-200, 75))

    model_kwargs = dict(
        niterations=100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["neg", "sin", "cos", "exp", "tanh", "sqrt", "log", "sech(x) = 1/cosh(x)"],
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
    def plot_points(ax, s, y, sigma, label="Data"):
        if sigma is None:
            ax.scatter(s, y, s=10, alpha=0.6, label=label)
        else:
            ax.errorbar(
                s, y,
                yerr=sigma,
                fmt="o",
                markersize=3,
                elinewidth=1,
                capsize=2,
                alpha=0.6,
                label=label
            )

    for track_idx, (s_all, x_all, y_all, z_all, f_all,
                sigx_from_loader, sigy_from_loader, sigz_from_loader) in enumerate(
                    zip(S, X, Y, Z, F, SIG_X_LIST, SIG_Y_LIST, SIG_Z_LIST)):
        s_all_reshaped = s_all.reshape(-1, 1)

        # --- Helper inner function to handle one coordinate --- #
        def fit_dimension(s_data, y_data, templates, eqn_list, families_list, coord_name, *, sigma=None):
            print(f"\n--- Fitting Track {track_idx}, coordinate {coord_name} ---")

            # Selection logic:
            #  - If sigma is provided: prefer reduced chi^2 (chi2_red) closest to 1
            #  - Else: prefer highest R^2 and require R2_THRESHOLD for acceptance
            CHI2_RED_TARGET = 1.0
            CHI2_RED_TOL = 0.2   # "good enough" band; tweak if you want stricter/looser

            def is_weighted():
                return sigma is not None

            def better(a, b):
                """Return True if metrics dict a is better than b under the active criterion."""
                if b is None:
                    return True
                if is_weighted():
                    da = abs(a["chi2_red"] - CHI2_RED_TARGET)
                    db = abs(b["chi2_red"] - CHI2_RED_TARGET)
                    if da != db:
                        return da < db
                    # tie-break: higher R2
                    return a["R2"] > b["R2"]
                else:
                    return a["R2"] > b["R2"]

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

                    if better(metrics, best_metrics):
                        best_metrics = metrics
                        best_expr = expr_fitted
                        best_template_index = idx

            print(f"[Track {track_idx} {coord_name}] best template = {best_expr}")
            # 2) If good enough, use best template
            if templates and best_metrics is not None:
                if is_weighted():
                    if abs(best_metrics["chi2_red"] - CHI2_RED_TARGET) <= CHI2_RED_TOL:
                        print(
                            f"[Track {track_idx} {coord_name}] Used existing template {best_template_index} "
                            f"with chi2_red={best_metrics['chi2_red']:.3f} (target 1)"
                        )
                        eqn_list.append(best_expr)
                        families_list.append(best_template_index)
                        return best_expr, best_metrics
                else:
                    if best_metrics["R2"] >= R2_THRESHOLD:
                        print(
                            f"[Track {track_idx} {coord_name}] Used existing template {best_template_index} "
                            f"with R^2={best_metrics['R2']:.3f}"
                        )
                        eqn_list.append(best_expr)
                        families_list.append(best_template_index)
                        return best_expr, best_metrics

            # 3) Otherwise, run PySR to discover new form
            if not RunPySR:
                if best_expr is None:
                    raise RuntimeError(
                        f"No templates available (or all failed) for Track {track_idx} {coord_name}, "
                        f"and RunPySR=False. Can't produce a fit."
                    )
                if is_weighted():
                    print(
                        f"[Track {track_idx} {coord_name}] RunPySR=False; using best available template "
                        f"(chi2_red={best_metrics['chi2_red']:.3f}, target 1)."
                    )
                else:
                    print(
                        f"[Track {track_idx} {coord_name}] best_R2={best_metrics['R2']:.3f} < {R2_THRESHOLD} "
                        f"but RunPySR=False; using best available template."
                    )
                eqn_list.append(best_expr)
                families_list.append(best_template_index)
                return best_expr, best_metrics

            if is_weighted():
                best_delta = float("inf") if best_metrics is None else abs(best_metrics["chi2_red"] - CHI2_RED_TARGET)
                print(
                    f"[Track {track_idx} {coord_name}] best |chi2_red-1|={best_delta:.3f} > {CHI2_RED_TOL} "
                    f"→ running PySR now."
                )
            else:
                best_R2 = -np.inf if best_metrics is None else best_metrics["R2"]
                print(f"best_R2 of {best_R2} is less than {R2_THRESHOLD}, running PYSR now.")

            # --- Seed PySR with fitted template(s) as guesses ---
            guesses = []

            if best_expr is not None:
                try:
                    guesses.append(sympy_expr_to_pysr_guess(best_expr, pysr_var="x0"))
                except Exception as e:
                    print(f"[Track {track_idx} {coord_name}] Could not convert best_expr to PySR guess: {e}")

            # Make a per-call copy of model_kwargs without rebinding the outer name
            mk = dict(model_kwargs)   # <-- key change: NEW name
            if guesses:
                mk["guesses"] = guesses
                print(f"[Track {track_idx} {coord_name}] PySR guesses = {guesses}")

            # For weighted fits, stop based on chi2_red being near 1 (plus min_iters),
            # and optionally ALSO require R2 >= threshold (I recommend not requiring both initially).
            if is_weighted():
                model, best_R2, iters = fit_until_both_conditions(
                    s_data.reshape(-1, 1),
                    y_data,
                    mk,
                    R2_THRESHOLD,              # keep it as a tie-break / info
                    min_iters=100,
                    step=50,
                    max_iters=MaxPySRIters,
                    sigma=sigma,               # <-- NEW
                    CHI2_TOL=CHI2_RED_TOL,      # <-- NEW (same tolerance you use above)
                    require_both=False,         # <-- NEW (stop on chi2_red band; R2 is just logged)
                )
            else:
                model, best_R2, iters = fit_until_both_conditions(
                    s_data.reshape(-1, 1),
                    y_data,
                    mk,
                    R2_THRESHOLD,
                    min_iters=100,
                    step=50,
                    max_iters=MaxPySRIters,
                    sigma=None,                # explicit
                )

            expr = model.sympy()

            # --- normalize PySR's input symbol name to "s" ---
            free_syms = list(expr.free_symbols)
            if len(free_syms) != 1:
                raise ValueError(f"PySR returned expr with free_symbols={free_syms}; expected exactly 1 input variable.")
            expr = expr.subs({free_syms[0]: sp.Symbol("s")})

            template = make_parametric_template(expr, s_name="s")

            templates.append(template)
            new_template_index = len(templates) - 1

            metrics_new, _, expr_fitted_new, _ = fit_template_to_data(template, s_data, y_data, sigma=sigma)

            # If templates existed and one of them was better, keep it instead of PySR
            if best_metrics is not None:
                if is_weighted():
                    if abs(best_metrics["chi2_red"] - CHI2_RED_TARGET) <= abs(metrics_new["chi2_red"] - CHI2_RED_TARGET):
                        templates.pop()  # remove appended PySR template
                        print(
                            f"[Track {track_idx} {coord_name}] PySR chi2_red={metrics_new['chi2_red']:.3f} "
                            f"but best template chi2_red={best_metrics['chi2_red']:.3f}; "
                            f"keeping template {best_template_index}."
                        )
                        eqn_list.append(best_expr)
                        families_list.append(best_template_index)
                        return best_expr, best_metrics
                else:
                    if best_metrics["R2"] >= metrics_new["R2"]:
                        templates.pop()  # remove appended PySR template
                        print(
                            f"[Track {track_idx} {coord_name}] PySR R^2={metrics_new['R2']:.3f} "
                            f"but best template R^2={best_metrics['R2']:.3f}; keeping template {best_template_index}."
                        )
                        eqn_list.append(best_expr)
                        families_list.append(best_template_index)
                        return best_expr, best_metrics

            # Otherwise accept PySR
            eqn_list.append(expr_fitted_new)
            families_list.append(new_template_index)
            return expr_fitted_new, metrics_new

        # prefer sigmas coming from loader (these match the CSV file)
        sig_x = sigx_from_loader
        sig_y = sigy_from_loader
        sig_z = sigz_from_loader

        # if loader returned None (old files without sigma cols), try the per-event loader as before
        if sig_x is None:
            event_num = parse_event_number(f_all)           # e.g. 100000001
            sig_x, sig_y, sig_z, n_hits = load_sigmas_for_event(track_folder, event_num)
        else:
            n_hits = len(s_all)

        if sig_x is None:
            print(f"[info] event {f_all}: no sigma columns found → unweighted fits")
        else:
            print(f"[info] event {f_all}: sigma columns found → weighted fits")

        # sanity: lengths must match loaded arrays (same check you had before)
        if len(s_all) != n_hits:
            print(f"[warning] mismatch in hit counts for event {f_all}: s_len={len(s_all)}, csv_hits={n_hits}. Using unweighted fit.")
            expr_x, m_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x")
            expr_y, m_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y")
            expr_z, m_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z")
        else:
            expr_x, m_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x", sigma=sig_x)
            expr_y, m_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y", sigma=sig_y)
            expr_z, m_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z", sigma=sig_z)

        sm_popt = None
        sm_metrics = None

        sm_x = sm_y = sm_z = None
        chi2nu_sm = None

        if is_standard_model_dataset:
            # --- 1) fit circle in x–y to get helix center ---
            xc0, yc0 = np.mean(x_all), np.mean(y_all)
            R0 = np.median(np.sqrt((x_all - xc0)**2 + (y_all - yc0)**2))

            def circle_resid(p):
                xc, yc, R = p
                return np.sqrt((x_all - xc)**2 + (y_all - yc)**2) - R

            res_xy = least_squares(circle_resid, [xc0, yc0, R0], loss="soft_l1", f_scale=1.0)
            xc_fit, yc_fit, R_fit = res_xy.x
            print("circle RMS:", np.std(circle_resid([xc_fit, yc_fit, R_fit])))

            # --- 2) compute phase about the fitted center (THIS is the generator phi) ---
            phi_unwrap = np.unwrap(np.arctan2(y_all - yc_fit, x_all - xc_fit))

            # --- 3) fit z vs phase: z = z0 + a*(phi - phi0) ---
            A = np.vstack([np.ones_like(phi_unwrap), phi_unwrap]).T
            if sig_z is None:
                b, a = np.linalg.lstsq(A, z_all, rcond=None)[0]
            else:
                w = 1.0 / (sig_z + 1e-12)
                Aw = A * w[:, None]
                zw = z_all * w
                b, a = np.linalg.lstsq(Aw, zw, rcond=None)[0]

            phi0 = phi_unwrap[0]
            z0 = b + a * phi0

            # --- 4) build SM predictions at the HIT POINTS (no interpolation) ---
            x_sm = xc_fit + R_fit * np.cos(phi_unwrap)
            y_sm = yc_fit + R_fit * np.sin(phi_unwrap)
            z_sm = z0 + a * (phi_unwrap - phi0)
            sm_x = x_sm
            sm_y = y_sm
            sm_z = z_sm

            # --- 5) diagnostics ---
            res_xyz = np.concatenate([x_sm - x_all, y_sm - y_all, z_sm - z_all])
            chi2 = np.sum((res_xyz / np.concatenate([sig_x, sig_y, sig_z]))**2) if sig_x is not None else np.sum(res_xyz**2)
            dof = max(len(res_xyz) - 5, 1)  # 5 params: xc,yc,R,z0,a
            chi2_red = chi2 / dof

            sm_metrics = {
                "chi2": chi2,
                "chi2_red": chi2_red,
                "xc": xc_fit, "yc": yc_fit, "R": R_fit,
                "z0": z0, "a": a
            }
            
            chi2nu_sm = sm_metrics["chi2_red"]

            print(f"[SM Fit] xc={xc_fit:.2f}, yc={yc_fit:.2f}, R={R_fit:.2f}")
            print(f"[SM Fit] chi2_red={chi2_red:.3e}, res_std={np.std(res_xyz):.3e}")

        # Store R^2 values for summaries
        r2_x_all.append(m_x["R2"])
        r2_y_all.append(m_y["R2"])
        r2_z_all.append(m_z["R2"])

        # LaTeX strings
        eq_x = r"$x(s) = " + sympy_to_latex_with_s(expr_x) + "$"
        eq_y = r"$y(s) = " + sympy_to_latex_with_s(expr_y) + "$"
        eq_z = r"$z(s) = " + sympy_to_latex_with_s(expr_z) + "$"

        S_SYM = sp.Symbol("s")

        def expr_to_callable(expr, s_sym=S_SYM):
            # constant expression: make it vectorized
            if len(expr.free_symbols) == 0:
                c = float(expr)
                return lambda x: c + 0*x

            # if it contains the expected independent symbol, use it
            if s_sym in expr.free_symbols:
                return sp.lambdify(s_sym, expr, "numpy")

            # otherwise: fall back to the only symbol if there is exactly one
            if len(expr.free_symbols) == 1:
                only = next(iter(expr.free_symbols))
                return sp.lambdify(only, expr, "numpy")

            raise ValueError(f"Expression has ambiguous symbols {expr.free_symbols}; expected {s_sym}.")

        fx = expr_to_callable(expr_x)
        fy = expr_to_callable(expr_y)
        fz = expr_to_callable(expr_z)

        # s_all is integer indices 0..(n-1). For plotting, use a dense float grid over the same range.
        if len(s_all) > 0:
            s_plot = np.linspace(s_all[0], s_all[-1], 500)
        else:
            s_plot = np.array([])
        x_pred = fx(s_plot)
        y_pred = fy(s_plot)
        z_pred = fz(s_plot)
        
        if is_standard_model_dataset and sm_popt is not None:
            # phi-fit => evaluate with helix_xyz_phi at the hit phis (NO s_plot)
            phi_unwrap = np.unwrap(np.arctan2(y_all, x_all))
            x_sm, y_sm, z_sm = helix_xyz_phi(phi_unwrap, sm_popt)
            chi2nu_sm = sm_metrics["chi2_red"]

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

        # x(s)
        label_x = (fr"Fit: $R^2={m_x['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_x['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_x['MAE']:.3e}$, "
           fr"$\chi^2={m_x['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_x['chi2_red']:.3e}$" + "\n" + eq_x)
        plot_points(axes[0], s_all, x_all, sig_x, label="Data")
        axes[0].plot(s_plot, x_pred, linewidth=2, label=label_x)
        if is_standard_model_dataset and (sm_x is not None):
            axes[0].plot(
                s_all, x_sm,
                linewidth=2,
                linestyle="--",
                label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$"
            )
        axes[0].set_ylabel("$x$")
        axes[0].legend(fontsize=8)

        # y(s)
        label_y = (fr"Fit: $R^2={m_y['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_y['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_y['MAE']:.3e}$, "
           fr"$\chi^2={m_y['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_y['chi2_red']:.3e}$" + "\n" + eq_y)
        plot_points(axes[1], s_all, y_all, sig_y, label="Data")
        axes[1].plot(s_plot, y_pred, linewidth=2, label=label_y)
        if is_standard_model_dataset and (sm_y is not None):
            axes[1].plot(s_all, y_sm, linewidth=2, linestyle="--", label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$")
        axes[1].set_ylabel("$y$")
        axes[1].legend(fontsize=8)

        # z(s)
        label_z = (fr"Fit: $R^2={m_z['R2']:.3f}$, "
           fr"$\mathrm{{MSE}}={m_z['MSE']:.3e}$, "
           fr"$\mathrm{{MAE}}={m_z['MAE']:.3e}$, "
           fr"$\chi^2={m_z['chi2']:.3e}$, "
           fr"$\chi^2_\nu={m_z['chi2_red']:.3e}$" + "\n" + eq_z)
        plot_points(axes[2], s_all, z_all, sig_z, label="Data")
        axes[2].plot(s_plot, z_pred, linewidth=2, label=label_z)
        if is_standard_model_dataset and (sm_z is not None):
            axes[2].plot(s_all, z_sm, linewidth=2, linestyle="--", label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$")
        axes[2].set_ylabel("$z$")
        axes[2].set_xlabel("hit index")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        img_str = f"../pngs/SR_track_{outfile_str}{f_all}.png"
        plt.savefig(img_str, dpi=5*96)
        system(f"open {img_str}") if OPEN_PNGS else None
        system(f"cp {img_str} /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/pngs/")
        png_rel = f"{img_str}"  # matches your save path

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
  <title>Track Fit Report - __DATASET_LABEL__</title>

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
  <h1>Track Fit Report - __DATASET_LABEL__</h1>
  <p class="subtle">Dataset: <span class="mono">__DATASET_FOLDER__</span></p>
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
    html_text = html_text.replace("__DATASET_LABEL__", html_escape(dataset_label))
    html_text = html_text.replace("__DATASET_FOLDER__", html_escape(track_folder))

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"[HTML] Wrote report to: {out_html}")
    system(f"mv {out_html} /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/")
    if OPEN_HTML:
        system(f"open /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/{out_html}")
