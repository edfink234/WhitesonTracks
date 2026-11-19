from load_tracks import load_many_tracks
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import least_squares
from collections import defaultdict
import pickle
import os
from os import system
import json
import re
import pandas as pd
import math
import glob
import time
USE_K = True
USE_ANALYTIC_JAC = False

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

def template_signature(template_or_expr, round_digits=8):
    """
    Canonical structural signature for a template, ignoring parameter names.

    This makes these equivalent:
      a0 + a2*sin(a1 + s)
      a1 + a2*sin(a0 + s)
      a0*sin(a2 + s) + a1
    """
    if isinstance(template_or_expr, dict):
        expr = template_or_expr["expr"]
        s_sym = template_or_expr.get("s_sym", sp.Symbol("s"))
        param_syms = list(template_or_expr.get("param_syms", []))
    else:
        expr = template_or_expr
        s_sym = sp.Symbol("s")
        param_syms = sorted(
            [x for x in expr.free_symbols if x.name.startswith("a")],
            key=lambda x: x.name,
        )

    expr = round_floats(expr, round_digits)

    # Normalize independent variable name.
    expr = expr.xreplace({s_sym: sp.Symbol("s")})

    # Normalize parameter names by first occurrence in the expression tree.
    canonical_params = {}
    counter = 0

    def visit(e):
        nonlocal counter

        if e == sp.Symbol("s"):
            return e

        if e in param_syms:
            if e not in canonical_params:
                canonical_params[e] = sp.Symbol(f"p{counter}")
                counter += 1
            return canonical_params[e]

        if not e.args:
            return e

        return e.func(*[visit(arg) for arg in e.args])

    expr = visit(expr)

    # Let SymPy canonicalize commutative Add/Mul ordering.
    expr = sp.expand_mul(expr)

    return sp.srepr(expr)

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

    resid = (y_true - y_pred).real
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
    
def load_rphi_covariance_for_event(track_folder: str, event_num: int):
    """
    Detect the new r-phi/z detector-noise model.

    Returns:
        has_rphi_covariance, phi, sigma_rphi, sigma_z, n_hits

    Backwards compatible:
      - Old datasets without sigma_rphi return False + None arrays.
    """
    hits_path = os.path.join(
        track_folder,
        f"event{int(event_num)}-hits.csv"
    )

    if not os.path.exists(hits_path):
        raise FileNotFoundError(
            f"Hits CSV not found: {hits_path}"
        )

    df = pd.read_csv(hits_path)
    n_hits = len(df)

    required = {
        "x", "y", "z",
        "r", "phi",
        "sigma_rphi", "sigma_z",
    }

    if not required.issubset(df.columns):
        return False, None, None, None, n_hits

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)

    # Must match load_tracks.py ordering.
    order = nearest_neighbor_order(x, y, z, r)

    phi = df["phi"].to_numpy(dtype=float)[order]
    sigma_rphi = df["sigma_rphi"].to_numpy(dtype=float)[order]
    sigma_z = df["sigma_z"].to_numpy(dtype=float)[order]

    return True, phi, sigma_rphi, sigma_z, n_hits

def sympy_expr_to_pysr_guess(expr, *, pysr_var="x0"):
    expr = sp.sympify(expr)
    free = list(expr.free_symbols)
    if len(free) != 1:
        raise ValueError(f"Expected 1 free symbol; got {free}")
    expr = expr.xreplace({free[0]: sp.Symbol(pysr_var)})

    s = sp.sstr(expr).replace("**", "^")

    # 1) leading unary minus -> binary minus
    if s.lstrip().startswith("-"):
        s = "0" + s

    # 2) "+ -something" -> "- something" (binary minus)
    s = s.replace("+ -", "- ")

    # 3) "(-something" -> "(0-something"  (unary minus inside parentheses)
    s = s.replace("(-", "(0-")

    # 4) "sin(-x0)" or ", -x0" -> "sin(0-x0)" / ", 0-x0"
    s = s.replace(",-", ",0-").replace(", -", ", 0-")

    # 5) negative exponents: "x0^-1" or "x0^(-1)" -> "x0^(0-1)"
    s = re.sub(r"\^\-(\d+(\.\d+)?)", r"^(0-\1)", s)
    s = s.replace("^(-", "^(0-")

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
    if not USE_K:
        return 0
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
        yhat = np.asarray(model.predict(s))
        if np.iscomplexobj(yhat):
            yhat = yhat.real.astype(float)

        # don't use model.score (it will choke on complex); compute R2 ourselves
        R2 = r2_score_1d(y, yhat)
        if R2 > best_R2:
            best_R2 = R2
            print(f"New best R2 = {best_R2:.3f}")

        # chi2_red (only if weighted)
        if weighted:
            k = _estimate_num_params(model)  # best-effort
            chi2_red = _chi2_red(y, yhat, sigma, k_params=k)
            if chi2_red < best_chi2:
                best_chi2 = chi2_red
                print(f"New best chi2_red = {best_chi2:.3f}")

        # stopping
        if total_iters >= min_iters:
            if weighted:
                chi2_ok = best_chi2 <= 1.0
                r2_ok = best_R2 >= R2_THRESHOLD
                if (chi2_ok and r2_ok):
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

    system("rm -r *hall_of_fame*")
    system("rm -r outputs")
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

def template_with_seed_params(template):
    """
    Return a copy of template where seed_params, if present, override init_params.
    """
    t = dict(template)
    if "seed_params" in t and t["seed_params"] is not None:
        t["init_params"] = np.asarray(t["seed_params"], dtype=float)
    return t

def sync_template_sets(x_templates, y_templates, z_templates):
    """
    Merge x/y/z template libraries into one unique list (by template_signature),
    then assign that same merged list back to x, y, z.
    """
    merged = []
    seen = set()

    for t in x_templates + y_templates + z_templates:
        t_use = template_with_seed_params(t)
        sig = template_signature(t_use)
        if sig not in seen:
            seen.add(sig)
            merged.append(t_use)

    print(
        f"[sync] x={len(x_templates)}, y={len(y_templates)}, z={len(z_templates)} "
        f"-> merged unique={len(merged)}"
    )

    # separate list objects, same template contents
    return list(merged), list(merged), list(merged)
    
def sanitize_prediction(y_pred, *, imag_tol=1e-8, return_mask=False):
    """
    Convert predictions to a usable real float array.

    If return_mask=True:
        returns (y_clean, invalid_mask)
    else:
        returns y_clean or None (old behavior)

    invalid_mask = True where values are bad
    """
    y_pred = np.asarray(y_pred)

    invalid_mask = np.zeros_like(y_pred, dtype=bool)

    # Handle complex
    if np.iscomplexobj(y_pred):
        imag = np.abs(np.imag(y_pred))
        bad_imag = (~np.isfinite(imag)) | (imag > imag_tol)
        invalid_mask |= bad_imag
        y_pred = np.real(y_pred)

    # Cast to float
    y_pred = np.asarray(y_pred, dtype=float)

    # Non-finite (nan/inf)
    bad_real = ~np.isfinite(y_pred)
    invalid_mask |= bad_real

    if not return_mask:
        if np.any(invalid_mask):
            return None
        return y_pred

    # return_mask=True path
    if np.all(invalid_mask):
        return None, invalid_mask

    # Replace bad values with nan for inspection
    y_clean = y_pred.copy()
    y_clean[invalid_mask] = np.nan

    return y_clean, invalid_mask

def fit_template_to_data(template, s_data, y_data, *, sigma=None, timeout_seconds=None):
    """
    Fast/stable-ish local template fit.

    Returns:
      (metrics_dict, best_params, expr_fitted, y_pred)
    """
    template = dict(template)
    expr_param = template["expr"]
    s_sym      = template["s_sym"]
    param_syms = template["param_syms"]
    p0         = np.asarray(template["init_params"], dtype=float)

    s_data = np.asarray(s_data).ravel()
    y_data = np.asarray(y_data).ravel()

    if sigma is not None:
        sig = np.asarray(sigma).ravel()
        if sig.shape != y_data.shape:
            raise ValueError(f"sigma shape {sig.shape} must match y shape {y_data.shape}")
        sig = np.maximum(sig, 1e-12)
    else:
        sig = None

    f = sp.lambdify((s_sym, *param_syms), expr_param, "numpy")
    
    # Analytic Jacobian wrt fitted parameters
    jac_exprs = [sp.diff(expr_param, a) for a in param_syms]
    jac_func = sp.lambdify((s_sym, *param_syms), jac_exprs, "numpy")
    
    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds

    def check_timeout():
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("template fit timed out")

    def jacobian(p):
        check_timeout()
        with np.errstate(all="ignore"):
            cols = jac_func(s_data, *p)

        J_cols = []

        for c in cols:
            c = np.asarray(c, dtype=float)

            if c.shape == ():
                c = np.full_like(y_data, float(c), dtype=float)
            else:
                c = c.ravel()

            # Bad shape: use neutral finite-ish column, not all-zero whole matrix
            if c.shape != y_data.shape:
                c = np.zeros_like(y_data, dtype=float)

            # Replace nan/inf column entries locally
            c = np.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)

            # Clip insane derivative spikes
            c = np.clip(c, -1e6, 1e6)

            J_cols.append(c)

        J = np.vstack(J_cols).T

        if sig is not None:
            J = J / sig[:, None]
            J = np.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)
            J = np.clip(J, -1e6, 1e6)

        return J

    def better(a_metrics, b_metrics):
        if b_metrics is None:
            return True
        if sig is not None:
            return a_metrics["chi2_red"] < b_metrics["chi2_red"]
        return a_metrics["R2"] > b_metrics["R2"]

    def evaluate_params(p):
        check_timeout()
        with np.errstate(all="ignore"):
            y_pred = f(s_data, *p)

        y_pred = sanitize_prediction(y_pred)
        if y_pred is None:
            return None

        metrics = regression_metrics(y_data, y_pred, sigma=sig, ddof=len(p) * USE_K)
        subs_dict = {sym: val for sym, val in zip(param_syms, p)}
        expr_used = expr_param.subs(subs_dict)
        return metrics, np.asarray(p, dtype=float), expr_used, y_pred

    if len(param_syms) == 0:
        with np.errstate(all="ignore"):
            y_pred = f(s_data)

        y_pred = sanitize_prediction(y_pred)
        if y_pred is None:
            y_pred = np.full_like(y_data, np.nan, dtype=float)

        metrics = regression_metrics(y_data, y_pred, sigma=sig, ddof=0)
        return metrics, np.array([]), expr_param, y_pred

    has_seed = ("seed_params" in template) and (template["seed_params"] is not None)

    # tighter for seeded templates, looser otherwise
    scale = np.maximum(np.abs(p0), 1.0 if not has_seed else 0.5)
    width = 3.0 if not has_seed else 2.0
    lb = p0 - width * scale
    ub = p0 + width * scale

    def residuals(p):
        check_timeout()
        with np.errstate(all="ignore"):
            y_pred = f(s_data, *p)

        y_pred = sanitize_prediction(y_pred)
        if y_pred is None:
            # still a hard penalty, but much smaller than 1e6
            r = np.full_like(y_data, 1e3, dtype=float)
        else:
            r = (y_pred - y_data).astype(float)

        if sig is not None:
            r = r / sig

        if not np.all(np.isfinite(r)):
            return np.full_like(y_data, 1e3, dtype=float)

        return r

    best_result = None

    seed_result = evaluate_params(p0)
    if seed_result is not None:
        best_result = seed_result
    timed_out = False
    try:
#        check_timeout()
        res = least_squares(
            residuals,
            p0,
            jac=jacobian if USE_ANALYTIC_JAC else '2-point',
#            bounds=(lb, ub),
            loss="linear",
#            x_scale=scale,
            method="trf",
            max_nfev=MAX_TEMPLATE_NFEV,
        )
#        check_timeout()
        ls_result = evaluate_params(res.x)
        if ls_result is not None:
            if (best_result is None) or better(ls_result[0], best_result[0]):
                best_result = ls_result
    except TimeoutError:
        print("TimeoutError, proceeding to next template...")
        timed_out = True
    except Exception:
        pass

    if best_result is None:
        with np.errstate(all="ignore"):
            y_pred = f(s_data, *p0)

        y_pred = sanitize_prediction(y_pred)
        if y_pred is None:
            y_pred = np.full_like(y_data, np.nan, dtype=float)

        metrics = regression_metrics(y_data, y_pred, sigma=sig, ddof=len(p0) * USE_K)
        subs_dict = {sym: val for sym, val in zip(param_syms, p0)}
        expr_fitted = expr_param.subs(subs_dict)
        return metrics, np.asarray(p0, dtype=float), expr_fitted, y_pred, timed_out

    return best_result+(timed_out,)

def template_num_params(template):
    return len(template.get("param_syms", []))

def make_manual_template(expr, s_sym):
    param_syms = sorted(
        expr.free_symbols - {s_sym},
        key=lambda x: x.name,
    )

    bad = [
        p for p in param_syms
        if not re.fullmatch(r"a\d+", p.name)
    ]

    if bad:
        raise ValueError(
            f"Template parameters must be named a0, a1, ...; got {bad}"
        )

    return {
        "expr": expr,
        "s_sym": s_sym,
        "param_syms": param_syms,
        "init_params": np.ones(len(param_syms), dtype=float),
    }

def freeze_one_template_param(template, p_opt, freeze_idx):
    """
    Return a new template where one parameter is fixed to its fitted value.

    This reduces k by 1. Remaining parameters are re-numbered a0, a1, ...
    so the HTML / later k-counting stays clean.
    """
    template = dict(template)

    expr = template["expr"]
    s_sym = template["s_sym"]
    old_params = list(template["param_syms"])
    p_opt = np.asarray(p_opt, dtype=float)

    if freeze_idx < 0 or freeze_idx >= len(old_params):
        raise IndexError(f"freeze_idx={freeze_idx} outside 0..{len(old_params)-1}")

    # Fix this parameter numerically.
    expr = expr.subs({old_params[freeze_idx]: float(p_opt[freeze_idx])})

    # Keep the remaining parameters that actually survive in the expression.
    remaining_old = [
        a for j, a in enumerate(old_params)
        if j != freeze_idx and a in expr.free_symbols
    ]

    # Re-number parameters to a0, a1, ...
    new_params = [sp.Symbol(f"a{i}") for i in range(len(remaining_old))]
    renumber = {old: new for old, new in zip(remaining_old, new_params)}
    expr_new = expr.xreplace(renumber)

    # Initial values are the previous fitted values for the remaining params.
    old_to_value = {a: float(p_opt[j]) for j, a in enumerate(old_params)}
    init_params = np.array([old_to_value[a] for a in remaining_old], dtype=float)

    return {
        "expr": expr_new,
        "s_sym": s_sym,
        "param_syms": new_params,
        "init_params": init_params,
        "seed_params": init_params.copy(),
    }

def zero_one_template_param(template, p_opt, zero_idx):
    """
    Return a new template where one parameter is set identically to zero.

    Unlike freezing, this can remove entire terms/subexpressions and may
    therefore reduce k by more than 1.
    """
    template = dict(template)

    expr = template["expr"]
    s_sym = template["s_sym"]
    old_params = list(template["param_syms"])
    p_opt = np.asarray(p_opt, dtype=float)

    if zero_idx < 0 or zero_idx >= len(old_params):
        raise IndexError(f"zero_idx={zero_idx} outside 0..{len(old_params)-1}")

    # Set parameter exactly to zero; SymPy will eliminate dead terms.
    expr = expr.subs({old_params[zero_idx]: sp.Integer(0)})

    # Only retain parameters that still occur after zeroing.
    remaining_old = [a for a in old_params if a in expr.free_symbols]

    new_params = [sp.Symbol(f"a{i}") for i in range(len(remaining_old))]
    renumber = {old: new for old, new in zip(remaining_old, new_params)}
    expr_new = expr.xreplace(renumber)

    old_to_value = {a: float(p_opt[j]) for j, a in enumerate(old_params)}
    init_params = np.array([old_to_value[a] for a in remaining_old], dtype=float)

    return {
        "expr": expr_new,
        "s_sym": s_sym,
        "param_syms": new_params,
        "init_params": init_params,
        "seed_params": init_params.copy(),
    }

def metric_value_for_selection(metrics, weighted):
    """
    Smaller is better.
    Weighted fits: minimize chi2_red.
    Unweighted fits: maximize R2, so use -R2.
    """
    if weighted:
        return metrics["chi2_red"]
    return -metrics["R2"]


def prune_template_by_backward_freezing(
    template,
    s_data,
    y_data,
    sigma,
    target_k,
    *,
    initial_p_opt=None,
    initial_metrics=None,
    weighted=True,
    time_limit_seconds=45.0,
    candidate_timeout_seconds=4.0,
    max_steps=50,
    max_candidates_per_round=12,
    prune_mode="freeze"
):
    """
    Greedy backward selection by freezing fitted constants.

    At each step:
      - try freezing candidate parameters one at a time,
      - refit remaining free parameters,
      - keep the freeze that hurts the metric least,
      - repeat until k <= target_k or time runs out.

    This is not globally optimal. It is a diagnostic complexity-control pass.
    """
    start = time.perf_counter()

    current_template = template_with_seed_params(template)

    if initial_p_opt is None or initial_metrics is None:
        try:
            initial_metrics, initial_p_opt, _, _, _ = fit_template_to_data(
                current_template,
                s_data,
                y_data,
                sigma=sigma,
                timeout_seconds=candidate_timeout_seconds,
            )
        except Exception:
            return current_template, None, None, {
                "pruned": False,
                "reason": "initial fit failed",
                "k_initial": template_num_params(current_template),
                "k_final": template_num_params(current_template),
            }

    current_p = np.asarray(initial_p_opt, dtype=float)
    current_metrics = initial_metrics

    k_initial = template_num_params(current_template)
    history = []

    if k_initial <= target_k:
        return current_template, current_metrics, current_p, {
            "pruned": False,
            "reason": "already within target",
            "k_initial": k_initial,
            "k_final": k_initial,
            "history": history,
        }

    for step in range(max_steps):
        if template_num_params(current_template) <= target_k:
            break

        if time.perf_counter() - start > time_limit_seconds:
            print("[prune] time limit reached")
            break

        k_now = template_num_params(current_template)

        # Cheap candidate ordering:
        # try freezing smaller absolute fitted constants first.
        # This is not perfect, but it avoids testing 100 params per step.
        candidate_order = sorted(
            range(k_now),
            key=lambda j: abs(float(current_p[j])) if j < len(current_p) else np.inf,
        )

        candidate_order = candidate_order[:max_candidates_per_round]

        best_candidate = None
        best_score = np.inf

        for freeze_idx in candidate_order:
            remaining_time = time_limit_seconds - (time.perf_counter() - start)
            if remaining_time <= 0:
                break

            try:
                if prune_mode == "zero":
                    cand_template = zero_one_template_param(
                        current_template,
                        current_p,
                        freeze_idx,
                    )
                elif prune_mode == "freeze":
                    cand_template = freeze_one_template_param(
                        current_template,
                        current_p,
                        freeze_idx,
                    )
                else:
                    raise ValueError(f"Unknown PRUNE_MODE={prune_mode!r}")

                cand_metrics, cand_p, cand_expr, cand_pred, cand_timed_out = fit_template_to_data(
                    cand_template,
                    s_data,
                    y_data,
                    sigma=sigma,
                    timeout_seconds=min(candidate_timeout_seconds, remaining_time),
                )

                score = metric_value_for_selection(cand_metrics, weighted)

                if score < best_score:
                    best_score = score
                    best_candidate = {
                        "freeze_idx": freeze_idx,
                        "template": cand_template,
                        "metrics": cand_metrics,
                        "p_opt": cand_p,
                        "score": score,
                    }

            except Exception as e:
                continue

        if best_candidate is None:
            print("[prune] no valid freeze candidate found")
            break

        old_score = metric_value_for_selection(current_metrics, weighted)
        new_score = best_candidate["score"]

        k_new = template_num_params(best_candidate["template"])

        print(
            f"[prune] step={step}, k {k_now}->{k_new}, "
            f"param_idx={best_candidate['freeze_idx']}, "
            f"score {old_score:.3e}->{new_score:.3e}"
        )

        history.append({
            "step": step,
            "k_before": k_now,
            "k_after": k_new,
            "param_idx": best_candidate["freeze_idx"],
            "old_score": old_score,
            "new_score": new_score,
        })

        current_template = best_candidate["template"]
        current_metrics = best_candidate["metrics"]
        current_p = best_candidate["p_opt"]

    return current_template, current_metrics, current_p, {
        "pruned": template_num_params(current_template) < k_initial,
        "reason": "done",
        "k_initial": k_initial,
        "k_final": template_num_params(current_template),
        "history": history,
    }

def summarize_3d_families(families_x, families_y, families_z, r2_x_all, r2_y_all, r2_z_all, chi2_x_all, chi2_y_all, chi2_z_all, weighted=False):
    """
    Group tracks by 3D family triples (Fx, Fy, Fz),
    print text summary, and return LaTeX table string.
    """
    assert len(families_x) == len(families_y) == len(families_z) == \
           len(r2_x_all) == len(r2_y_all) == len(r2_z_all) == \
           len(chi2_x_all) == len(chi2_y_all) == len(chi2_z_all), \
           "All family and metric arrays must have the same length."

    n_tracks = len(families_x)

    # Collect stats per triple
    triple_stats = defaultdict(lambda: {
        "count": 0,
        "r2x": [],
        "r2y": [],
        "r2z": [],
        "chi2x": [],
        "chi2y": [],
        "chi2z": [],
        "indices": []
    })

    for i in range(n_tracks):
        triple = (families_x[i], families_y[i], families_z[i])
        triple_stats[triple]["count"]  += 1
        triple_stats[triple]["r2x"].append(r2_x_all[i])
        triple_stats[triple]["r2y"].append(r2_y_all[i])
        triple_stats[triple]["r2z"].append(r2_z_all[i])
        triple_stats[triple]["chi2x"].append(chi2_x_all[i])
        triple_stats[triple]["chi2y"].append(chi2_y_all[i])
        triple_stats[triple]["chi2z"].append(chi2_z_all[i])
        triple_stats[triple]["indices"].append(i)

    # Sort triples in a deterministic way
    unique_triples = sorted(triple_stats.keys())

    # Assign a simple integer ID to each triple for nicer reporting
    triple_id_map = {triple: k for k, triple in enumerate(unique_triples)}

    # -------- LaTeX table --------
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{c c c c c c c c}")
    lines.append(r"\hline")
    if weighted:
        lines.append(r"Family & $F_x$ & $F_y$ & $F_z$ & \#~Tracks & Mean $\chi^2_{\nu,x}$ & Mean $\chi^2_{\nu,y}$ & Mean $\chi^2_{\nu,z}$ \\")
    else:
        lines.append(r"Family & $F_x$ & $F_y$ & $F_z$ & \#~Tracks & Mean $R_x^2$ & Mean $R_y^2$ & Mean $R_z^2$ \\")
    lines.append(r"\hline")

    for triple in unique_triples:
        fid = triple_id_map[triple]
        data = triple_stats[triple]
        if weighted:
            arr_x = np.array(data["chi2x"])
            arr_y = np.array(data["chi2y"])
            arr_z = np.array(data["chi2z"])
        else:
            arr_x = np.array(data["r2x"])
            arr_y = np.array(data["r2y"])
            arr_z = np.array(data["r2z"])

        mean_x = arr_x.mean()
        mean_y = arr_y.mean()
        mean_z = arr_z.mean()

        lines.append(
            fr"{fid} & {triple[0]} & {triple[1]} & {triple[2]} & "
            fr"{data['count']} & {mean_x:.3f} & {mean_y:.3f} & {mean_z:.3f} \\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary of 3D track families, with indices $(F_x, F_y, F_z)$ "
                 r"referring to the 1D template families for $x(s)$, $y(s)$, and $z(s)$.}")
    lines.append(r"\end{table}")

    latex_table_3d = "\n".join(lines)

    return latex_table_3d, triple_id_map

def nearest_neighbor_order(x, y, z, r=None):
    xyz = np.column_stack([x, y, z])
    n = len(xyz)

    if n <= 1:
        return np.arange(n)

    if r is None:
        r = np.sqrt(np.asarray(x)**2 + np.asarray(y)**2)

    remaining = set(range(n))
    current = int(np.argmin(r))

    order = [current]
    remaining.remove(current)

    while remaining:
        candidates = np.array(sorted(remaining), dtype=int)

        d2 = np.sum(
            (xyz[candidates] - xyz[current])**2,
            axis=1
        )

        current = int(candidates[np.argmin(d2)])
        order.append(current)
        remaining.remove(current)

    return np.asarray(order, dtype=int)

def covariance_chi2(
    x_obs, y_obs, z_obs,
    x_pred, y_pred, z_pred,
    phi_obs,
    sigma_rphi,
    sigma_z,
):
    """
    Project Cartesian residual onto the measured tangential direction.

    Per hit:
        chi2_i =
            (delta_tangent / sigma_rphi)^2
          + (delta_z       / sigma_z)^2

    This is equivalent to using the Moore-Penrose pseudoinverse
    of the singular x-y covariance induced by pure r*phi smearing.
    """

    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)

    x_pred = np.asarray(x_pred, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    z_pred = np.asarray(z_pred, dtype=float)

    phi_obs = np.asarray(phi_obs, dtype=float)

    sigma_rphi = np.broadcast_to(
        np.asarray(sigma_rphi, dtype=float),
        x_obs.shape
    )

    sigma_z = np.broadcast_to(
        np.asarray(sigma_z, dtype=float),
        z_obs.shape
    )

    dx = x_obs - x_pred
    dy = y_obs - y_pred
    dz = z_obs - z_pred

    # Local tangential unit vector:
    # t_hat = (-sin(phi), cos(phi))
    d_tangent = -np.sin(phi_obs) * dx + np.cos(phi_obs) * dy

    chi2_t = np.sum((d_tangent / sigma_rphi)**2)
    chi2_z = np.sum((dz / sigma_z)**2)

    return float(chi2_t + chi2_z)

def fit_helix_covariance(
    x, y, z,
    phi_detector,
    sigma_rphi,
    sigma_z,
):
    """
    Fit a 5-parameter helix:
        xc, yc, R, z0, dz/dphi

    Returns predictions and covariance-aware chi2.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    n = len(x)

    xc0 = np.mean(x)
    yc0 = np.mean(y)
    R0 = np.median(
        np.sqrt((x - xc0)**2 + (y - yc0)**2)
    )

    def circle_resid(p):
        xc, yc, R = p
        return np.sqrt(
            (x - xc)**2 + (y - yc)**2
        ) - R

    res_xy = least_squares(
        circle_resid,
        [xc0, yc0, R0],
        loss="linear",
    )

    xc, yc, R = res_xy.x

    # Intrinsic phase around fitted helix center
    phi_h = np.unwrap(
        np.arctan2(y - yc, x - xc)
    )

    # z = b + a*phi_h
    A = np.vstack([
        np.ones_like(phi_h),
        phi_h
    ]).T

    # sigma_z is constant in this sample, but retain weighting
    sz = np.broadcast_to(
        np.asarray(sigma_z, dtype=float),
        z.shape
    )

    w = 1.0 / sz
    Aw = A * w[:, None]
    zw = z * w

    b, a = np.linalg.lstsq(
        Aw, zw, rcond=None
    )[0]

    x_pred = xc + R * np.cos(phi_h)
    y_pred = yc + R * np.sin(phi_h)
    z_pred = b + a * phi_h

    chi2 = covariance_chi2(
        x, y, z,
        x_pred, y_pred, z_pred,
        phi_detector,
        sigma_rphi,
        sigma_z,
    )

    # two noisy directions/hit, 5 fitted helix parameters
    dof = max(2*n - 5, 1)
    chi2_red = chi2 / dof

    return {
        "x": x_pred,
        "y": y_pred,
        "z": z_pred,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "dof": dof,
        "xc": xc,
        "yc": yc,
        "R": R,
        "b": b,
        "a": a,
    }

def fit_best_split_helix_covariance(
    x, y, z,
    phi_detector,
    sigma_rphi,
    sigma_z,
    min_hits=5,
):
    n = len(x)

    if n < 2 * min_hits:
        raise ValueError(
            f"Need at least {2*min_hits} hits for split helix; got {n}"
        )

    best = None

    # split_idx = first hit belonging to right-hand helix
    for split_idx in range(min_hits, n - min_hits + 1):
        left = slice(0, split_idx)
        right = slice(split_idx, n)

        try:
            fit_left = fit_helix_covariance(
                x[left], y[left], z[left],
                phi_detector[left],
                sigma_rphi[left],
                sigma_z[left],
            )

            fit_right = fit_helix_covariance(
                x[right], y[right], z[right],
                phi_detector[right],
                sigma_rphi[right],
                sigma_z[right],
            )
        except Exception:
            continue

        chi2 = fit_left["chi2"] + fit_right["chi2"]
        dof = fit_left["dof"] + fit_right["dof"]
        chi2_red = chi2 / max(dof, 1)

        if best is None or chi2_red < best["chi2_red"]:
            best = {
                "split_idx": split_idx,
                "left": fit_left,
                "right": fit_right,
                "chi2": chi2,
                "dof": dof,
                "chi2_red": chi2_red,
            }

    if best is None:
        raise RuntimeError("No valid split-helix fit found")

    return best

def cpp_expr_to_sympy(expr_str):
    """
    Convert C++ SR output to a SymPy expression.

    Conventions:
      x0 -> s
      ~  -> unary minus
      all numeric literals -> SymPy Floats, so make_parametric_template()
                              turns them into a0, a1, ...
    """
    s = sp.Symbol("s")

    # Your framework uses ~ for unary minus.
    expr_str = expr_str.replace("~", "-")

    # Make integer literals such as 32 into 32.0 so the existing
    # make_parametric_template() treats them as fitted constants too.
    #
    # Avoid touching the 0 in x0.
    number_re = re.compile(
        r"(?<![A-Za-z_])"
        r"(?<!\d)"
        r"(\d+\.\d*|\.\d+|\d+)"
        r"([eE][+-]?\d+)?"
        r"(?![A-Za-z_\d])"
    )

    def force_float(m):
        token = m.group(0)
        return token if ("." in token or "e" in token.lower()) else token + ".0"

    expr_str = number_re.sub(force_float, expr_str)

    locals_dict = {
        "x0": s,
        "s": s,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "tanh": sp.tanh,
        "sech": sp.sech,
        "sqrt": sp.sqrt,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
    }

    return sp.sympify(
        expr_str,
        locals=locals_dict,
    )

if __name__ == '__main__':
    np.random.seed(0)  # Fix random seed
    def safe_sech(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = np.abs(x) < 700
        out[mask] = 1.0 / np.cosh(x[mask])
        return out
    np.sech = safe_sech
#    np.sech = lambda x: 1/np.cosh(x)
    TEMPLATE_FIT_TIMEOUT_SECONDS = 20.0
    MAX_TEMPLATE_NFEV = None
    
    REQUIRE_SR_CONSTRAINED = True
    # Coordinate-wise pruning target.
    # For n_hits=25 and margin=10:
    #   target per coordinate = floor((3*25 - 10)/3) = 21
    # so a 32-param coord template gets pruned toward <=21 params.
    # For rphi-covariance data:
    # target per coordinate = floor((2*n_hits - SR_DOF_MARGIN_TOTAL)/3)
    SR_DOF_MARGIN_TOTAL = 10

    def sr_target_k_coord(n_hits, has_rphi_covariance):
        """
        Legacy datasets: 3 independent coordinate observations/hit.
        rphi-covariance datasets: 2 noisy directions/hit (tangent + z).
        """
        n_obs_per_hit = 2 if has_rphi_covariance else 3

        return max(
            1,
            int(
                (
                    n_obs_per_hit * n_hits
                    - SR_DOF_MARGIN_TOTAL
                ) // 3
            )
        )
    
    PRUNE_UNDERCONSTRAINED_TEMPLATES = False
    PRUNE_MODE = "zero"   # "zero" or "freeze"
    PRUNE_TIME_LIMIT_SECONDS = 120.
    PRUNE_CANDIDATE_TIMEOUT_SECONDS = 4.0
    PRUNE_MAX_STEPS = 5
    PRUNE_MAX_CANDIDATES_PER_ROUND = 12
    
    PROFILE_TEMPLATE_FITS = True
    fit_time_records = []
    OPEN_PNGS = False
    OPEN_HTML = True
    ADD_TEMPLATE_ONLY = False
    ADD_TEMPLATE_COORD = "all"
    TEMPLATE_PATH = "track_templates.pkl"
    SYNC_TEMPLATES = False
    printTemplatesOnly = False
    #templates_to_delete = {"x_templates": set(range(50,110)), "y_templates": range(50,110), "z_templates": range(50,110)}
    R2_THRESHOLD = 0.997
    RunPySR = False   # Whether to enable (True) or disable (False) PySR discovery
    MaxPySRIters = 100
    num_tracks = 100

    CPP_TEMPLATE = r"""
    ((sin(tanh((x0 * 9.839682808358576))) * -17.43230415238745) + (((45.82850715099027 * arcsin(x0)) * sech(cos(asin(x0)))) + (166.85074093671616 * (tanh(sin(x0)) * (1.346327831726017 - (2.2583182429038966 * sin(x0)))))))
    """

    s = sp.Symbol("s")
    ADD_TEMPLATE_EXPR = cpp_expr_to_sympy(CPP_TEMPLATE)

    R2_THRESHOLD = 0.997
    RunPySR = False   # Whether to enable (True) or disable (False) PySR discovery
    MaxPySRIters = 100
    num_tracks = 100
#    FIT_MODE = "SR"
    FIT_MODE = "SR"
    if FIT_MODE not in {"SR", "SPLIT_HELIX"}:
        raise ValueError(f"Unknown FIT_MODE={FIT_MODE!r}")
    SPLIT_MIN_HITS = 5
    loaded = {}
    x_templates, y_templates, z_templates = [], [], []
    
    track_dataset_idx = 18
    out_html = [
        #legacy
        #------
        "v20260122_163839__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.html", #old
        "v1_noiseless_69000.html", #old
        "v20260202_142140__train10_test10__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel.html", #old
        #10 tracks
        #---------
        "v20260305_002410__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.html", #25-mode 10 tracks
        "v20260407_091434__train5_test5__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel.html", #helix 10-tracks
        "v20260430_092453__train5_test5__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.html", #5-mode 10 tracks
        "v20260430_193549__train5_test5__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01__randomNoise.html", #random-noise 10 tracks
        #100 tracks
        #----------
        "v20260518_131139__train50_test50__layers25_len320p0__r3p1-53p0__fd5-5__func3-3__noiseXY0p01_Z0p01.html", #5-mode 100 tracks
        "v20260518_142036__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01.html", #25-mode 100 tracks
        "v20260518_142850__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__standardModel.html", #helix 100 tracks
        "v20260519_110640__train50_test50__layers25_len320p0__r3p1-53p0__fd25-25__func3-3__noiseXY0p01_Z0p01__randomNoise.html", #random-noise 100 tracks
        "Levi_dominant_purity_lt_0p6_minhits_20.html", #low purity <= 0.6 & min_hits >= 20 finder output
        "Levi_reco_nonhelical_PID15_purity_ge_0p9_eff_ge_0p5_minhits_20.html", #purity >= 0.9 & min_hits >= 20 & eff >= 0.5 non-helical finder output
        "Levi_SM_high_purity_minhits_20.html", #high SM purity >= 0.9 & min_hits >= 20 finder output
        "Levi_impure_purity_lt_0p9_minhits_20.html", #purity <= 0.9 & min_hits >= 20 finder output,
        #Finder-output 100 tracks with pileup <mu=10>
        "mu10_finder_SM_100.html",
        "mu10_finder_PID15_100.html",
        "mu10_finder_fakes_100.html",
        #Finder-output 100 tracks with pileup <mu=20>, ~2% dropout probability, and gaussian-smearing σ_{rɸ} = 0.003 cm, σ_z = 0.01 cm
        "mu20_finder_SM_100.html",
        "mu20_finder_PID15_100.html",
        "mu20_finder_fakes_100.html",
    ]
    track_folder = [f"../tracks_for_ed/{i[:-5]}" for i in out_html]
    dataset_labels = [
        #legacy
        #------
        "v20260122_163839 train/test (noise XY=0.01, Z=0.01) Fourier-Dim = 25",
        "v1_noiseless_69000 (no sigma columns)",
        "v20260202_142140 train/test (noise XY=0.01, Z=0.01) Standard Model",
        #10 tracks
        #---------
        "v20260305_002410 train/test (noise XY=0.01, Z=0.01) Fourier-Dim = 25",
        "v20260407_091434 train/test (noise XY=0.01, Z=0.01) Standard Model",
        "v20260430_092453 train/test (noise XY=0.01, Z=0.01) Fourier-Dim = 5",
        "v20260430_193549 train/test (noise XY=0.01, Z=0.01) Random Noise",
        #100 tracks
        #----------
        "v20260518_131139 train/test (noise XY=0.01, Z=0.01) Fourier-Dim = 5",
        "v20260518_142036 train/test (noise XY=0.01, Z=0.01) Fourier-Dim = 25",
        "v20260518_142850 train/test (noise XY=0.01, Z=0.01) Standard Model",
        "v20260519_110640 train/test (noise XY=0.01, Z=0.01) Random Noise",
        "Levi_dominant_purity_lt_0p6_minhits_20 (noise XY=0.01, Z=0.01) True Fakes",
        "Levi_reco_nonhelical_PID15_purity_ge_0p9_eff_ge_0p5_minhits_20 (noise XY=0.01, Z=0.01) Nonhelical Tracks",
        "Levi_SM_high_purity_minhits_20.html (noise XY=0.01, Z=0.01) Standard Model Tracks",
        "Levi_impure_purity_lt_0p9_minhits_20 (noise XY=0.01, Z=0.01) True Fakes",
        #Finder-output 100 tracks with pileup <mu=10>
        r"$\mu=10$ finder-output matched Standard Model",
        r"$\mu=10$ finder-output matched PID15",
        r"$\mu=10$ finder-output background-only fakes",
        #Finder-output 100 tracks with pileup <mu=20>, ~2% dropout probability, and gaussian-smearing σ_{rɸ} = 0.003 cm, σ_z = 0.01 cm
        r"$\mu=20$ finder-output matched Standard Model",
        r"$\mu=20$ finder-output matched PID15",
        r"$\mu=20$ finder-output background-only fakes",
    ]
#    print(out_html, track_folder, dataset_labels, sep='\n');
    track_folder = track_folder[track_dataset_idx]
    dataset_label = dataset_labels[track_dataset_idx]
    out_html = out_html[track_dataset_idx]

    if FIT_MODE == "SPLIT_HELIX":
        out_html = out_html.replace(
            ".html",
            "__split_helix.html"
        )
        dataset_label += " — split helix"
    
    outfile_str = out_html[:-5]
    
    independent_param = "arc"
    print(f"independent_param = {independent_param}")
    if FIT_MODE == "SR" and os.path.exists(TEMPLATE_PATH):
        print("Loading existing template library...")
        with open(TEMPLATE_PATH, "rb") as f:
            loaded = pickle.load(f)

        x_templates = loaded["x_templates"]
        y_templates = loaded["y_templates"]
        z_templates = loaded["z_templates"]
        
        if printTemplatesOnly:
            import sympy as sp

            # ---- print templates with indices ----
            for coord_name, templates in (
                ("x_templates", x_templates),
                ("y_templates", y_templates),
                ("z_templates", z_templates),
            ):
                print(f"\n=== {coord_name} ({len(templates)}) ===")
                for i, t in enumerate(templates):
                    expr = t.get("expr", None)
                    nparams = len(t.get("param_syms", []))

                    if expr is not None:
                        expr_s = sp.sstr(expr)
                        temp = sp.symbols(f"f_{i}")
                        expr_latex = sp.multiline_latex(temp, expr, 2)
                    else:
                        expr_s = "<no expr>"
                        expr_latex = "<no expr>"

                    num_chars = 1500
                    expr_s = (expr_s[:num_chars] + "...") if len(expr_s) > num_chars else expr_s
                    
                    print(f"[{i:3d}] nparams = {nparams:2d}:  {expr_s}")
                    #print(f"      latex:\n{expr_latex}")
                    print()

            if templates_to_delete:
                # ---- delete selected indices + write back ----
                import shutil
                shutil.copyfile(TEMPLATE_PATH, TEMPLATE_PATH + ".bak")
                print(f"\nBackup created: {TEMPLATE_PATH}.bak")

                x_bad = set(templates_to_delete.get("x_templates", set()))
                y_bad = set(templates_to_delete.get("y_templates", set()))
                z_bad = set(templates_to_delete.get("z_templates", set()))

                x_before, y_before, z_before = len(x_templates), len(y_templates), len(z_templates)

                x_templates = [t for i, t in enumerate(x_templates) if i not in x_bad]
                y_templates = [t for i, t in enumerate(y_templates) if i not in y_bad]
                z_templates = [t for i, t in enumerate(z_templates) if i not in z_bad]

                print(f"Deleted {x_before - len(x_templates)} from x_templates: {sorted(x_bad)}")
                print(f"Deleted {y_before - len(y_templates)} from y_templates: {sorted(y_bad)}")
                print(f"Deleted {z_before - len(z_templates)} from z_templates: {sorted(z_bad)}")

                with open(TEMPLATE_PATH, "wb") as f:
                    pickle.dump(
                        {"x_templates": x_templates, "y_templates": y_templates, "z_templates": z_templates},
                        f,
                    )
                print(f"Saved updated template library to: {TEMPLATE_PATH}")
            exit()
    else:
        print('No template file found — starting with empty template lists.' if FIT_MODE=='SR' else 'No template file needed')
    
    if ADD_TEMPLATE_ONLY:
        import shutil

        if ADD_TEMPLATE_COORD not in {"x", "y", "z", "all"}:
            raise ValueError(
                f"ADD_TEMPLATE_COORD must be x/y/z/all, "
                f"got {ADD_TEMPLATE_COORD!r}"
            )

        template = make_parametric_template(
            ADD_TEMPLATE_EXPR,
            s_name="s",
        )

        targets = {
            "x": [x_templates],
            "y": [y_templates],
            "z": [z_templates],
            "all": [
                x_templates,
                y_templates,
                z_templates,
            ],
        }[ADD_TEMPLATE_COORD]

        new_sig = template_signature(template)

        added = 0

        for templates in targets:
            already_exists = any(
                template_signature(t) == new_sig
                for t in templates
            )

            if not already_exists:
                templates.append(template.copy())
                added += 1

        if added == 0:
            print("Template already exists; nothing changed.")
            raise SystemExit

        # .bak is the OLD library, not the one we modify.
        if os.path.exists(TEMPLATE_PATH):
            shutil.copyfile(
                TEMPLATE_PATH,
                TEMPLATE_PATH + ".bak",
            )
            print(
                f"Backup created: {TEMPLATE_PATH}.bak"
            )

        with open(TEMPLATE_PATH, "wb") as f:
            pickle.dump(
                {
                    "x_templates": x_templates,
                    "y_templates": y_templates,
                    "z_templates": z_templates,
                },
                f,
            )

        print(
            f"Added template to {ADD_TEMPLATE_COORD}:"
        )
        print(f"    {ADD_TEMPLATE_EXPR}")
        print(
            f"n_params = {template_num_params(template)}"
        )
        print(f"Saved to {TEMPLATE_PATH}")

        raise SystemExit
#        z_templates.append(make_parametric_template(-34.520199*sp.sech(s**(-1.45342) - 3.00504026595319), s_name="s"))
    track_infos = []  # store per-track info for HTML
    x_eqns, y_eqns, z_eqns = [], [], []   # store final numeric expressions used
    families_x, families_y, families_z = [], [], []  # indices of template used
    r2_x_all, r2_y_all, r2_z_all = [], [], [] # R^2 values for each track and coordinate
    chi2_x_all, chi2_y_all, chi2_z_all = [], [], []
    chi2_sr_track_all = []
    chi2_helix_track_all = []
    log_chi2_ratio_all = []
    split_infos = []
    chi2_full_helix_all = []
    chi2_split_helix_all = []
    log_split_ratio_all = []

    # now load sigma arrays too
    S, X, Y, Z, SIG_X_LIST, SIG_Y_LIST, SIG_Z_LIST, F = load_many_tracks(track_folder, offset = 0, max_tracks=num_tracks, param = independent_param)
    CHECK_ORDER = False
    if CHECK_ORDER:
        print("\n=== HIT-ORDER DIAGNOSTIC ===")

        n_exact = 0
        n_reverse = 0
        n_other = 0

        for i, (x, y, z, f) in enumerate(zip(X, Y, Z, F)):

            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)

            r = np.sqrt(x**2 + y**2)

            order = nearest_neighbor_order(x, y, z, r)

            identity = np.arange(len(x))
            reverse = identity[::-1]

            if np.array_equal(order, identity):
                status = "EXACT"
                n_exact += 1

            elif np.array_equal(order, reverse):
                status = "REVERSE"
                n_reverse += 1

            else:
                status = "DIFFERENT"
                n_other += 1

            print(
                f"{i:3d} {f}: "
                f"{status}, "
                f"n_hits={len(x)}, "
                f"order={order.tolist()}"
            )

        print("\n=== SUMMARY ===")
        print("exact:", n_exact)
        print("reverse:", n_reverse)
        print("different:", n_other)
        print(
            "fraction preserved up to reversal:",
            (n_exact + n_reverse) / len(X)
        )

        exit()
    num_tracks = len(S)
    assert((len(S), len(X), len(Y), len(Z), len(F)) == (num_tracks, num_tracks, num_tracks, num_tracks, num_tracks))
#    print(f"len(S), len(X), len(Y), len(Z), len(F) = {(len(S), len(X), len(Y), len(Z), len(F))}")
#    print(SIG_X_LIST[0], SIG_Y_LIST[0], SIG_Z_LIST[0]); exit()
    
    model_kwargs = dict(
        niterations=100,
        binary_operators=["+", "-", "*",  "^"],
        unary_operators=["neg", "sin", "cos", "tanh", "sqrt", "sech", "asin", "acos"],
        extra_sympy_mappings={"sech": lambda x: sp.sech(x)},
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

    #THE BIG TRACK LOOP!
    for track_idx, (s_all, x_all, y_all, z_all, f_all,
                sigx_from_loader, sigy_from_loader, sigz_from_loader) in enumerate(
                    zip(S, X, Y, Z, F, SIG_X_LIST, SIG_Y_LIST, SIG_Z_LIST)):
        event_num = parse_event_number(f_all)

        (
            has_rphi_covariance,
            phi_detector,
            sigma_rphi,
            sigma_z_cov,
            n_hits_cov,
        ) = load_rphi_covariance_for_event(
            track_folder,
            event_num
        )

        if has_rphi_covariance:
            print(
                f"[noise model] event {f_all}: "
                f"using rphi/z covariance "
                f"(2 stochastic observations per hit)"
            )
        else:
            print(
                f"[noise model] event {f_all}: "
                f"legacy uncertainty handling"
            )
        
        if FIT_MODE == "SPLIT_HELIX":

            if not has_rphi_covariance:
                raise RuntimeError(
                    "SPLIT_HELIX mode currently requires "
                    "sigma_rphi and sigma_z columns."
                )

            if len(s_all) != n_hits_cov:
                raise RuntimeError(
                    f"[event {f_all}] hit-count mismatch: "
                    f"{len(s_all)} loaded vs {n_hits_cov} CSV"
                )

            full_fit = fit_helix_covariance(
                x_all, y_all, z_all,
                phi_detector,
                sigma_rphi,
                sigma_z_cov,
            )

            split_fit = fit_best_split_helix_covariance(
                x_all, y_all, z_all,
                phi_detector,
                sigma_rphi,
                sigma_z_cov,
                min_hits=SPLIT_MIN_HITS,
            )

            split_idx = split_fit["split_idx"]

            chi2nu_full = full_fit["chi2_red"]
            chi2nu_split = split_fit["chi2_red"]

            log_ratio = np.log(
                (chi2nu_split + 1e-300)
                /
                (chi2nu_full+ 1e-300)
            )
            
            x_split = np.empty_like(x_all)
            y_split = np.empty_like(y_all)
            z_split = np.empty_like(z_all)

            x_split[:split_idx] = split_fit["left"]["x"]
            y_split[:split_idx] = split_fit["left"]["y"]
            z_split[:split_idx] = split_fit["left"]["z"]

            x_split[split_idx:] = split_fit["right"]["x"]
            y_split[split_idx:] = split_fit["right"]["y"]
            z_split[split_idx:] = split_fit["right"]["z"]
            
            hit_idx = np.arange(len(x_all))

            fig, axes = plt.subplots(
                3, 1,
                figsize=(9, 10),
                sharex=True
            )

            fig.suptitle(
                fr"best split={split_idx}, "
                fr"$\chi^2_{{\nu,\mathrm{{helix}}}}={chi2nu_full:.3e}$, "
                fr"$\chi^2_{{\nu,\mathrm{{split}}}}={chi2nu_split:.3e}$, "
                fr"$\log(\chi^2_{{\nu,\mathrm{{split}}}}/"
                fr"\chi^2_{{\nu,\mathrm{{helix}}}})={log_ratio:.3f}$"
            )

            coords = [
                ("x", x_all, full_fit["x"], x_split),
                ("y", y_all, full_fit["y"], y_split),
                ("z", z_all, full_fit["z"], z_split),
            ]

            for ax, (name, obs, full_pred, split_pred) in zip(
                axes, coords
            ):
                ax.scatter(
                    hit_idx, obs,
                    s=16,
                    label="hits"
                )

                ax.plot(
                    hit_idx,
                    full_pred,
                    "--",
                    linewidth=1.5,
                    label="single helix"
                )

                ax.plot(
                    hit_idx[:split_idx],
                    split_pred[:split_idx],
                    linewidth=2,
                    label="split helix 1"
                )

                ax.plot(
                    hit_idx[split_idx:],
                    split_pred[split_idx:],
                    linewidth=2,
                    label="split helix 2"
                )

                ax.axvline(
                    split_idx - 0.5,
                    linestyle=":",
                    linewidth=1
                )

                ax.set_ylabel(name)
                ax.legend(fontsize=8)

            axes[-1].set_xlabel("ordered hit index")

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            img_str = (
                f"../pngs/SPLIT_HELIX_track_"
                f"{outfile_str}{f_all}.png"
            )

            plt.savefig(img_str, dpi=5*96)
            plt.close()
            
            chi2_full_helix_all.append(chi2nu_full)
            chi2_split_helix_all.append(chi2nu_split)
            log_split_ratio_all.append(log_ratio)

            split_infos.append({
                "track_idx": track_idx,
                "event_id": str(f_all),
                "png": img_str,
                "split_idx": split_idx,
                "chi2nu_full": chi2nu_full,
                "chi2nu_split": chi2nu_split,
                "log_ratio": log_ratio,
            })

            print(
                f"[Split helix] {f_all}: "
                f"split={split_idx}, "
                f"full={chi2nu_full:.3e}, "
                f"split={chi2nu_split:.3e}, "
                f"log ratio={log_ratio:.3f}"
            )

            continue

        # --- Helper inner function to handle one coordinate --- #
        def fit_dimension(s_data, y_data, templates, eqn_list, families_list, coord_name, *, sigma=None):
            print(f"\n--- Fitting Track {track_idx}, coordinate {coord_name} ---")

            # Selection logic:
            #  - If sigma is provided: prefer reduced chi^2 (chi2_red) closest to 1
            #  - Else: prefer highest R^2 and require R2_THRESHOLD for acceptance
            def is_weighted():
                return sigma is not None

            def better(a, b):
                """Return True if metrics dict a is better than b under the active criterion."""
                if b is None:
                    return True
                if is_weighted():
                    return a["chi2_red"] < b["chi2_red"]
                else:
                    return a["R2"] > b["R2"]

            def accept_template(expr, metrics, template_index, template_obj, p_opt):
                """
                Accept an already-selected template.

                If REQUIRE_SR_CONSTRAINED=True, this function assumes any necessary
                pruning has already happened during candidate evaluation and only
                performs a final constrainedness check.
                """

                if template_obj is None:
                    raise RuntimeError(
                        f"[Track {track_idx} {coord_name}] selected template_obj is None."
                    )

                if REQUIRE_SR_CONSTRAINED:
                    n_hits_here = len(s_data)
                    target_k_coord = sr_target_k_coord(n_hits_here,has_rphi_covariance)

                    k_selected = template_num_params(template_obj)

                    if k_selected > target_k_coord:
                        raise RuntimeError(
                            f"[Track {track_idx} {coord_name}] attempted to accept "
                            f"underconstrained template: "
                            f"k={k_selected} > target_k_coord={target_k_coord}"
                        )

                # If pruning created a new template object that is not already the
                # original template at template_index, add the derived template.
                if (
                    template_index is None
                    or template_index >= len(templates)
                    or template_obj is not templates[template_index]
                ):
                    templates.append(template_obj)
                    template_index = len(templates) - 1

                eqn_list.append(expr)
                families_list.append(template_index)

                return expr, metrics
                
            # Best template that already satisfies constrainedness.
            best_valid_metrics = None
            best_valid_expr = None
            best_valid_template_index = None
            best_valid_template_obj = None
            best_valid_p_opt = None

            # Best raw fit among templates that violate constrainedness.
            # We will try pruning ONLY this one after scanning the whole library.
            best_prune_metrics = None
            best_prune_expr = None
            best_prune_template_index = None
            best_prune_template_obj = None
            best_prune_p_opt = None

            # 1) Try existing templates
            if templates:
                for idx, template in enumerate(templates):
                    try:
                        n_hits_here = len(s_data)
                        target_k_coord = sr_target_k_coord(n_hits_here,has_rphi_covariance)
                        if REQUIRE_SR_CONSTRAINED and not PRUNE_UNDERCONSTRAINED_TEMPLATES:
                            if template_num_params(template) > target_k_coord:
                                continue
                        k_here = template_num_params(template)
                        # --------------------------------------------------
                        # Decide whether this template is even worth fitting.
                        # --------------------------------------------------
                        if REQUIRE_SR_CONSTRAINED:
                            if k_here > target_k_coord:

                                # If pruning is disabled, underconstrained templates
                                # are completely ineligible.
                                if not PRUNE_UNDERCONSTRAINED_TEMPLATES:
                                    continue

                                # If pruning is enabled, only fit templates that can
                                # plausibly be brought within the target.
                                min_required_steps = k_here - target_k_coord

                                if PRUNE_MODE == "freeze":
                                    # Exact: freezing removes exactly one parameter/step.
                                    if min_required_steps > PRUNE_MAX_STEPS:
                                        continue

                                elif PRUNE_MODE == "zero":
                                    # Conservative cheap filter:
                                    # zeroing removes at least one parameter/step,
                                    # and sometimes several.
                                    if min_required_steps > PRUNE_MAX_STEPS:
                                        continue

                                else:
                                    raise ValueError(f"Unknown PRUNE_MODE={PRUNE_MODE!r}")

                        # --------------------------------------------------
                        # Only now pay for the actual template fit.
                        # --------------------------------------------------
                        t0 = time.perf_counter() if PROFILE_TEMPLATE_FITS else 0
                        metrics, p_opt, expr_fitted, y_pred, timed_out = fit_template_to_data(template, s_data, y_data, sigma=sigma, timeout_seconds = TEMPLATE_FIT_TIMEOUT_SECONDS)
                        n_hits_here = len(s_data)
                        target_k_coord = sr_target_k_coord(n_hits_here,has_rphi_covariance)

                        k_here = template_num_params(template)

                        if not REQUIRE_SR_CONSTRAINED:
                            # Normal unconstrained behavior.
                            if better(metrics, best_valid_metrics):
                                best_valid_metrics = metrics
                                best_valid_expr = expr_fitted
                                best_valid_template_index = idx
                                best_valid_template_obj = template
                                best_valid_p_opt = p_opt

                        elif k_here <= target_k_coord:
                            # Already valid: keep best such template.
                            if better(metrics, best_valid_metrics):
                                best_valid_metrics = metrics
                                best_valid_expr = expr_fitted
                                best_valid_template_index = idx
                                best_valid_template_obj = template
                                best_valid_p_opt = p_opt

                        elif PRUNE_UNDERCONSTRAINED_TEMPLATES:
                            # Underconstrained: do NOT prune now.
                            # Merely remember the best one for one pruning attempt after the scan.
                            if better(metrics, best_prune_metrics):
                                best_prune_metrics = metrics
                                best_prune_expr = expr_fitted
                                best_prune_template_index = idx
                                best_prune_template_obj = template
                                best_prune_p_opt = p_opt
                        if PROFILE_TEMPLATE_FITS:
                            dt = time.perf_counter() - t0
                            fit_time_records.append({
                                "track_idx": track_idx,
                                "coord": coord_name,
                                "template_idx": idx,
                                "seconds": dt,
                                "chi2_red": metrics.get("chi2_red", np.nan),
                                "R2": metrics.get("R2", np.nan),
                                "timed_out": timed_out
                            })
                            print(f"[timing] track={track_idx} coord={coord_name} template={idx} dt={dt:.3f}s")
                        if is_weighted():
                            print(f"Template {idx} metrics chi2_red = {metrics['chi2_red']}")
                        else:
                            print(f"Template {idx} metrics R2 = {metrics['R2']}")
                    except TimeoutError:
                        dt = time.perf_counter() - t0
                        print(f"[timeout] track={track_idx} coord={coord_name} "
                              f"template={idx} exceeded {TEMPLATE_FIT_TIMEOUT_SECONDS:.1f}s")
                        if PROFILE_TEMPLATE_FITS:
                            fit_time_records.append({
                                "track_idx": track_idx,
                                "coord": coord_name,
                                "template_idx": idx,
                                "seconds": dt,
                                "chi2_red": np.nan,
                                "R2": np.nan,
                                "timed_out": True,
                            })
                        continue
                    except ValueError as e:
                        # Often: "array must not contain infs or NaNs" from pathological templates
                        print(f"[Track {track_idx} {coord_name}] Skipping template {idx} due to ValueError: {e}")
                        continue
                    
                # ------------------------------------------------------------
                # After scanning ALL templates, optionally prune only the best
                # underconstrained candidate.
                # ------------------------------------------------------------

                best_metrics = best_valid_metrics
                best_expr = best_valid_expr
                best_template_index = best_valid_template_index
                best_template_obj = best_valid_template_obj
                best_p_opt = best_valid_p_opt

                if (
                    REQUIRE_SR_CONSTRAINED
                    and PRUNE_UNDERCONSTRAINED_TEMPLATES
                    and best_prune_template_obj is not None
                ):
                    k_before = template_num_params(best_prune_template_obj)

                    print(
                        f"[Track {track_idx} {coord_name}] best underconstrained candidate "
                        f"is template {best_prune_template_index}: "
                        f"k={k_before}, target<={target_k_coord}; attempting one pruning pass."
                    )

                    # Exact impossibility check for freezing.
                    can_attempt = True
                    if PRUNE_MODE == "freeze":
                        min_required_steps = k_before - target_k_coord
                        if min_required_steps > PRUNE_MAX_STEPS:
                            print(
                                f"[Track {track_idx} {coord_name}] cannot prune template "
                                f"{best_prune_template_index}: needs at least "
                                f"{min_required_steps} freeze steps > "
                                f"PRUNE_MAX_STEPS={PRUNE_MAX_STEPS}"
                            )
                            can_attempt = False

                    if can_attempt:
                        pruned_template, pruned_metrics, pruned_p, prune_info = \
                            prune_template_by_backward_freezing(
                                best_prune_template_obj,
                                s_data,
                                y_data,
                                sigma,
                                target_k_coord,
                                initial_p_opt=best_prune_p_opt,
                                initial_metrics=best_prune_metrics,
                                weighted=is_weighted(),
                                time_limit_seconds=PRUNE_TIME_LIMIT_SECONDS,
                                candidate_timeout_seconds=PRUNE_CANDIDATE_TIMEOUT_SECONDS,
                                max_steps=PRUNE_MAX_STEPS,
                                max_candidates_per_round=PRUNE_MAX_CANDIDATES_PER_ROUND,
                                prune_mode=PRUNE_MODE,
                            )

                        k_after = template_num_params(pruned_template)

                        if pruned_metrics is not None and k_after <= target_k_coord:
                            # Refit once cleanly after pruning.
                            pruned_metrics2, pruned_p2, pruned_expr, _, _ = \
                                fit_template_to_data(
                                    pruned_template,
                                    s_data,
                                    y_data,
                                    sigma=sigma,
                                    timeout_seconds=PRUNE_CANDIDATE_TIMEOUT_SECONDS,
                                )

                            print(
                                f"[Track {track_idx} {coord_name}] pruning succeeded: "
                                f"k={k_before}->{k_after}, "
                                f"chi2_red={pruned_metrics2.get('chi2_red', np.nan):.3e}"
                            )

                            # Use pruned result only if it beats the best template that
                            # was constrained out of the box.
                            if better(pruned_metrics2, best_metrics):
                                best_metrics = pruned_metrics2
                                best_expr = pruned_expr
                                best_template_index = best_prune_template_index
                                best_template_obj = pruned_template
                                best_p_opt = pruned_p2

                        else:
                            print(
                                f"[Track {track_idx} {coord_name}] pruning failed to reach "
                                f"constraint: k={k_after}, target={target_k_coord}; "
                                f"using best naturally constrained template."
                            )
                            print(f"[Track {track_idx} {coord_name}] best template = {best_expr}")

            # 3) Otherwise, run PySR to discover new form
            if not RunPySR:
                if best_expr is None:
                    raise RuntimeError(
                        f"No templates available (or all failed) for Track {track_idx} {coord_name}, "
                        f"and RunPySR=False. Can't produce a fit."
                    )
                if is_weighted():
                    print(
                        f"[Track {track_idx} {coord_name}] RunPySR=False; using best available template [{best_template_index}] "
                        f"(chi2_red={best_metrics['chi2_red']:.3f}, target 1)."
                    )
                else:
                    print(
                        f"[Track {track_idx} {coord_name}] best_R2={best_metrics['R2']:.3f} < {R2_THRESHOLD} "
                        f"but RunPySR=False; using best available template [{best_template_index}]. "
                    )
                return accept_template(
                    best_expr,
                    best_metrics,
                    best_template_index,
                    best_template_obj,
                    best_p_opt,
                )

            if is_weighted():
                best_chi2_red = float("inf") if best_metrics is None else best_metrics["chi2_red"]
                print(
                    f"[Track {track_idx} {coord_name}] best chi2_red={best_chi2_red:.3f}"
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
                    sigma=sigma,
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

            metrics_new, p_opt_new, expr_fitted_new, _, _ = fit_template_to_data(
                template,
                s_data,
                y_data,
                sigma=sigma,
            )

            # If templates existed and one of them was better, keep it instead of PySR
            if best_metrics is not None:
                if is_weighted():
                    if best_metrics["chi2_red"] <= metrics_new["chi2_red"]:
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
            return accept_template(
                expr_fitted_new,
                metrics_new,
                new_template_index,
                template,
                p_opt_new,
            )

        if has_rphi_covariance:
            # Important:
            # do NOT create independent sigma_x/sigma_y weights.
            # SR template selection remains coordinate-wise R^2.
            sig_x = None
            sig_y = None
            sig_z = None
            n_hits = n_hits_cov

        else:
            # Exact legacy behavior.
            sig_x = sigx_from_loader
            sig_y = sigy_from_loader
            sig_z = sigz_from_loader

            if sig_x is None:
                sig_x, sig_y, sig_z, n_hits = \
                    load_sigmas_for_event(
                        track_folder,
                        event_num
                    )
            else:
                n_hits = len(s_all)

        if has_rphi_covariance:
            print(
                f"[info] event {f_all}: "
                f"rphi covariance → SR templates selected by R^2"
            )
        elif sig_x is None:
            print(
                f"[info] event {f_all}: "
                f"no sigma columns → unweighted fits"
            )
        else:
            print(
                f"[info] event {f_all}: "
                f"legacy Cartesian sigma columns → weighted fits"
            )

        # sanity: lengths must match loaded arrays (same check you had before)
        if len(s_all) != n_hits:
            if has_rphi_covariance:
                raise RuntimeError(
                    f"[event {f_all}] covariance data length mismatch: "
                    f"loaded track has {len(s_all)} hits but CSV has {n_hits}"
                )

            print(
                f"[warning] mismatch in hit counts for event {f_all}: "
                f"s_len={len(s_all)}, csv_hits={n_hits}. "
                f"Using unweighted fit."
            )

            expr_x, m_x = fit_dimension(
                s_all, x_all, x_templates,
                x_eqns, families_x, "x"
            )
            expr_y, m_y = fit_dimension(
                s_all, y_all, y_templates,
                y_eqns, families_y, "y"
            )
            expr_z, m_z = fit_dimension(
                s_all, z_all, z_templates,
                z_eqns, families_z, "z"
            )
        else:
            expr_x, m_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x", sigma=sig_x)
            expr_y, m_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y", sigma=sig_y)
            expr_z, m_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z", sigma=sig_z)

        sm_metrics = None

        sm_x = sm_y = sm_z = None
        chi2nu_sm = None
        chi2nu_helix = None
        
        kx_selected = template_num_params(x_templates[families_x[-1]])
        ky_selected = template_num_params(y_templates[families_y[-1]])
        kz_selected = template_num_params(z_templates[families_z[-1]])
        k_sr_selected = kx_selected + ky_selected + kz_selected
        n_obs_per_hit = 2 if has_rphi_covariance else 3
        n_obs_sr = n_obs_per_hit * len(s_all)

        dof_sr_raw_selected = (
            n_obs_sr - k_sr_selected
        )

        print(
            f"[SR DOF] n_hits={len(s_all)}, "
            f"n_obs={n_obs_sr}, "
            f"has_rphi_covariance={has_rphi_covariance}, "
            f"kx={kx_selected}, "
            f"ky={ky_selected}, "
            f"kz={kz_selected}, "
            f"k_sr={k_sr_selected}, "
            f"N-k={dof_sr_raw_selected}"
        )
        
        S_SYM_CHI2 = sp.Symbol("s")

        def eval_selected_expr(expr):
            if len(expr.free_symbols) == 0:
                return np.full(
                    len(s_all),
                    float(expr),
                    dtype=float
                )

            if S_SYM_CHI2 in expr.free_symbols:
                f = sp.lambdify(
                    S_SYM_CHI2,
                    expr,
                    "numpy"
                )
            elif len(expr.free_symbols) == 1:
                only = next(iter(expr.free_symbols))
                f = sp.lambdify(
                    only,
                    expr,
                    "numpy"
                )
            else:
                raise ValueError(
                    f"Ambiguous symbols in {expr}"
                )

            out = np.asarray(
                f(s_all),
                dtype=float
            )

            if out.shape == ():
                out = np.full(
                    len(s_all),
                    float(out)
                )

            return out
        
        if has_rphi_covariance:
            x_sr_hits = eval_selected_expr(expr_x)
            y_sr_hits = eval_selected_expr(expr_y)
            z_sr_hits = eval_selected_expr(expr_z)

            chi2_sr = covariance_chi2(
                x_all, y_all, z_all,
                x_sr_hits,
                y_sr_hits,
                z_sr_hits,
                phi_detector,
                sigma_rphi,
                sigma_z_cov,
            )

        else:
            # Exact legacy behavior.
            chi2_sr = (
                m_x["chi2"]
                + m_y["chi2"]
                + m_z["chi2"]
            )

        if REQUIRE_SR_CONSTRAINED:
            dof_sr = max(
                n_obs_sr - k_sr_selected,
                1
            )
        else:
            dof_sr = max(
                n_obs_sr - 1,
                1
            )

        chi2nu_sr = chi2_sr / dof_sr
        if has_rphi_covariance:

            helix_fit = fit_helix_covariance(
                x_all,
                y_all,
                z_all,
                phi_detector,
                sigma_rphi,
                sigma_z_cov,
            )

            x_sm = helix_fit["x"]
            y_sm = helix_fit["y"]
            z_sm = helix_fit["z"]

            sm_x = x_sm
            sm_y = y_sm
            sm_z = z_sm

            chi2nu_sm = helix_fit["chi2_red"]
            chi2nu_helix = chi2nu_sm

            sm_metrics = helix_fit
        else:
            # --- 1) fit circle in x–y to get helix center ---
            xc0, yc0 = np.mean(x_all), np.mean(y_all)
            R0 = np.median(np.sqrt((x_all - xc0)**2 + (y_all - yc0)**2))

            def circle_resid(p):
                xc, yc, R = p
                return np.sqrt((x_all - xc)**2 + (y_all - yc)**2) - R

            res_xy = least_squares(circle_resid, [xc0, yc0, R0], loss="linear", f_scale=1.0)
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
            dof = max(len(res_xyz) - 5*USE_K, 1)  # 5 params: xc,yc,R,z0,a
            chi2_red = chi2 / dof

            sm_metrics = {
                "chi2": chi2,
                "chi2_red": chi2_red,
                "xc": xc_fit, "yc": yc_fit, "R": R_fit,
                "z0": z0, "a": a
            }
            
            chi2nu_sm = sm_metrics["chi2_red"]
            
            chi2nu_helix = chi2nu_sm
            
        log_chi2_ratio = np.log((chi2nu_sr + 1e-300) / (chi2nu_helix + 1e-300))

        chi2_sr_track_all.append(chi2nu_sr)
        chi2_helix_track_all.append(chi2nu_helix)
        log_chi2_ratio_all.append(log_chi2_ratio)

        print(f"[Ratio] chi2nu_SR={chi2nu_sr:.3e}, chi2nu_helix={chi2nu_helix:.3e}, "
              f"log(SR/helix)={log_chi2_ratio:.3f}")
        
        

        if has_rphi_covariance:
            print(
                f"[SM Fit] xc={helix_fit['xc']:.2f}, "
                f"yc={helix_fit['yc']:.2f}, "
                f"R={helix_fit['R']:.2f}"
            )
            print(
                f"[SM Fit] covariance chi2_red="
                f"{helix_fit['chi2_red']:.3e}"
            )
        else:
            print(
                f"[SM Fit] xc={xc_fit:.2f}, "
                f"yc={yc_fit:.2f}, "
                f"R={R_fit:.2f}"
            )
            print(
                f"[SM Fit] chi2_red={chi2_red:.3e}, "
                f"res_std={np.std(res_xyz):.3e}"
            )

        # Store R^2 values for summaries
        r2_x_all.append(m_x["R2"])
        r2_y_all.append(m_y["R2"])
        r2_z_all.append(m_z["R2"])
        
        chi2_x_all.append(m_x["chi2_red"])
        chi2_y_all.append(m_y["chi2_red"])
        chi2_z_all.append(m_z["chi2_red"])

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
            s_plot = np.linspace(s_all[0], s_all[-1], 200000)
        else:
            s_plot = np.array([])
        x_pred = fx(s_plot)
        y_pred = fy(s_plot)
        z_pred = fz(s_plot)

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        fig.suptitle(
            fr"$\chi^2_{{\nu,\mathrm{{SR}}}}={chi2nu_sr:.3e}$, "
            fr"$\chi^2_{{\nu,\mathrm{{helix}}}}={chi2nu_helix:.3e}$, "
            fr"$\log(\chi^2_{{\nu,\mathrm{{SR}}}}/\chi^2_{{\nu,\mathrm{{helix}}}})={log_chi2_ratio:.3f}$",
            fontsize=11
        )
        MAX_EQ_CHARS = 80  # tune
        # x(s)
        if has_rphi_covariance:
            label_x = (
                fr"Fit: $R^2={m_x['R2']:.3f}$, "
                fr"$\mathrm{{MSE}}={m_x['MSE']:.3e}$, "
                fr"$\mathrm{{MAE}}={m_x['MAE']:.3e}$"
                + (
                    ("\n" + eq_x)
                    if (eq_x and len(eq_x) <= MAX_EQ_CHARS)
                    else ""
                )
            )
        else:
            label_x = (
                fr"Fit: $R^2={m_x['R2']:.3f}$, "
                fr"$\mathrm{{MSE}}={m_x['MSE']:.3e}$, "
                fr"$\mathrm{{MAE}}={m_x['MAE']:.3e}$, "
                fr"$\chi^2={m_x['chi2']:.3e}$, "
                fr"$\chi^2_\nu={m_x['chi2_red']:.3e}$"
                + (
                    ("\n" + eq_x)
                    if (eq_x and len(eq_x) <= MAX_EQ_CHARS)
                    else ""
                )
            )
        plot_points(axes[0], s_all, x_all, sig_x, label="Data")
        axes[0].plot(s_plot, x_pred, linewidth=2, label=label_x)
        if (sm_x is not None):
            axes[0].plot(
                s_all, x_sm,
                linewidth=2,
                linestyle="--",
                label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$"
            )
        axes[0].set_ylabel("$x$")
        axes[0].legend(fontsize=8)

        # y(s)
        if has_rphi_covariance:
            label_y = (
                fr"Fit: $R^2={m_y['R2']:.3f}$, "
                fr"$\mathrm{{MSE}}={m_y['MSE']:.3e}$, "
                fr"$\mathrm{{MAE}}={m_y['MAE']:.3e}$"
                + (
                    ("\n" + eq_y)
                    if (eq_y and len(eq_y) <= MAX_EQ_CHARS)
                    else ""
                )
            )
        else:
            label_y = (fr"Fit: $R^2={m_y['R2']:.3f}$, "
               fr"$\mathrm{{MSE}}={m_y['MSE']:.3e}$, "
               fr"$\mathrm{{MAE}}={m_y['MAE']:.3e}$, "
               fr"$\chi^2={m_y['chi2']:.3e}$, "
               fr"$\chi^2_\nu={m_y['chi2_red']:.3e}$"
               + (("\n" + eq_y) if (eq_y and len(eq_y) <= MAX_EQ_CHARS) else ""))
        plot_points(axes[1], s_all, y_all, sig_y, label="Data")
        axes[1].plot(s_plot, y_pred, linewidth=2, label=label_y)
        if (sm_y is not None):
            axes[1].plot(s_all, y_sm, linewidth=2, linestyle="--", label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$")
        axes[1].set_ylabel("$y$")
        axes[1].legend(fontsize=8)

        # z(s)
        if has_rphi_covariance:
            label_z = (
                fr"Fit: $R^2={m_z['R2']:.3f}$, "
                fr"$\mathrm{{MSE}}={m_z['MSE']:.3e}$, "
                fr"$\mathrm{{MAE}}={m_z['MAE']:.3e}$"
                + (
                    ("\n" + eq_z)
                    if (eq_z and len(eq_z) <= MAX_EQ_CHARS)
                    else ""
                )
            )
        else:
            label_z = (fr"Fit: $R^2={m_z['R2']:.3f}$, "
               fr"$\mathrm{{MSE}}={m_z['MSE']:.3e}$, "
               fr"$\mathrm{{MAE}}={m_z['MAE']:.3e}$, "
               fr"$\chi^2={m_z['chi2']:.3e}$, "
               fr"$\chi^2_\nu={m_z['chi2_red']:.3e}$"
               + (("\n" + eq_z) if (eq_z and len(eq_z) <= MAX_EQ_CHARS) else ""))
        plot_points(axes[2], s_all, z_all, sig_z, label="Data")
        axes[2].plot(s_plot, z_pred, linewidth=2, label=label_z)
        if (sm_z is not None):
            axes[2].plot(s_all, z_sm, linewidth=2, linestyle="--", label=fr"SM helix: $\chi^2_\nu={chi2nu_sm:.3e}$")
        axes[2].set_ylabel("$z$")
        axes[2].set_xlabel(r"reconstructed helix phase $\phi$" if independent_param == "phi" else "normalized arc length")
        axes[2].legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
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
            "chi2nu_sr": chi2nu_sr,
            "chi2nu_helix": chi2nu_helix,
            "log_chi2_ratio": log_chi2_ratio,
        })

    if FIT_MODE == "SPLIT_HELIX":
    
        ratio_txt = (
            f"../data_files/"
            f"split_helix_ratio_values_{outfile_str}.csv"
        )

        np.savetxt(
            ratio_txt,
            np.column_stack([
                np.asarray(log_split_ratio_all),
                np.asarray(chi2_split_helix_all),
                np.asarray(chi2_full_helix_all),
            ]),
            delimiter=",",
            header=(
                "log_chi2_split_over_helix,"
                "chi2nu_split,"
                "chi2nu_helix"
            ),
            comments="",
        )

        print(f"Saved split-helix ratio values to {ratio_txt}")

        ratio_hist_png = (
            f"../pngs/split_helix_ratio_hist_{outfile_str}.png"
        )

        vals = np.asarray(log_split_ratio_all, dtype=float)
        vals = vals[np.isfinite(vals)]

        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=min(10, max(3, len(vals))))
        plt.axvline(0.0, linestyle="--", linewidth=1)

        plt.xlabel(
            r"$\log(\chi^2_{\nu,\mathrm{split}}/"
            r"\chi^2_{\nu,\mathrm{helix}})$"
        )
        plt.ylabel("Tracks")
        plt.title("Single helix vs split helix")
        plt.tight_layout()
        plt.savefig(ratio_hist_png, dpi=5*96)
        plt.close()

        rows = []

        for info in split_infos:
            rows.append(f"""
            <div style="margin-bottom:30px">
              <h3>
                Track {info['track_idx'] + 1}
                — {info['event_id']}
              </h3>

              <p>
                split = {info['split_idx']}<br>
                single-helix chi2nu =
                {info['chi2nu_full']:.4e}<br>
                split-helix chi2nu =
                {info['chi2nu_split']:.4e}<br>
                log(split/single) =
                {info['log_ratio']:.4f}
              </p>

              <img
                src="{info['png']}"
                style="width:100%;max-width:900px"
              >
            </div>
            """)

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Split Helix - {dataset_label}</title>
        </head>
        <body style="
            font-family:serif;
            max-width:1000px;
            margin:30px auto;
        ">

        <h1>Split-Helix Study</h1>
        <h2>{dataset_label}</h2>

        <p>
        Each finder candidate is compared with a single
        five-parameter helix and the best pair of helices
        obtained by scanning every allowed split point.
        </p>

        <img
            src="{ratio_hist_png}"
            style="width:100%;max-width:700px"
        >

        {''.join(rows)}

        </body>
        </html>
        """

        with open(out_html, "w") as f:
            f.write(html)

        print(f"Wrote {out_html}")

        if OPEN_HTML:
            system(f"open {out_html}")

        raise SystemExit
    # Build explicit LaTeX equations per 3D family
    def latex_equations_for_3d_families(triple_id_map,
                                        families_x, families_y, families_z,
                                        x_eqns, y_eqns, z_eqns):
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
    
    if PROFILE_TEMPLATE_FITS and fit_time_records:
        timing_df = pd.DataFrame(fit_time_records)

        timing_csv = f"../data_files/template_fit_times_{outfile_str}.csv"
        timing_df.to_csv(timing_csv, index=False)

        worst = timing_df.loc[timing_df["seconds"].idxmax()]
        print("\n=== Template refit timing summary ===")
        print(f"Saved timing data to {timing_csv}")
        print(
            "Worst single component/template refit: "
            f"track={int(worst['track_idx'])}, "
            f"coord={worst['coord']}, "
            f"template={int(worst['template_idx'])}, "
            f"time={worst['seconds']:.3f} s, "
            f"chi2_red={worst['chi2_red']:.3e}"
        )

        print("\nSlowest 10 refits:")
        print(timing_df.sort_values("seconds", ascending=False).head(10).to_string(index=False))

    weighted_flag = (SIG_X_LIST[0] is not None)
    latex_3d_table, triple_id_map = summarize_3d_families(
        families_x, families_y, families_z,
        r2_x_all, r2_y_all, r2_z_all,
        chi2_x_all, chi2_y_all, chi2_z_all,
        weighted=weighted_flag
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

    with open(TEMPLATE_PATH, "wb") as f:
        pickle.dump(save_obj, f)

    ratio_hist_png = f"../pngs/chi2_ratio_hist_{outfile_str}.png"

    vals = np.asarray(log_chi2_ratio_all, dtype=float)
    vals = vals[np.isfinite(vals)]

    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=min(10, max(3, len(vals))))
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel(r"$\log(\chi^2_{\nu,\mathrm{SR}} / \chi^2_{\nu,\mathrm{helix}})$")
    plt.ylabel("Tracks")
    plt.title("SR vs helix goodness-of-fit ratio")
    plt.tight_layout()
    plt.savefig(ratio_hist_png, dpi=5*96)
    plt.close()

    system(f"cp {ratio_hist_png} /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/pngs/")
    
    ratio_txt = f"../data_files/chi2_ratio_values_{outfile_str}.csv"

    np.savetxt(ratio_txt,
        np.column_stack([
            np.asarray(log_chi2_ratio_all),
            np.asarray(chi2_sr_track_all),
            np.asarray(chi2_helix_track_all),
        ]),
        delimiter=",",
        header="log_chi2_sr_over_helix,chi2nu_sr,chi2nu_helix",
        comments="")

    print(f"Saved SR/helix chi2 ratio values to {ratio_txt}")

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
    
    .eqScroll {
      margin-top: 8px;
      padding: 10px 12px;
      background: #f7f7f7;
      border: 1px solid #eee;
      border-radius: 12px;
      overflow-x: auto;
      overflow-y: hidden;
      white-space: nowrap;
    }
    .eqScroll mjx-container {
      display: inline-block; /* keeps MathJax from forcing weird wraps */
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
  <p class="subtle">Use ← / → or the buttons to scroll through tracks in track order.</p>

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
      const eqBlock = `
          <div class="eqScroll">${t.eqx}</div>
          <div class="eqScroll">${t.eqy}</div>
          <div class="eqScroll">${t.eqz}</div>
        `;

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
      <br>
      <div class="card" style="margin-bottom:18px;">
          <h2>χ² Ratio Distribution</h2>
          <p class="subtle">
            Negative values mean SR fits better than helix; positive values mean helix fits better.
          </p>
          <img src="__RATIO_HIST_PNG__" alt="chi2 ratio histogram"/>
        </div>
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
          <div class="eqScroll" style="margin-top:6px;">${e.eq}</div>
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
    eq_map = defaultdict(list)
    slide_index = 0

    # Preserve original track order: item 1 = track 1, etc.
    for t in sorted(track_infos, key=lambda d: d["track_idx"]):
        mx, my, mz = t["mx"], t["my"], t["mz"]

        metrics_line = (
            r"\begin{align*}"
            rf"&R^2:\, x={mx['R2']:.3f},\;"
            rf"y={my['R2']:.3f},\;"
            rf"z={mz['R2']:.3f}"
            r"\end{align*}"
        )

        track_num = t["track_idx"] + 1

        sig_x = template_signature(x_templates[t["Fx"]])
        sig_y = template_signature(y_templates[t["Fy"]])
        sig_z = template_signature(z_templates[t["Fz"]])

        eq_map[sig_x].append({"slide": slide_index, "label": f"{track_num}-x", "coord": "x", "rep_fam": t["Fx"]})
        eq_map[sig_y].append({"slide": slide_index, "label": f"{track_num}-y", "coord": "y", "rep_fam": t["Fy"]})
        eq_map[sig_z].append({"slide": slide_index, "label": f"{track_num}-z", "coord": "z", "rep_fam": t["Fz"]})

        slides_payload.append({
            "fid": t["fid"],
            "Fx": t["Fx"], "Fy": t["Fy"], "Fz": t["Fz"],
            "track_idx": t["track_idx"],
            "event_id": t["event_id"],
            "png": t["png"].replace("../",""),
            "img_src": t["png"].replace("../",""),
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
        for coord, fam_idx in [("x", sp_item["Fx"]), ("y", sp_item["Fy"]), ("z", sp_item["Fz"])]:
            if coord == "x":
                sig = template_signature(x_templates[fam_idx])
            elif coord == "y":
                sig = template_signature(y_templates[fam_idx])
            else:
                sig = template_signature(z_templates[fam_idx])

            if sig in seen:
                continue
            seen.add(sig)

            uses = eq_map.get(sig, [])
            rep = uses[0] if uses else {"coord": coord, "rep_fam": fam_idx}
            rep_coord = rep["coord"]
            rep_idx = rep["rep_fam"]

            if rep_coord == "x":
                eq_text = r"$x(s) = " + template_to_latex_with_s(x_templates[rep_idx]) + r"$"
            elif rep_coord == "y":
                eq_text = r"$y(s) = " + template_to_latex_with_s(y_templates[rep_idx]) + r"$"
            else:
                eq_text = r"$z(s) = " + template_to_latex_with_s(z_templates[rep_idx]) + r"$"

            eq_slides.append({
                "key": sig,
                "eq": eq_text,
                "uses": uses,
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
    html_text = html_text.replace("__RATIO_HIST_PNG__", html_escape(ratio_hist_png[3:]))

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"[HTML] Wrote report to: {out_html}")
    system(f"mv {out_html} /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/")
    if OPEN_HTML:
        system(f"open /Users/edwardfinkelstein/AIFeynmanExpressionTrees/Whiteson/{out_html}")
