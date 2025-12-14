from load_tracks import load_many_tracks
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from os import system
from scipy.optimize import least_squares
from collections import defaultdict
import pickle
import os

round_floats = lambda expr, ndigits: expr.xreplace({f: sp.Float(round(float(f), ndigits)) for f in expr.atoms(sp.Float)})
OPEN = False

def fit_until_both_conditions(s, y, model_kwargs, R2_THRESHOLD, min_iters=100, step=50, max_iters=np.inf):
    """
    Continue training PySR until BOTH:
      - R^2 >= R2_THRESHOLD
      - total iterations >= min_iters
    Training stops if max_iters is reached.
    """

    model = PySRRegressor(**model_kwargs)
    total_iters = 0
    best_R2 = 0

    while True:
        # Run more iterations *without resetting the model*
        model.niterations = total_iters + step
        model.fit(s, y)
        total_iters += step

        # Compute R^2
        R2 = model.score(s, y)
        if R2 > best_R2:
            print(f"New best R2 = {R2:.3f}")
        # Check stopping condition
        if total_iters >= min_iters and R2 >= R2_THRESHOLD:
            break

        # Safety: prevent infinite loops
        if total_iters >= max_iters:
            print(f"Stopping at max_iters={max_iters} with R2={R2:.3f}")
            break

    return model, R2, total_iters

def sympy_to_latex_with_s(expr):
    """Return a LaTeX string with rounded floats and x0→s replacement."""
    # Round floats first
    expr = round_floats(expr, 3)

    # Convert to LaTeX
    expr_latex = sp.latex(expr)

    # Replace PySR variable name with s
    expr_latex = expr_latex.replace("x_{0}", "s").replace("x0", "s")

    return expr_latex

def template_to_latex_with_s(template):
    """Convert a parametric template to LaTeX, replacing x0→s."""
    expr = template["expr"]  # parametric sympy expression
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

def fit_template_to_data(template, s_data, y_data):
    """
    Given a template dict and data (s_data, y_data),
    optimize the parameters and return (best_R2, best_params, expr_fitted).
    """
    expr_param = template["expr"]
    s_sym      = template["s_sym"]
    param_syms = template["param_syms"]
    p0         = template["init_params"]

    # Build f(s, *a) using lambdify
    f = sp.lambdify((s_sym, *param_syms), expr_param, "numpy")

    def residuals(p):
        y_pred = f(s_data, *p)
        return y_pred - y_data

    # Nonlinear least squares
    res = least_squares(residuals, p0)

    p_opt = res.x
    y_pred = f(s_data, *p_opt)

    # Compute R^2
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Build numeric expression with fitted params
    subs_dict = {sym: val for sym, val in zip(param_syms, p_opt)}
    expr_fitted = expr_param.subs(subs_dict)

    return R2, p_opt, expr_fitted

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
    TEMPLATE_PATH = "track_templates.pkl"
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

    x_eqns, y_eqns, z_eqns = [], [], []   # store final numeric expressions used
    families_x, families_y, families_z = [], [], []  # indices of template used
    r2_x_all, r2_y_all, r2_z_all = [], [], [] # R^2 values for each track and coordinate

    track_folder = "../tracks_for_ed"
    S, X, Y, Z, F = load_many_tracks(track_folder, max_tracks=5)

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
    R2_THRESHOLD = 0.95

    for track_idx, (s_all, x_all, y_all, z_all, f_all) in enumerate(zip(S, X, Y, Z, F)):
        s_all_reshaped = s_all.reshape(-1, 1)

        # --- Helper inner function to handle one coordinate --- #
        def fit_dimension(
            s_data, y_data, templates, eqn_list, families_list, coord_name
        ):
            print(f"\n--- Fitting Track {track_idx}, coordinate {coord_name} ---")
            
            best_R2 = -np.inf
            best_expr = None
            best_template_index = None

            # 1) Try existing templates
            if templates:
                for idx, template in enumerate(templates):
                    R2, p_opt, expr_fitted = fit_template_to_data(template, s_data, y_data)
                    if R2 > best_R2:
                        best_R2 = R2
                        best_expr = expr_fitted
                        best_template_index = idx

            # 2) If good enough, use best template
            if templates and best_R2 >= R2_THRESHOLD:
                print(f"[Track {track_idx} {coord_name}] Used existing template {best_template_index} with R^2={best_R2:.3f}")
                eqn_list.append(best_expr)
                families_list.append(best_template_index)
                return best_expr, best_R2

            # 3) Otherwise, run PySR to discover new form
            model, R2_final, iters = fit_until_both_conditions(
                s_data.reshape(-1, 1),
                y_data,
                model_kwargs,
                R2_THRESHOLD,
                min_iters=100,
                step=50,
                max_iters=np.inf,
            )

            expr = model.sympy()


            # Make parametric template from this expression
            template = make_parametric_template(expr, s_name="x0")
            templates.append(template)
            new_template_index = len(templates) - 1

            # For consistency, also treat this as "fitted" expression (with its initial parameters)
            R2_new, _, expr_fitted_new = fit_template_to_data(template, s_data, y_data)

            eqn_list.append(expr_fitted_new)
            families_list.append(new_template_index)

            return expr_fitted_new, R2_new

        # --- Fit x, y, z this track --- #
        expr_x, r2_x = fit_dimension(s_all, x_all, x_templates, x_eqns, families_x, "x")
        expr_y, r2_y = fit_dimension(s_all, y_all, y_templates, y_eqns, families_y, "y")
        expr_z, r2_z = fit_dimension(s_all, z_all, z_templates, z_eqns, families_z, "z")
        
        # Store R^2 values for summaries
        r2_x_all.append(r2_x)
        r2_y_all.append(r2_y)
        r2_z_all.append(r2_z)

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
        axes[0].scatter(s_all, x_all, s=10, alpha=0.6, label="Data")
        axes[0].plot(s_plot, x_pred, linewidth=2,
                     label=fr"Fit: $R^2={r2_x:.3f}$" + "\n" + eq_x)
        axes[0].set_ylabel("$x$")
        axes[0].legend(fontsize=8)

        # y(s)
        axes[1].scatter(s_all, y_all, s=10, alpha=0.6, label="Data")
        axes[1].plot(s_plot, y_pred, linewidth=2,
                     label=fr"Fit: $R^2={r2_y:.3f}$" + "\n" + eq_y)
        axes[1].set_ylabel("$y$")
        axes[1].legend(fontsize=8)

        # z(s)
        axes[2].scatter(s_all, z_all, s=10, alpha=0.6, label="Data")
        axes[2].plot(s_plot, z_pred, linewidth=2,
                     label=fr"Fit: $R^2={r2_z:.3f}$" + "\n" + eq_z)
        axes[2].set_ylabel("$z$")
        axes[2].set_xlabel("$s$")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(f"../pdfs/SR_track_{f_all}.pdf")
        system(f"open ../pdfs/SR_track_{f_all}.pdf") if OPEN else None

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

