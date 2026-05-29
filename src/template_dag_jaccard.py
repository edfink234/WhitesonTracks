import pickle
import numpy as np
import pandas as pd
import sympy as sp


CONFIG = {
    "template_path": "track_templates.pkl",

    # choose one: "x", "y", or "z"
    "coord": "x",

    # template indices to compare
    "indices": [12,14,16,17,49],

    # output CSV, or None
    "out_csv": "../data_files/5_mode_100_dag_jaccard.csv",

    # for structural similarity, usually keep both False
    "keep_const_values": False,
    "keep_param_names": False,
}


def sech_stable(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = np.abs(x) < 700
    out[mask] = 1.0 / np.cosh(x[mask])
    return out


class SympyDagEvaluator:
    """
    Build a DAG from a SymPy Expr by hash-consing subexpressions,
    then evaluate bottom-up on NumPy arrays/scalars.

    Node format:
        (op, data, children)

    where:
        op       : string like "const", "symbol", "add", "mul", ...
        data     : constant value or symbol name or None
        children : tuple of child node ids
    """

    def __init__(self, expr):
        self.expr = expr
        self.nodes = []
        self.root = -1

        # SymPy expr -> node id
        self._intern_expr = {}

        # (op, data, children) -> node id
        self._intern_node = {}

        self.root = self._build(expr)

    def _intern(self, op, data, children):
        key = (op, data, children)
        if key in self._intern_node:
            return self._intern_node[key]
        idx = len(self.nodes)
        self.nodes.append((op, data, children))
        self._intern_node[key] = idx
        return idx

    def _build(self, expr):
        if expr in self._intern_expr:
            return self._intern_expr[expr]

        if expr.is_Symbol:
            node_id = self._intern("symbol", str(expr), ())

        elif expr.is_Integer:
            node_id = self._intern("const", float(int(expr)), ())

        elif expr.is_Rational:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Float:
            node_id = self._intern("const", float(expr), ())

        elif expr.is_Number:
            node_id = self._intern("const", float(expr.evalf()), ())

        else:
            args = tuple(self._build(a) for a in expr.args)

            if expr.func is sp.Add:
                node_id = self._intern("add", None, args)

            elif expr.func is sp.Mul:
                node_id = self._intern("mul", None, args)

            elif expr.func is sp.Pow:
                node_id = self._intern("pow", None, args)

            elif expr.func is sp.sin:
                node_id = self._intern("sin", None, args)

            elif expr.func is sp.cos:
                node_id = self._intern("cos", None, args)

            elif expr.func is sp.exp:
                node_id = self._intern("exp", None, args)

            elif expr.func is sp.log:
                node_id = self._intern("log", None, args)

            elif expr.func is sp.sqrt:
                node_id = self._intern("sqrt", None, args)

            elif expr.func is sp.tanh:
                node_id = self._intern("tanh", None, args)

            elif expr.func.__name__ == "sech":
                node_id = self._intern("sech", None, args)

            elif expr.func is sp.asin:
                node_id = self._intern("asin", None, args)

            elif expr.func is sp.acos:
                node_id = self._intern("acos", None, args)

            elif expr.func is sp.Abs:
                node_id = self._intern("abs", None, args)

            else:
                raise NotImplementedError(
                    "Unsupported SymPy node: func=%r expr=%r" % (expr.func, expr)
                )

        self._intern_expr[expr] = node_id
        return node_id

    def evaluate(self, env, shape=None, dtype=np.float64):
        """
        env maps symbol names to numpy arrays/scalars, e.g.
            {"r": r_vals, "theta": theta_vals}

        shape is only needed if expression is constant and you want an array result.
        """
        values = [None] * len(self.nodes)

        for i, node in enumerate(self.nodes):
            op, data, children = node

            if op == "symbol":
                if data not in env:
                    raise KeyError("Missing value for symbol '%s'" % data)
                values[i] = env[data]

            elif op == "const":
                c = np.array(data, dtype=dtype)
                if shape is None:
                    values[i] = c
                else:
                    values[i] = np.full(shape, c, dtype=dtype)

            else:
                ch = [values[j] for j in children]

                if op == "add":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out + x
                    values[i] = out

                elif op == "mul":
                    out = ch[0]
                    for x in ch[1:]:
                        out = out * x
                    values[i] = out

                elif op == "pow":
                    base, expo = ch
                    values[i] = np.power(base, expo)

                elif op == "sin":
                    values[i] = np.sin(ch[0])

                elif op == "cos":
                    values[i] = np.cos(ch[0])

                elif op == "exp":
                    values[i] = np.exp(ch[0])

                elif op == "log":
                    values[i] = np.log(ch[0])

                elif op == "sqrt":
                    values[i] = np.sqrt(ch[0])

                elif op == "tanh":
                    values[i] = np.tanh(ch[0])

                elif op == "sech":
                    values[i] = sech_stable(ch[0])

                elif op == "asin":
                    values[i] = np.arcsin(ch[0])

                elif op == "acos":
                    values[i] = np.arccos(ch[0])

                elif op == "abs":
                    values[i] = np.abs(ch[0])

                else:
                    raise RuntimeError("Unknown op '%s'" % op)

        return values[self.root]


def canonical_node_signatures(dag, *, ignore_const_values=True, ignore_param_names=True):
    sigs = [None] * len(dag.nodes)

    def rec(i):
        if sigs[i] is not None:
            return sigs[i]

        op, data, children = dag.nodes[i]
        child_sigs = tuple(rec(j) for j in children)

        if op == "const" and ignore_const_values:
            data_use = "CONST"
        elif op == "symbol" and ignore_param_names and str(data).startswith("a"):
            data_use = "PARAM"
        else:
            data_use = data

        sigs[i] = (op, data_use, child_sigs)
        return sigs[i]

    for i in range(len(dag.nodes)):
        rec(i)

    return set(sigs)

def canonical_node_signature_map(dag, *, ignore_const_values=True, ignore_param_names=True):
    sigs = [None] * len(dag.nodes)

    def rec(i):
        if sigs[i] is not None:
            return sigs[i]

        op, data, children = dag.nodes[i]
        child_sigs = tuple(rec(j) for j in children)

        if op == "const" and ignore_const_values:
            data_use = "CONST"
        elif op == "symbol" and ignore_param_names and str(data).startswith("a"):
            data_use = "PARAM"
        else:
            data_use = data

        sigs[i] = (op, data_use, child_sigs)
        return sigs[i]

    for i in range(len(dag.nodes)):
        rec(i)

    return {sig: i for i, sig in enumerate(sigs)}, sigs


def dag_subexpr_size(sig):
    op, data, children = sig
    return 1 + sum(dag_subexpr_size(c) for c in children)


def dag_sig_to_sympy(sig):
    op, data, children = sig
    args = [dag_sig_to_sympy(c) for c in children]

    if op == "symbol":
        return sp.Symbol(str(data))
    if op == "const":
        return sp.Symbol("C") if data == "CONST" else sp.Float(data)

    if op == "add":
        return sp.Add(*args)
    if op == "mul":
        return sp.Mul(*args)
    if op == "pow":
        return sp.Pow(*args)
    if op == "sin":
        return sp.sin(args[0])
    if op == "cos":
        return sp.cos(args[0])
    if op == "exp":
        return sp.exp(args[0])
    if op == "log":
        return sp.log(args[0])
    if op == "sqrt":
        return sp.sqrt(args[0])
    if op == "tanh":
        return sp.tanh(args[0])
    if op == "sech":
        return sp.sech(args[0])
    if op == "asin":
        return sp.asin(args[0])
    if op == "acos":
        return sp.acos(args[0])
    if op == "abs":
        return sp.Abs(args[0])

    raise ValueError(f"Unknown op: {op}")


def dag_jaccard(expr1, expr2, **kwargs):
    A = canonical_node_signatures(SympyDagEvaluator(expr1), **kwargs)
    B = canonical_node_signatures(SympyDagEvaluator(expr2), **kwargs)
    return 1.0 if not A and not B else len(A & B) / len(A | B)


def load_templates(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    return {
        "x": obj["x_templates"],
        "y": obj["y_templates"],
        "z": obj["z_templates"],
    }


def main():
    cfg = CONFIG

    templates = load_templates(cfg["template_path"])[cfg["coord"]]
    indices = cfg["indices"]

    exprs = [templates[i]["expr"] for i in indices]
    n = len(exprs)

    M = np.eye(n)

    for a in range(n):
        for b in range(a + 1, n):
            sim = dag_jaccard(
                exprs[a],
                exprs[b],
                ignore_const_values=not cfg["keep_const_values"],
                ignore_param_names=not cfg["keep_param_names"],
            )
            M[a, b] = sim
            M[b, a] = sim

    labels = [f"{cfg['coord']}{i}" for i in indices]
    df = pd.DataFrame(M, index=labels, columns=labels)

    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    upper = M[np.triu_indices(n, k=1)]
    upper_sum = upper.sum()
    upper_max = n * (n - 1) / 2
    coherence = upper_sum / upper_max if upper_max > 0 else 1.0
    
    # ---- largest common DAG subexpression across all selected templates ----
    sig_sets = []

    for expr in exprs:
        dag = SympyDagEvaluator(expr)
        sig_map, sigs = canonical_node_signature_map(
            dag,
            ignore_const_values=not cfg["keep_const_values"],
            ignore_param_names=not cfg["keep_param_names"],
        )
        sig_sets.append(set(sig_map.keys()))

    common = set.intersection(*sig_sets) if sig_sets else set()

    # Usually ignore trivial leaves unless you want "s" or "PARAM" to win.
    nontrivial_common = {
        sig for sig in common
        if dag_subexpr_size(sig) > 1
    }

    if nontrivial_common:
        best_sig = max(nontrivial_common, key=dag_subexpr_size)
        best_expr = dag_sig_to_sympy(best_sig)

        print()
        print("=== Largest common DAG subexpression ===")
        print(f"dag_size = {dag_subexpr_size(best_sig)}")
        print("latex:")
        print(sp.latex(best_expr))
        print("========================================\n")
        print("=== All Common nontrivial DAG subexpressions ===")
        nontrivial_common = sorted(nontrivial_common, key=dag_subexpr_size)
        for i, expr in enumerate(nontrivial_common):
            print(f"({i+1}) latex:", sp.latex(dag_sig_to_sympy(expr)))
    else:
        print()
        print("=== Largest common DAG subexpression ===")
        print("No nontrivial common subexpression found.")

    print()
    print(f"upper_triangle_sum = {upper_sum:.6f}")
    print(f"max_possible        = {upper_max:.6f}")
    print(f"coherence_score     = {coherence:.6f}")

    if cfg["out_csv"]:
        df.to_csv(cfg["out_csv"])
        print(f"\nSaved matrix to {cfg['out_csv']}")


if __name__ == "__main__":
    main()
