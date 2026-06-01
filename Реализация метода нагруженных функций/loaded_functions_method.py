import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "loaded_functions_mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / "graphs"
GENERATED_DIR = BASE_DIR / "generated"
GRAPHS_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

L = 1.0
M = 20.0
X_N = 5001
T_N = 1201
TOL = 1e-4
G_TOL = 1e-8


def max0(z):
    return np.maximum(z, 0.0)


def f_a(x):
    return np.arctan(x)


def f_b(x):
    return x * np.sin(x)


def f_g(x):
    x = np.asarray(x, dtype=float)
    result = np.ones_like(x)
    mask = x > 1.0
    result[mask] = 1.0 / x[mask]
    return result


def f_d(x):
    return np.ones_like(np.asarray(x, dtype=float))


def g_square_minus_4(x):
    return x**2 - 4.0


def g_square_minus_4_scaled(x):
    return (x**2 - 4.0) / (x**4 + 1.0)


def g_identity(x):
    return x


def g_identity_scaled(x):
    return x / (x**2 + 1.0)


def g_square(x):
    return x**2


def g_square_minus_1(x):
    return x**2 - 1.0


def g_square_minus_1_scaled(x):
    return (x**2 - 1.0) / (x**4 + 1.0)


def g_square_minus_1_exp(x):
    return (x**2 - 1.0) * np.exp(-(x**2))


def g_minus_1(x):
    return x - 1.0


def g_minus_2_scaled(x):
    return (x - 2.0) / (x**2 + 1.0)


def g_exp_negative_square(x):
    return np.exp(-(x**2))


def g_square_plus_1(x):
    return x**2 + 1.0


def g_exp_square_plus_1(x):
    return np.exp(-(x**2)) * (x**2 + 1.0)


def g_square_exp(x):
    return x**2 * np.exp(-(x**2))


X0_VARIANTS = {
    "xge1": {
        "latex": r"$\{u\in E^1:\ u\geq 1\}$",
        "text": "u >= 1",
        "bounds": (1.0, math.inf),
    },
    "xge0": {
        "latex": r"$\{u\in E^1:\ u\geq 0\}$",
        "text": "u >= 0",
        "bounds": (0.0, math.inf),
    },
    "R": {
        "latex": r"$E^1$",
        "text": "E^1",
        "bounds": (-math.inf, math.inf),
    },
}


TASKS = {
    "a": {
        "label": "а",
        "f": f_a,
        "f_latex": r"$J(u)=\arctan u$",
        "g_variants": [
            ("g1", r"$u^2-4$", "u^2 - 4", g_square_minus_4),
            ("g2", r"$(u^2-4)(u^4+1)^{-1}$", "(u^2 - 4)/(u^4 + 1)", g_square_minus_4_scaled),
            ("g3", r"$u$", "u", g_identity),
            ("g4", r"$u(u^2+1)^{-1}$", "u/(u^2 + 1)", g_identity_scaled),
            ("g5", r"$u^2$", "u^2", g_square),
        ],
        "x0_keys": ["xge1", "xge0", "R"],
    },
    "b": {
        "label": "б",
        "f": f_b,
        "f_latex": r"$J(u)=u\sin u$",
        "g_variants": [
            ("g1", r"$u^2-1$", "u^2 - 1", g_square_minus_1),
            ("g2", r"$(u^2-1)(u^4+1)^{-1}$", "(u^2 - 1)/(u^4 + 1)", g_square_minus_1_scaled),
            ("g3", r"$(u^2-1)e^{-u^2}$", "(u^2 - 1) exp(-u^2)", g_square_minus_1_exp),
        ],
        "x0_keys": ["xge0"],
    },
    "g": {
        "label": "г",
        "f": f_g,
        "f_latex": r"$J(u)=1,\ u\leq 1;\ J(u)=u^{-1},\ u>1$",
        "g_variants": [
            ("g1", r"$u-1$", "u - 1", g_minus_1),
            ("g2", r"$(u-2)(u^2+1)^{-1}$", "(u - 2)/(u^2 + 1)", g_minus_2_scaled),
            ("g3", r"$e^{-u^2}$", "exp(-u^2)", g_exp_negative_square),
        ],
        "x0_keys": ["xge0", "R"],
    },
    "d": {
        "label": "д",
        "f": f_d,
        "f_latex": r"$J(u)=1$",
        "g_variants": [
            ("g1", r"$u$", "u", g_identity),
            ("g2", r"$u^2+1$", "u^2 + 1", g_square_plus_1),
            ("g3", r"$e^{-u^2}(u^2+1)$", "exp(-u^2)(u^2 + 1)", g_exp_square_plus_1),
            ("g4", r"$u^2e^{-u^2}$", "u^2 exp(-u^2)", g_square_exp),
        ],
        "x0_keys": ["R", "xge0"],
    },
}


def numeric_window(task_key, x0_key):
    if task_key == "a":
        return {"xge1": (1.0, 4.0), "xge0": (0.0, 4.0), "R": (-20.0, 4.0)}[x0_key]
    if task_key == "b":
        return (0.0, 4.0)
    if task_key == "g":
        return {"xge0": (0.0, 8.0), "R": (-8.0, 8.0)}[x0_key]
    if task_key == "d":
        return {"xge0": (0.0, 8.0), "R": (-8.0, 8.0)}[x0_key]
    raise KeyError(task_key)


def x0_mask(x, bounds):
    lo, hi = bounds
    mask = np.ones_like(x, dtype=bool)
    if math.isfinite(lo):
        mask &= x >= lo - G_TOL
    if math.isfinite(hi):
        mask &= x <= hi + G_TOL
    return mask


def domain_from_x0(x0_bounds, window):
    x_lo, x_hi = x0_bounds
    w_lo, w_hi = window
    lo = max(w_lo, x_lo) if math.isfinite(x_lo) else w_lo
    hi = min(w_hi, x_hi) if math.isfinite(x_hi) else w_hi
    if lo > hi:
        return None
    return lo, hi


def phi_values(fvals, pvals, t, phi_kind):
    if phi_kind == "13":
        return L * max0(fvals - t) + M * pvals
    return L * np.abs(fvals - t) + M * pvals


def rho_curve(fvals, pvals, t_grid, phi_kind, chunk=120):
    rho = np.empty_like(t_grid)
    argmin_idx = np.empty_like(t_grid, dtype=int)
    for start in range(0, len(t_grid), chunk):
        stop = min(start + chunk, len(t_grid))
        t = t_grid[start:stop, None]
        if phi_kind == "13":
            vals = L * np.maximum(fvals[None, :] - t, 0.0) + M * pvals[None, :]
        else:
            vals = L * np.abs(fvals[None, :] - t) + M * pvals[None, :]
        idx = np.argmin(vals, axis=1)
        rows = np.arange(stop - start)
        rho[start:stop] = vals[rows, idx]
        argmin_idx[start:stop] = idx
    return rho, argmin_idx


def solve_variant(task, g_func, x0_key, phi_kind):
    x0 = X0_VARIANTS[x0_key]
    domain = domain_from_x0(x0["bounds"], numeric_window(task["task_key"], x0_key))
    if domain is None:
        return {
            "x_grid": np.array([]),
            "f_grid": np.array([]),
            "g_grid": np.array([]),
            "p_grid": np.array([]),
            "t_grid": np.array([]),
            "rho_grid": np.array([]),
            "x_star": math.nan,
            "t_star": math.nan,
            "f_star_num": math.nan,
            "rho_star": math.nan,
            "has_root": False,
            "feasible_count": 0,
            "status": "empty_domain",
        }

    x_base = np.linspace(domain[0], domain[1], X_N)
    special = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    special = special[(special >= domain[0] - G_TOL) & (special <= domain[1] + G_TOL)]
    x = np.unique(np.concatenate([x_base, special]))
    fvals = task["f"](x)
    gvals = g_func(x)
    pvals = max0(gvals)
    feasible = x0_mask(x, x0["bounds"]) & (gvals <= G_TOL)
    feasible_count = int(np.count_nonzero(feasible))

    finite_f = fvals[np.isfinite(fvals)]
    f_min = float(np.min(finite_f))
    f_max = float(np.max(finite_f))
    t_margin = max(0.5, 0.1 * (f_max - f_min + 1.0))
    t_grid = np.linspace(f_min - t_margin, f_max + t_margin, T_N)
    rho, argmin_idx = rho_curve(fvals, pvals, t_grid, phi_kind)

    zero_idx = np.where(rho <= TOL)[0]
    has_root = len(zero_idx) > 0
    if has_root:
        first = int(zero_idx[0])
        left = t_grid[max(first - 1, 0)]
        right = t_grid[first]
        t_ref = np.linspace(left, right, 1200)
        rho_ref, arg_ref = rho_curve(fvals, pvals, t_ref, phi_kind)
        ref_zero = np.where(rho_ref <= TOL)[0]
        j = int(ref_zero[0]) if len(ref_zero) else int(np.argmin(rho_ref))
        t_star = float(t_ref[j])
        idx = int(arg_ref[j])
        rho_star = float(rho_ref[j])
    else:
        best = int(np.argmin(rho))
        left = t_grid[max(best - 1, 0)]
        right = t_grid[min(best + 1, len(t_grid) - 1)]
        t_ref = np.linspace(left, right, 2400)
        rho_ref, arg_ref = rho_curve(fvals, pvals, t_ref, phi_kind)
        ref_zero = np.where(rho_ref <= TOL)[0]
        has_root = len(ref_zero) > 0
        j = int(ref_zero[0]) if has_root else int(np.argmin(rho_ref))
        t_star = float(t_ref[j])
        idx = int(arg_ref[j])
        rho_star = float(rho_ref[j])

    x_star = float(x[idx])
    return {
        "x_grid": x,
        "f_grid": fvals,
        "g_grid": gvals,
        "p_grid": pvals,
        "t_grid": t_grid,
        "rho_grid": rho,
        "x_star": x_star,
        "t_star": t_star,
        "f_star_num": float(task["f"](np.array([x_star]))[0]),
        "rho_star": rho_star,
        "has_root": has_root,
        "feasible_count": feasible_count,
        "status": "ok" if has_root else "no_root",
        "domain": domain,
    }


def analytic_info(task_key, g_key, x0_key):
    pi = math.pi
    if task_key == "a":
        if g_key in {"g1", "g2"}:
            return {
                "x": {"xge1": "1", "xge0": "0", "R": "-2"}[x0_key],
                "j": {"xge1": math.atan(1.0), "xge0": 0.0, "R": math.atan(-2.0)}[x0_key],
                "status": "достигается",
            }
        if g_key in {"g3", "g4"}:
            if x0_key == "xge1":
                return {"x": "-", "j": None, "status": r"$U=\varnothing$"}
            if x0_key == "xge0":
                return {"x": "0", "j": 0.0, "status": "достигается"}
            return {"x": r"$-\infty$", "j": -pi / 2.0, "status": "inf не достигается"}
        if g_key == "g5":
            if x0_key == "xge1":
                return {"x": "-", "j": None, "status": r"$U=\varnothing$"}
            return {"x": "0", "j": 0.0, "status": "достигается"}
    if task_key == "b":
        return {"x": "0", "j": 0.0, "status": "достигается"}
    if task_key == "g":
        if g_key == "g1":
            return {"x": "[0,1]" if x0_key == "xge0" else r"$(-\infty,1]$", "j": 1.0, "status": "неединственный минимум"}
        if g_key == "g2":
            return {"x": "2", "j": 0.5, "status": "достигается"}
        if g_key == "g3":
            return {"x": "-", "j": None, "status": r"$U=\varnothing$"}
    if task_key == "d":
        if g_key in {"g2", "g3"}:
            return {"x": "-", "j": None, "status": r"$U=\varnothing$"}
        return {"x": "0" if x0_key == "xge0" or g_key == "g4" else r"$(-\infty,0]$", "j": 1.0, "status": "неединственный минимум" if g_key == "g1" and x0_key == "R" else "достигается"}
    raise KeyError((task_key, g_key, x0_key))


def fmt_num(value, digits=6):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    rounded = round(float(value), digits)
    if abs(rounded) < 10 ** (-digits):
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def tex_num(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    rounded = round(float(value), digits)
    if abs(rounded) < 10 ** (-digits):
        rounded = 0.0
    return f"{rounded:.{digits}f}"


def plot_graph1(ax, task, g_func, x0_key, result, phi_kind):
    x0 = X0_VARIANTS[x0_key]
    domain = result["domain"]
    x = np.linspace(domain[0], domain[1], 900)
    y = task["f"](x)
    feasible = x0_mask(x, x0["bounds"]) & (g_func(x) <= G_TOL)
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.1, 0.08 * (ymax - ymin + 1.0))

    ax.plot(x, y, linewidth=1.8, label="$J(u)$")
    if np.any(feasible):
        ax.fill_between(x, ymin - pad, ymax + pad, where=feasible, alpha=0.18, label="$U$")
    ax.scatter([result["x_star"]], [result["f_star_num"]], s=42, marker="o", label="числ. точка")
    ax.axvline(result["x_star"], linestyle="--", linewidth=0.8)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_title(f"$\\Phi_{{{phi_kind}}}$: график 1", fontsize=10)
    ax.set_xlabel("$u$")
    ax.set_ylabel("$J(u)$")
    ax.grid(True, alpha=0.25)


def plot_graph2(ax, task, g_func, x0_key, result, phi_kind):
    x0 = X0_VARIANTS[x0_key]
    domain = result["domain"]
    x = np.linspace(domain[0], domain[1], 450)
    t_grid = result["t_grid"]
    t = np.linspace(float(t_grid[0]), float(t_grid[-1]), 260)
    X, T = np.meshgrid(x, t)
    F = task["f"](X)
    P = max0(g_func(X))
    if phi_kind == "13":
        Z = L * np.maximum(F - T, 0.0) + M * P
    else:
        Z = L * np.abs(F - T) + M * P

    zmax = np.nanpercentile(Z, 95)
    levels = np.linspace(0.0, max(zmax, TOL), 30)
    cs = ax.contourf(X, T, Z, levels=levels, extend="max")
    feasible = x0_mask(x, x0["bounds"]) & (g_func(x) <= G_TOL)
    if np.any(feasible):
        ax.fill_between(x, t.min(), t.max(), where=feasible, alpha=0.16, color="white", label="$U$")
    ax.scatter([result["x_star"]], [result["t_star"]], s=42, marker="o", color="black", label="точка")
    ax.set_title(f"$\\Phi_{{{phi_kind}}}$: график 2", fontsize=10)
    ax.set_xlabel("$u$")
    ax.set_ylabel("$t$")
    ax.grid(True, alpha=0.15)
    return cs


def plot_overview(task_key, task, g_key, g_latex, g_func, x0_key, results):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for col, phi_kind in enumerate(("13", "32")):
        plot_graph1(axes[0, col], task, g_func, x0_key, results[phi_kind], phi_kind)
        cs = plot_graph2(axes[1, col], task, g_func, x0_key, results[phi_kind], phi_kind)
        fig.colorbar(cs, ax=axes[1, col], fraction=0.046, pad=0.04)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    title = f"Задача {task['label']}: {task['f_latex']}, g(u)={g_latex}, X0={X0_VARIANTS[x0_key]['latex']}"
    fig.suptitle(title, y=0.985, fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    filename = f"{task_key}_{g_key}_{x0_key}_overview.png"
    path = GRAPHS_DIR / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def generate_results():
    all_results = {}
    rows = []
    figure_entries = []
    figure_no = 1

    for task_key, task_spec in TASKS.items():
        task = dict(task_spec)
        task["task_key"] = task_key
        for g_key, g_latex, g_text, g_func in task["g_variants"]:
            for x0_key in task["x0_keys"]:
                combo_key = (task_key, g_key, x0_key)
                combo_results = {}
                for phi_kind in ("13", "32"):
                    combo_results[phi_kind] = solve_variant(task, g_func, x0_key, phi_kind)
                all_results[combo_key] = combo_results

                fig_path = plot_overview(task_key, task, g_key, g_latex, g_func, x0_key, combo_results)
                info = analytic_info(task_key, g_key, x0_key)
                row = {
                    "case": task["label"],
                    "task_key": task_key,
                    "g_key": g_key,
                    "g": g_text,
                    "g_latex": g_latex,
                    "x0_key": x0_key,
                    "X0": X0_VARIANTS[x0_key]["text"],
                    "X0_latex": X0_VARIANTS[x0_key]["latex"],
                    "analytic_x": info["x"],
                    "analytic_J": info["j"],
                    "status": info["status"],
                    "phi13_x": combo_results["13"]["x_star"],
                    "phi13_t": combo_results["13"]["t_star"],
                    "phi13_rho": combo_results["13"]["rho_star"],
                    "phi13_root": combo_results["13"]["has_root"],
                    "phi32_x": combo_results["32"]["x_star"],
                    "phi32_t": combo_results["32"]["t_star"],
                    "phi32_rho": combo_results["32"]["rho_star"],
                    "phi32_root": combo_results["32"]["has_root"],
                    "figure": fig_path.relative_to(BASE_DIR).as_posix(),
                    "figure_no": figure_no,
                }
                rows.append(row)
                figure_entries.append(row)
                figure_no += 1

    return pd.DataFrame(rows), figure_entries


def write_csv(summary_df):
    csv_df = summary_df.copy()
    for col in ["analytic_J", "phi13_x", "phi13_t", "phi13_rho", "phi32_x", "phi32_t", "phi32_rho"]:
        csv_df[col] = csv_df[col].map(lambda v: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt_num(v, 8))
    csv_df.to_csv(BASE_DIR / "results_loaded_functions.csv", index=False)


def write_latex_table(summary_df):
    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{c c p{3.2cm} p{2.7cm} p{2.7cm} c c c c c}",
        r"\caption{Результаты для всех вариантов $g(u)$ и $X_0$. Если $U=\varnothing$, условие $t_*=J_*$ не проверяется.}\\",
        r"\toprule",
        r"№ & п. & $g(u)$ & $X_0$ & аналитически & $x_{13}$ & $t_{13}$ & $x_{32}$ & $t_{32}$ & рис. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"№ & п. & $g(u)$ & $X_0$ & аналитически & $x_{13}$ & $t_{13}$ & $x_{32}$ & $t_{32}$ & рис. \\",
        r"\midrule",
        r"\endhead",
    ]
    for idx, row in summary_df.iterrows():
        analytic = row["status"] if row["analytic_J"] is None else f"$J_*={tex_num(row['analytic_J'])}$; {row['status']}"
        if not row["phi13_root"]:
            phi13_t = r"\text{нет}"
        else:
            phi13_t = tex_num(row["phi13_t"])
        if not row["phi32_root"]:
            phi32_t = r"\text{нет}"
        else:
            phi32_t = tex_num(row["phi32_t"])
        lines.append(
            f"{idx + 1} & {row['case']} & {row['g_latex']} & {row['X0_latex']} & "
            f"{analytic} & {tex_num(row['phi13_x'])} & {phi13_t} & "
            f"{tex_num(row['phi32_x'])} & {phi32_t} & {int(row['figure_no'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{landscape}"])
    (GENERATED_DIR / "results_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_latex_figures(summary_df):
    lines = []
    for _, row in summary_df.iterrows():
        g_formula = str(row["g_latex"]).strip("$")
        x0_formula = str(row["X0_latex"]).strip("$")
        caption = (
            f"Задача {row['case']}, вариант {row['g_key']}: "
            f"$g(u)={g_formula}$, $X_0={x0_formula}$. "
            r"Верхний ряд: график 1 для $\Phi_{13}$ и $\Phi_{32}$; "
            r"нижний ряд: график 2 для $\Phi_{13}$ и $\Phi_{32}$."
        )
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.96\textwidth]{{{row['figure']}}}",
                rf"\caption{{{caption}}}",
                r"\end{figure}",
                "",
            ]
        )
    (GENERATED_DIR / "figures.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    summary_df, _ = generate_results()
    write_csv(summary_df)
    write_latex_table(summary_df)
    write_latex_figures(summary_df)
    print(f"Сохранена таблица: {(BASE_DIR / 'results_loaded_functions.csv').resolve()}")
    print(f"Сохранены графики: {GRAPHS_DIR.resolve()}")
    print(f"Сохранены LaTeX-фрагменты: {GENERATED_DIR.resolve()}")
    return summary_df


if __name__ == "__main__":
    main()
