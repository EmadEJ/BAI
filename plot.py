import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from io_utils import *
from environment import ALGORITHMS

# Draws every figure and table from the stored results (nothing is rerun):
#   python plot.py all
#   python plot.py fig2 fig3 fig4 fig5 tightness tables
# The figures go to figures/ (pdf + png), the tables to figures/tables.md and figures/*.tex

FIG_DIR = "figures/"
FIG2_INSTANCES = [1, 2]
FIG3_INSTANCES = {4: "Instance (16)", 5: "Instance (17)"}
FIG5_N, FIG5_K = [3, 6, 9, 12], [2, 3, 4]  # instance index is 10 n + k
FIG5_INSTANCES = [10 * n + k for n in FIG5_N for k in FIG5_K]
DELTA_DIR = "results/delta_correctness/"

ORDER = ["ASTS", "MuSTS", "STS-LB", "STS-LB-Naive", "STS-Plain", "SGTS"]
SHORT_LABELS = {
    "ASTS": r"GTS-$\mathcal{A}$",
    "MuSTS": r"GTS-$\mu$",
    "STS-LB": "GTS-Unknown\n(LB)",
    "STS-LB-Naive": "GTS-Unknown\n(LB_Naive)",
    "STS-Plain": "GTS-Unknown\n(solver)",
    "SGTS": "GTS-SG",
}
TABLE_NAMES = {"ASTS": "GTS-A", "MuSTS": "GTS-mu", "STS-LB": "GTS-Unknown (LB)", "STS-LB-Naive": "GTS-Unknown (LB_Naive)", "STS-Plain": "GTS-Unknown (solver)", "SGTS": "GTS-SG"}
# Okabe-Ito colors, the same color for an algorithm and for the statistic it uses
COLORS = {
    "ASTS": "#0072B2",
    "MuSTS": "#009E73",
    "STS-LB": "#D55E00",
    "STS-LB-Naive": "#56B4E9",
    "STS-Plain": "#CC79A7",
    "SGTS": "#8C8C8C",
}
C_L, C_NAIVE, C_PLAIN, C_BETA = "#D55E00", "#56B4E9", "#CC79A7", "#332288"
SINGLE, DOUBLE = 3.25, 6.75  # AISTATS column widths in inches


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    Path(FIG_DIR).mkdir(exist_ok=True)
    fig.savefig(FIG_DIR + name + ".pdf")
    fig.savefig(FIG_DIR + name + ".png", dpi=200)
    plt.close(fig)
    print(f"saved {FIG_DIR}{name}.pdf")


def load_runs(index, algorithm, out_dir="results/simulation/"):
    data = load_json(simulation_path(index, algorithm, out_dir))
    return [] if data is None else data['runs']


############################## figure 2

def fig2(instances=FIG2_INSTANCES, name="fig2_glr_trace"):
    # the whole run on a linear scale, cut a bit above the naive threshold so that both thresholds are visible
    fig, axes = plt.subplots(1, len(instances), figsize=(DOUBLE if len(instances) > 1 else 4.2, 2.9), squeeze=False)
    axes = axes[0]
    statistics = [("reference", "k"), ("L", C_L), ("L_naive", C_NAIVE)]
    for ax, index in zip(axes, instances):
        d = load_json(f"results/glr_trace/instance{index}.json")
        if d is None:
            continue
        tr = {key: np.array(value, dtype=float) for key, value in d['trace'].items()}
        T, L, upper = tr['T'], tr['L'], tr['L_upper']
        upper[~np.isfinite(upper)] = np.nan
        # first round where each statistic is above each threshold
        crossing = {}
        for key, _ in statistics:
            for beta in ["beta", "naive_beta"]:
                above = np.nonzero(tr[key] > tr[beta])[0]
                crossing[key, beta] = above[0] if len(above) else None

        ax.fill_between(T, L, upper, color=C_L, alpha=0.13, lw=0)
        ax.plot(T, upper, color=C_L, lw=0.6, alpha=0.5)
        ax.plot(T, tr['plain'], color=C_PLAIN, lw=2.6, alpha=0.9)
        ax.plot(T, tr['reference'], color="k", ls=(0, (1.2, 1.2)), lw=1.3)
        ax.plot(T, L, color=C_L)
        ax.plot(T, tr['L_naive'], color=C_NAIVE)
        ax.plot(T, tr['beta'], color=C_BETA, lw=1.2)
        ax.plot(T, tr['naive_beta'], color=C_BETA, lw=1.2, ls="--")
        for key, color in statistics:
            for beta, face in [("beta", color), ("naive_beta", "white")]:
                j = crossing[key, beta]
                if j is not None:
                    ax.plot(T[j], tr[beta][j], marker="o", ms=4.5, mfc=face, mec=color, mew=1.1, zorder=5)
                    if beta == "beta":
                        ax.axvline(T[j], color=color, lw=0.6, ls=":", alpha=0.8)
        ax.set_xlim(0, T[-1])
        ax.set_ylim(0, 1.35 * tr['naive_beta'].max())
        ax.set_xlabel("Timestep $t$")
        ax.set_title(f"$n={d['n']},\\ k={d['k']},\\ \\delta=10^{{{int(np.log10(d['confidence']))}}}$")
    axes[0].set_ylabel("statistic")

    handles = [
        Line2D([], [], color=C_PLAIN, lw=2.6, label="GLR, plain solver"),
        Line2D([], [], color="k", ls=(0, (1.2, 1.2)), lw=1.3, label="GLR, true value"),
        Line2D([], [], color=C_L, label="$L$ (ours)"),
        Patch(color=C_L, alpha=0.13, label="proven zone of $L$"),
        Line2D([], [], color=C_NAIVE, label=r"$L_{\mathrm{naive}}$"),
        Line2D([], [], color=C_BETA, lw=1.2, label=r"our $\beta$"),
        Line2D([], [], color=C_BETA, lw=1.2, ls="--", label=r"naive $\beta$"),
        Line2D([], [], marker="o", ls="", mfc="0.4", mec="0.4", ms=4.5, label=r"stops with our $\beta$"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", mew=1.1, ms=4.5, label=r"stops with naive $\beta$"),
    ]
    ncol = 5 if len(instances) > 1 else 3
    fig.tight_layout(rect=(0, 0.11 if len(instances) > 1 else 0.22, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005), handlelength=1.6, columnspacing=1.0)
    save(fig, name)


############################## figures 3 and 5

def compact_number(x, pos=None):
    # 2000 -> 2k, 1500000 -> 1.5M
    for scale, suffix in [(1e6, "M"), (1e3, "k")]:
        if x >= scale:
            return f"{x / scale:g}{suffix}"
    return f"{x:g}"


def sample_complexity_panel(ax, index, show_labels=True):
    Ts = [np.array([run['T'] for run in load_runs(index, alg)]) for alg in ORDER]
    positions = [p for p, T in enumerate(Ts) if len(T)]
    if not positions:
        return
    data = [Ts[p] for p in positions]
    box = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False,
                     showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="k", markersize=3.5),
                     medianprops=dict(color="k", lw=1.0), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, p in zip(box['boxes'], positions):
        patch.set_facecolor(COLORS[ORDER[p]])
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for p, T in zip(positions, data):
        ax.scatter(p + rng.uniform(-0.18, 0.18, len(T)), T, s=4, color="k", alpha=0.35, lw=0, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(compact_number))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT_LABELS[alg] for alg in ORDER] if show_labels else [])
    ax.grid(axis="x", visible=False)


def algorithm_legend(fig, ax=None, ncol=3):
    handles = [Patch(facecolor=COLORS[alg], alpha=0.75, edgecolor="k", lw=0.6, label=ALGORITHMS[alg]) for alg in ORDER]
    handles.append(Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor="k", markersize=3.5, label="mean"))
    if ax is None:
        fig.legend(handles=handles, loc="lower center", ncol=ncol + 1, bbox_to_anchor=(0.5, -0.01), columnspacing=1.2)
    else:
        ax.legend(handles=handles, loc="center")


def fig3():
    fig, axes = plt.subplots(1, len(FIG3_INSTANCES), figsize=(DOUBLE, 2.4))
    for ax, (index, title) in zip(axes, FIG3_INSTANCES.items()):
        sample_complexity_panel(ax, index, show_labels=False)
        ax.set_title(title)
    axes[0].set_ylabel("Sample complexity")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    algorithm_legend(fig)
    save(fig, "fig3_sample_complexity")


def fig5():
    # one row per n and one column per k
    fig, axes = plt.subplots(len(FIG5_N), len(FIG5_K), figsize=(DOUBLE, 7.2))
    for r, n in enumerate(FIG5_N):
        for c, k in enumerate(FIG5_K):
            ax = axes[r, c]
            sample_complexity_panel(ax, 10 * n + k, show_labels=False)
            ax.set_title(f"$n={n},\\ k={k}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sample complexity")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    algorithm_legend(fig)
    save(fig, "fig5_sample_complexity")


############################## figure 4

def fig4(index=6):
    d = load_json(f"results/heatmaps/instance{index}.json")
    if d is None:
        return
    w = np.array(d['w'])
    h = np.sqrt(3) / 2
    x, y = w[:, 1] + 0.5 * w[:, 2], h * w[:, 2]
    triangulation = mtri.Triangulation(x, y)

    panels = [("ASTS", r"GTS-$\mathcal{A}$"), ("MuSTS", r"GTS-$\mu$"), ("STS", "GTS-Unknown"), ("SGTS", "GTS-SG")]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.35))
    for ax, (key, title) in zip(axes, panels):
        F = np.array(d['F'][key], dtype=float)
        F[~np.isfinite(F) | (F < 1e-8)] = 0  # the boundary of the simplex is evaluated at w_i = 1e-9
        contour = ax.tricontourf(triangulation, F, levels=40, cmap="viridis")
        contour.set_edgecolor("face")
        best = np.argmax(F)
        ax.plot(x[best], y[best], marker="*", ms=9, color="#FF4040", mec="white", mew=0.6)
        ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="k", lw=0.8)
        ax.text(0, -0.04, "$w_1{=}1$", ha="center", va="top", fontsize=7.5)
        ax.text(1, -0.04, "$w_2{=}1$", ha="center", va="top", fontsize=7.5)
        ax.text(0.5, h + 0.03, "$w_3{=}1$", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(title, pad=12)
        ax.text(0.5, -0.2, f"$w^*=({w[best][0]:.2f}, {w[best][1]:.2f}, {w[best][2]:.2f})$", ha="center", va="top", fontsize=7.5)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.3, h + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        cbar = fig.colorbar(contour, ax=ax, orientation="horizontal", fraction=0.06, pad=0.02, aspect=18)
        cbar.set_ticks([F.min(), F.max()])
        cbar.ax.set_xticklabels([f"{F.min():.1e}" if F.min() > 0 else "0", f"{F.max():.1e}"], fontsize=7)
        cbar.outline.set_linewidth(0.5)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig4_heatmaps")


############################## bound tightness

def tightness_pairs():
    results = load_json("results/bound_tightness/pairs.json")
    rows = []
    for r in results:
        for p in r['pairs']:
            rows.append({**{key: r.get(key) for key in ['family', 'instance', 'id', 'eps', 'alpha', 'n', 'k']}, **p})
    return results, rows


def fig_tightness():
    results, rows = tightness_pairs()
    # hard families: A_i = (1 - eps) A_star + eps A0_i
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    hard = [row for row in rows if row['family'] == 'hard']
    eps_values = sorted({row['eps'] for row in hard}, reverse=True)
    curves = [
        ("L_naive", r"$\Lambda_i / L_{\mathrm{naive},i}$", C_NAIVE, "-", "s"),
        ("L", r"$\Lambda_i / L_i$ (ours)", C_L, "-", "o"),
        ("factor", r"proven factor of $L_i$", C_L, "--", None),
    ]
    for key, label, color, ls, marker in curves:
        values = [[(row['factor'] if key == 'factor' else row['Lambda'] / row[key]) for row in hard if row['eps'] == eps] for eps in eps_values]
        med = [np.median(v) for v in values]
        lo, hi = [np.quantile(v, 0.25) for v in values], [np.quantile(v, 0.75) for v in values]
        ax.plot(eps_values, med, color=color, ls=ls, marker=marker, ms=3.5, label=label)
        if key != "factor":
            ax.fill_between(eps_values, lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10])
    ax.set_yticklabels(["1", "2", "3", "5", "10"])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel(r"$\epsilon$ (smaller is harder)")
    ax.set_ylabel("ratio to the true GLR")
    ax.legend(loc="center right", fontsize=7.5)

    fig.tight_layout()
    save(fig, "fig_bound_tightness")


def tightness_table():
    results, rows = tightness_pairs()

    def instance_ratios(r):
        # whole instance: Lambda = min_i Lambda_i, and Lambda <= min_i factor_i L_i
        Lam = min(p['Lambda'] for p in r['pairs'])
        L = min(p['L'] for p in r['pairs'])
        Ln = min(p['L_naive'] for p in r['pairs'])
        factor = min(p['factor'] * p['L'] for p in r['pairs']) / L
        return Lam / Ln, Lam / L, factor

    lines = []
    for r in results:
        if r['family'] == 'paper':
            lines.append((f"{r['name']} ($n={r['n']}, k={r['k']}$)", *instance_ratios(r)))
    fmt = lambda v: f"{np.median(v):.2f} ({np.quantile(v, 0.9):.1f})"
    random_rows = [row for row in rows if row['family'] == 'random']
    for alpha in sorted({row['alpha'] for row in random_rows}) + [None]:
        family = [row for row in random_rows if alpha is None or row['alpha'] == alpha]
        name = "random, all" if alpha is None else f"random, $\\alpha={alpha:g}$"
        lines.append((f"{name} ({len(family)} pairs): median (90\\%)",
                      fmt([row['Lambda'] / row['L_naive'] for row in family]),
                      fmt([row['Lambda'] / row['L'] for row in family]),
                      fmt([row['factor'] for row in family])))
    for eps in sorted({row['eps'] for row in rows if row['family'] == 'hard'}, reverse=True):
        hard = [row for row in rows if row['family'] == 'hard' and row['eps'] == eps]
        lines.append((f"hard, $\\epsilon={eps:g}$: median", np.median([row['Lambda'] / row['L_naive'] for row in hard]),
                      np.median([row['Lambda'] / row['L'] for row in hard]), np.median([row['factor'] for row in hard])))
    for r in results:
        if r['family'] == 'prop5':
            p = r['pairs'][0]
            lines.append((f"Prop. 5, $\\epsilon=10^{{{int(round(np.log10(r['eps'])))}}}$", p['Lambda'] / p['L_naive'], p['Lambda'] / p['L'], p['factor']))

    cell = lambda v: v if isinstance(v, str) else f"{v:.3g}"
    tex = ["\\begin{tabular}{lccc}", "\\toprule",
           "Setting & $\\Lambda/L_{\\mathrm{naive}}$ & $\\Lambda/L$ & proven factor ($L$) \\\\", "\\midrule"]
    tex += [" & ".join(cell(v) for v in line) + " \\\\" for line in lines]
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path(FIG_DIR).mkdir(exist_ok=True)
    with open(FIG_DIR + "table_bound_tightness.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    md = ["| Setting | Λ/L_naive | Λ/L | proven factor (L) |", "|---|---|---|---|"]
    md += ["| " + " | ".join(cell(v).replace("$", "").replace("\\", "") for v in line) + " |" for line in lines]

    valid = all(row['L'] <= row['Lambda'] * (1 + 1e-6) <= row['factor'] * row['L'] * (1 + 1e-5) for row in rows)
    md.append(f"\nL <= Lambda <= factor * L on all {len(rows)} pairs: {valid}")

    times = load_json("results/bound_tightness/characteristic_times.json")
    if times is not None:
        md += ["", "| instances | U/T* | U_naive/T* |", "|---|---|---|"]
        tex = ["\\begin{tabular}{lcc}", "\\toprule", "instances & $U/T^*$ & $U_{\\mathrm{naive}}/T^*$ \\\\", "\\midrule"]
        table = [(f"{t['name']} ($n={t['n']}, k={t['k']}$)", f"{t['U'] / t['T_star']:.2f}", f"{t['U_naive'] / t['T_star']:.2f}")
                 for t in times if t['family'] == 'paper']
        for alpha in sorted({t['alpha'] for t in times if t['family'] == 'random'}):
            family = [t for t in times if t['family'] == 'random' and t['alpha'] == alpha]
            table.append((f"random, $\\alpha={alpha:g}$ ({len(family)} instances): median (90\\%)",
                          fmt([t['U'] / t['T_star'] for t in family]), fmt([t['U_naive'] / t['T_star'] for t in family])))
        for line in table:
            md.append("| " + " | ".join(v.replace("$", "").replace("\\", "") for v in line) + " |")
            tex.append(" & ".join(line) + " \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        with open(FIG_DIR + "table_characteristic_times.tex", "w") as f:
            f.write("\n".join(tex) + "\n")
    return md


############################## tables of the simulations

def simulation_tables():
    md = ["## Sample complexity, error rate and runtime", "",
          "| instance | algorithm | runs | mean T | median T | error rate | mean time / run (s) | ms / round |",
          "|---|---|---|---|---|---|---|---|"]
    for index in list(FIG3_INSTANCES) + FIG5_INSTANCES:
        for alg in ORDER:
            runs = load_runs(index, alg)
            if not runs:
                continue
            T = np.array([run['T'] for run in runs])
            times = np.array([run['time'] for run in runs])
            error = 1 - np.mean([run['correct'] for run in runs])
            md.append(f"| {index} | {TABLE_NAMES[alg]} | {len(runs)} | {T.mean():.0f} | {np.median(T):.0f} | {error:.3f} | "
                      f"{times.mean():.1f} | {times.sum() / T.sum() * 1e3:.3f} |")

    if Path(DELTA_DIR).exists():
        md += ["", "## delta-correctness", "",
               "| instance | algorithm | delta | runs | errors | error rate | 95% upper bound |", "|---|---|---|---|---|---|---|"]
        for path in sorted(Path(DELTA_DIR).glob("*.json")):
            data = load_json(path)
            runs = data['runs']
            errors = sum(not run['correct'] for run in runs)
            md.append(f"| {data['instance']} | {TABLE_NAMES[data['algorithm']]} | {data['confidence']} | {len(runs)} | {errors} | "
                      f"{errors / len(runs):.3f} | {clopper_pearson_upper(errors, len(runs)):.3f} |")
    return md


def clopper_pearson_upper(errors, runs, level=0.95):
    from scipy.stats import beta
    return 1.0 if errors == runs else beta.ppf(level, errors + 1, runs - errors)


if __name__ == "__main__":
    set_style()
    targets = sys.argv[1:] or ["all"]
    # numbers are instances for a figure 2 of their own: python plot.py fig2 16 17 -> figures/fig2_glr_trace_16_17.pdf
    fig2_instances = [int(t) for t in targets if t.isdigit()]
    if "all" in targets:
        targets = ["fig2", "fig3", "fig4", "fig5", "tightness", "tables"]
    if "fig2" in targets and fig2_instances:
        fig2(fig2_instances, "fig2_glr_trace_" + "_".join(map(str, fig2_instances)))
    elif "fig2" in targets:
        fig2()
        if Path("results/glr_trace/instance3.json").exists():
            fig2([3], "fig2_glr_trace_k4")  # the same on an instance with k = 4
        if Path("results/glr_trace/instance9.json").exists():
            fig2([9], "fig2_glr_trace_solver_miss")  # an instance where the plain solver stays above the true GLR
    if "fig3" in targets:
        fig3()
    if "fig4" in targets:
        fig4()
    if "fig5" in targets:
        fig5()
    if "tightness" in targets:
        fig_tightness()
    if "tightness" in targets or "tables" in targets:
        # tables.md always gets every table, so drawing one figure never drops the others
        md = ["## Bound tightness", ""] + tightness_table() + [""] + simulation_tables()
        Path(FIG_DIR).mkdir(exist_ok=True)
        with open(FIG_DIR + "tables.md", "w") as f:
            f.write("\n".join(md) + "\n")
        print("\n".join(md))
