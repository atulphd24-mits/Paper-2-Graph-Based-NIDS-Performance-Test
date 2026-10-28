"""
Paper 2 figures (PDF + PNG) from the committed final-run files. Frozen-protocol results only.

Colour: the first three slots of the validated categorical palette (blue, orange, aqua; colour-blind safe,
all pairs), always paired with distinct marker shapes, hatching or direct labels so every figure reads in
greyscale. Every TPR figure states the FPR target; Figure 3 shows the FPR actually reached.

Usage: python scripts/make_paper2_figures.py  -> outputs/paper2_final/figures/
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper2_stats import load  # noqa: E402
from src.models.predictions import slug  # noqa: E402
from src.stats.protocol import bootstrap_ci  # noqa: E402

F = ROOT / "outputs/paper2_final"
OUT = F / "figures"
MAIN = "episodes gap 1 min, embargo 5 min"
CHRONO = "chronological, embargo 5 min"
BF = ["ToN-IoT", "CSE-CIC-IDS2018"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {"Brute_Force_-Web": "Brute Force-Web", "DDoS_attacks-LOIC-HTTP": "DDoS LOIC-HTTP",
         "Infilteration": "Infiltration", "DDOS_attack-HOIC": "DDoS HOIC", "DDOS_attack-LOIC-UDP": "DDoS LOIC-UDP",
         "DoS_attacks-GoldenEye": "DoS GoldenEye", "DoS_attacks-Hulk": "DoS Hulk",
         "DoS_attacks-SlowHTTPTest": "DoS SlowHTTPTest", "DoS_attacks-Slowloris": "DoS Slowloris",
         "Brute_Force_-XSS": "Brute Force-XSS", "SQL_Injection": "SQL injection", "FTP-BruteForce": "FTP brute force",
         "SSH-Bruteforce": "SSH brute force"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.dpi": 300})

CTX, GAT, EGS = load("context"), load("gated"), load("egraphsage")
PF = pd.read_csv(F / "stats/per_family_tpr.csv")
AMB = pd.read_csv(F / "v1_baseline/leakage_study/label_ambiguity_by_family.csv")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fam_name(f):
    return SHORT.get(f, f)


# ------------------------------------------------------------------------------- Figure 1: protocol

def fig1_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 4.0)
    ax.axis("off")
    col = {"train": BLUE, "validation": ORANGE, "test": AQUA}
    hatch = {"train": "", "validation": "////", "test": "...."}

    def box(x0, x1, y, kind, label=None):
        ax.add_patch(Rectangle((x0, y - 0.22), x1 - x0, 0.44, facecolor=col[kind], edgecolor="white",
                               hatch=hatch[kind], linewidth=0.8, alpha=0.9))
        if label:
            ax.text((x0 + x1) / 2, y - 0.36, label, ha="center", va="top", fontsize=7, color=INK)

    def embargo(x0, x1, y):
        ax.add_patch(Rectangle((x0, y - 0.3), x1 - x0, 0.6, facecolor="none", edgecolor=MUTED, linestyle=(0, (2, 2)),
                               linewidth=0.8))

    y = 3.2
    ax.text(0, y + 0.42, "Family with ≥ 3 full-data episodes (gap > 1 min): whole episodes go to one split",
            fontsize=8, color=INK, weight="bold")
    for x0, x1, k in [(2, 14, "train"), (20, 27, "validation"), (33, 41, "train"), (47, 52, "test"), (58, 70, "train"),
                      (76, 83, "test"), (89, 97, "train")]:
        box(x0, x1, y, k)
    for x in (27, 41, 52, 70, 83):
        embargo(x - 0.5, x + 5.5, y) if x in (41, 70) else None
    ax.text(2, y - 0.4, "test episodes → subset (i), cross-episode", fontsize=7, color=MUTED, va="top")
    ax.text(97, y - 0.4, "dashed: 5-min embargo; training flows near held-out units are purged", fontsize=6.5,
            color=MUTED, va="top", ha="right")

    y = 1.55
    ax.text(0, y + 0.42, "Family with < 3 full-data episodes: one episode split by time (subset (ii), within-episode)",
            fontsize=8, color=INK, weight="bold")
    box(2, 69, y, "train", "train (earliest 70%)")
    box(70, 84, y, "validation", "validation")
    box(85, 97, y, "test", "test → subset (ii)")

    y = 0.05
    ax.text(0, y, "Subset (iii) = all test attacks. Benign: 30-min blocks assigned like episodes; thresholds set on "
                  "validation benign flows for 1% FPR, then applied unchanged to test.", fontsize=7, color=MUTED)
    handles = [Patch(facecolor=col[k], hatch=hatch[k], edgecolor="white", label=k) for k in col]
    handles.append(Patch(facecolor="none", edgecolor=MUTED, linestyle=(0, (2, 2)), label="embargo"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=7)
    save(fig, "fig1_protocol")


# ---------------------------------------------------------------------------- Figure 2: composition

def fig2_composition():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    ax = axes[0]
    subsets = [("all", "(iii) all"), ("cross-episode", "(i) cross-ep."), ("fallback", "(ii) within-ep.")]
    markers = {"ToN-IoT": ("o", BLUE), "CSE-CIC-IDS2018": ("s", ORANGE)}
    for k, src in enumerate(BF):
        g = CTX[(CTX.source == src) & (CTX.protocol == MAIN) & (CTX.model == "HistGB") & (CTX.representation == "flow")]
        for i, (sub, _) in enumerate(subsets):
            v = g[g.subset == sub]["TPR@FPR1pct"].to_numpy()
            lo, hi = bootstrap_ci(v)
            x = i + (k - 0.5) * 0.28
            m, c = markers[src]
            ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt=m, color=c, ms=5, lw=1.2, capsize=0,
                        label=src if i == 0 else None)
            ax.text(x + 0.06, v.mean(), f"{v.mean():.2f}", fontsize=6.5, color=INK, va="center")
    ax.set_xticks(range(3), ["(iii)\nall", "(i)\ncross", "(ii)\nwithin"], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("TPR at 1% FPR (HistGB, flow)")
    ax.set_title("(a) Pooled vs subsets", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)

    amb = AMB.assign(source=AMB.source.map(slug)).set_index(["source", "attack"]).share_identical_to_benign
    for ax, src, title in ((axes[1], "CSE-CIC-IDS2018", "(b) CSE-CIC-IDS2018, subset (i)"),
                           (axes[2], "ToN-IoT", "(c) ToN-IoT, subset (i)")):
        pf = PF[PF.part.isin(["context_final_ctx_main", "context_final_hl_ctx_main"]) & (PF.source == slug(src))
                & (PF.protocol == slug(MAIN)) & (PF.score == "HistGB|flow") & (PF.subset == "(i) cross-episode")]
        g = pf.groupby("family").agg(tpr=("tpr", "mean"), share=("weight_share", "mean"), seeds=("seed", "nunique"))
        g = g.sort_values("share")
        y = np.arange(len(g))
        ax.barh(y, g.tpr, color=BLUE, height=0.55, edgecolor="white", linewidth=0.8)
        for yi, (fam, r) in zip(y, g.iterrows()):
            ceil = 1 - amb.get((slug(src), fam), np.nan)
            ax.plot([ceil, ceil], [yi - 0.38, yi + 0.38], color=INK, lw=1.6)
            ax.text(1.02, yi, f"{100 * r.share:.0f}%", fontsize=6.5, color=MUTED, va="center", transform=ax.get_yaxis_transform())
        ax.set_yticks(y, [f"{fam_name(f)} ({int(r.seeds)})" for f, r in g.iterrows()], fontsize=6.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("TPR at 1% FPR")
        ax.set_title(title, loc="left")
        ax.text(1.02, 1.0, "weight", fontsize=6.5, color=MUTED, transform=ax.transAxes, va="bottom")
    axes[1].plot([], [], color=INK, lw=1.6, label="1 − share identical to benign")
    axes[1].legend(loc="upper left", fontsize=6.3, bbox_to_anchor=(0.0, -0.22))
    fig.text(0.0, -0.1, "HistGB on flow features, main protocol, 30 seeds. (b, c): bars = design-weighted TPR; tick = "
             "share of the family's flows not identical to benign flows;\n"
             "right = share of subset (i) test-attack weight "
             "(mean over seeds in the subset); (n) = seeds in subset.", fontsize=6.5, color=MUTED)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_composition")


# ------------------------------------------------------------------------- Figure 3: reached FPR

def fig3_fpr():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    reps = [("flow", "o", BLUE), ("flow + neighbourhood", "^", AQUA), ("flow + host", "s", ORANGE),
            ("flow + neighbourhood + host", "D", MUTED)]
    protos = [(MAIN, "main"), ("episodes gap 1 min, embargo 0 min", "no embargo"),
              ("episodes gap 5 min, embargo 5 min", "5-min gap"), (CHRONO, "chronological")]
    rng = np.random.default_rng(0)
    for ax, src in zip(axes, BF):
        for i, (proto, plabel) in enumerate(protos):
            for j, (rep, m, c) in enumerate(reps):
                g = CTX[(CTX.source == src) & (CTX.protocol == proto) & (CTX.model == "HistGB")
                        & (CTX.representation == rep) & (CTX.subset == "all")]
                v = np.maximum(g["testFPR@1pct"].to_numpy(), 1e-4)
                x = i + (j - 1.5) * 0.18
                ax.scatter(x + rng.uniform(-0.04, 0.04, len(v)), v, marker=m, s=9, facecolor="none", edgecolor=c,
                           linewidth=0.7, alpha=0.8, label=rep if i == 0 else None)
                ax.plot([x - 0.07, x + 0.07], [v.mean()] * 2, color=INK, lw=1.4)
        ax.axhline(0.01, color=INK, lw=0.9, ls=(0, (4, 3)))
        ax.text(3.55, 0.0106, "target 1%", fontsize=6.5, color=INK, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(8e-5, 0.6)
        ax.set_xticks(range(4), [p[1] for p in protos], fontsize=7)
        ax.set_title(src, loc="left")
    axes[0].set_ylabel("FPR reached on test (log)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.8,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.0, -0.1, "HistGB; each marker one seed (main protocol 30 seeds, others 10), bar = mean. Thresholds set "
             "for 1% FPR on validation benign flows. Values below 0.01% drawn at 0.01%.", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig3_target_vs_reached_fpr")


# --------------------------------------------------------------- Figure 4: context and graph per family

def fig4_context():
    sel = json.loads((F / "reports/endpoint_selection.json").read_text())["best_endpoint"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, src in zip(axes, BF):
        base = PF[(PF.source == slug(src)) & (PF.protocol == slug(MAIN)) & (PF.subset == "(i) cross-episode")]
        hist = base[base.part.isin(["context_final_ctx_main", "context_final_hl_ctx_main"])]
        egs = base[base.part.isin(["egraphsage_final_egs_ind", f"egraphsage_final_hl_egs_ind_{slug(src)}"])
                   & (base.score == f"E-GraphSAGE ({sel[slug(src)]})|inductive")]
        series = [("HistGB flow", hist[hist.score == "HistGB|flow"], "o", BLUE),
                  ("HistGB flow + host", hist[hist.score == "HistGB|flow + host"], "s", ORANGE),
                  (f"E-GraphSAGE ({sel[slug(src)].replace('_', ':')})", egs, "^", AQUA)]
        fams = (hist[hist.score == "HistGB|flow"].groupby("family").weight_share.mean().sort_values().index.tolist())
        y = np.arange(len(fams))
        for k, (label, d, m, c) in enumerate(series):
            g = d.groupby("family").tpr.mean().reindex(fams)
            ax.scatter(g.to_numpy(), y + (k - 1) * 0.22, marker=m, s=24, color=c, edgecolor="white", linewidth=0.5,
                       label=label, zorder=3)
        ax.set_yticks(y, [fam_name(f) for f in fams], fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("TPR at 1% FPR, subset (i)")
        ax.set_title(src, loc="left")
        ax.legend(loc="upper center", fontsize=6.5, ncol=3, bbox_to_anchor=(0.45, -0.2))
    fig.text(0.0, -0.12, "Per-family design-weighted TPR, main protocol, 30 seeds (families ordered by share of "
             "subset (i) weight, largest at top). E-GraphSAGE endpoint chosen on validation PR-AUC.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig4_context_by_family")


# ------------------------------------------------------------------------------- Figure 5: drift

def fig5_drift():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, src in zip(axes, BF):
        a = PF[(PF.part == "context_final_ctx_main") & (PF.source == slug(src)) & (PF.protocol == slug(MAIN))
               & (PF.score == "HistGB|flow") & (PF.subset == "(iii) all")].groupby("family").agg(
            tpr=("tpr", "mean"), share=("weight_share", "mean"))
        b = PF[(PF.part == "context_final_ctx_chrono") & (PF.source == slug(src)) & (PF.protocol == slug(CHRONO))
               & (PF.score == "HistGB|flow") & (PF.subset == "(iii) all")].groupby("family").tpr.mean()
        a = a.assign(chrono=b).sort_values("share")
        y = np.arange(len(a))
        for yi, (fam, r) in zip(y, a.iterrows()):
            ax.plot([r.tpr, r.chrono], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(a.tpr, y, marker="o", s=22, color=BLUE, label="episodes (main)", zorder=3)
        ax.scatter(a.chrono, y, marker="s", s=22, color=ORANGE, label="chronological", zorder=3)
        ax.set_yticks(y, [f"{fam_name(f)} ({100 * r.share:.0f}%)" for f, r in a.iterrows()], fontsize=6.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("TPR at 1% FPR, all test attacks (iii)")
        ax.set_title(src, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=6.8, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.0, -0.1, "HistGB on flow features, seeds 0-9; (%) = family's share of test-attack weight under the "
             "main protocol. Chronological: every class trained on its early part, tested on its late part.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig5_drift_by_family")


# ------------------------------------------------------------------------- Figure 6: uncertainty

def fig6_uncertainty():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, src, c in ((axes[0], "ToN-IoT", BLUE), (axes[1], "CSE-CIC-IDS2018", ORANGE)):
        g = CTX[(CTX.source == src) & (CTX.protocol == MAIN) & (CTX.model == "HistGB") & (CTX.representation == "flow")
                & (CTX.subset == "cross-episode")].sort_values("TPR@FPR1pct").reset_index(drop=True)
        x = np.arange(len(g))
        ax.vlines(x, g["TPR@FPR1pct_ci_low"], g["TPR@FPR1pct_ci_high"], color=c, lw=1.0, alpha=0.7)
        ax.scatter(x, g["TPR@FPR1pct"], color=c, s=10, zorder=3)
        v = g["TPR@FPR1pct"].to_numpy()
        lo, hi = bootstrap_ci(v)
        ax.axhspan(lo, hi, color=GRID, zorder=0)
        ax.text(len(g) - 0.5, 0.03, f"band: seed-level 95% CI [{lo:.2f}, {hi:.2f}]", fontsize=6.5, ha="right", color=INK)
        ax.set_xticks([])
        ax.set_xlabel(f"seeds, sorted by TPR (n = {len(g)}; median {int(g.n_episodes.median())} test episodes)")
        ax.set_title(src, loc="left")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("TPR at 1% FPR, subset (i)")
    fig.text(0.0, -0.05, "HistGB on flow features, main protocol. Lines: 95% episode-level bootstrap CI within each "
             "seed (test episodes resampled); band: 95% CI of the mean over seeds.", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig6_episode_uncertainty")


def main():
    for f in (fig1_protocol, fig2_composition, fig3_fpr, fig4_context, fig5_drift, fig6_uncertainty):
        f()
        print("done", f.__name__)


if __name__ == "__main__":
    main()
