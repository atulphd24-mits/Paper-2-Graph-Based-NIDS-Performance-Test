"""
Paper figures and LaTeX tables from the leakage-study CSVs.

Figures (PDF + PNG) -> outputs/figures/paper/
Tables  (LaTeX)     -> outputs/reports/paper_tables/

Usage:
    python scripts/make_paper_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "outputs/reports/leakage_study"
FIGURES = PROJECT_ROOT / "outputs/figures/paper"
TABLES = PROJECT_ROOT / "outputs/reports/paper_tables"

# Reference palette, light mode (validated: adjacent CVD dE >= 9.1).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

MAIN_SOURCES = ["ToN-IoT", "CSE-CIC-IDS2018"]
REPRESENTATIONS = ["flow", "flow + neighbourhood", "flow + host", "flow + neighbourhood + host"]
SPLITS = ["random rows", "10-min time blocks", "attack episodes"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "text.color": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean(name):
    """Readable attack names: 'Brute_Force_-XSS' -> 'Brute Force XSS'."""
    return name.replace("_-", " ").replace("_", " ").replace("Infilteration", "Infiltration")


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def bar_labels(ax, bars, values, offset=0.01, fontsize=6):
    """Compact value labels ('.92', '1.0', '0') so neighbouring bars never collide."""
    for bar, value in zip(bars, values):
        text = "0" if value < 0.005 else "1.0" if value > 0.995 else f"{value:.2f}".lstrip("0")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                text, ha="center", va="bottom", fontsize=fontsize, color=TEXT_SECONDARY)


# ----------------------------------------------------------------------
# Figure 1 - label ambiguity
# ----------------------------------------------------------------------

def figure_ambiguity():
    fam = pd.read_csv(RESULTS / "label_ambiguity_by_family.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, source in zip(axes, MAIN_SOURCES):
        d = fam[fam["source"] == source].sort_values("share_identical_to_benign")
        ax.barh(d["attack"].map(clean), d["share_identical_to_benign"], color=SERIES[0], height=0.6)
        for y, v in enumerate(d["share_identical_to_benign"]):
            ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xlim(0, 1.12)
        ax.set_title(source, fontsize=9, loc="left", color=TEXT)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.supxlabel("Share of attack flows whose exact feature vector also occurs as benign",
                  fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, "fig1_label_ambiguity")


# ----------------------------------------------------------------------
# Figure 2 - split leakage
# ----------------------------------------------------------------------

def figure_splits():
    s = pd.read_csv(RESULTS / "part3_splits_summary.csv")
    sources = ["ToN-IoT", "CSE-CIC-IDS2018", "BoT-IoT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.15})
    width = 0.27
    x = np.arange(len(sources))

    panels = [
        ("behaviour", "F1_w_mean", "F1_w_std", "Behaviour features: weighted F1"),
        ("timestamp only", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
         "Timestamp only: detection @ 1% FPR"),
    ]
    for ax, (rep, metric, err, title) in zip(axes, panels):
        for i, split in enumerate(SPLITS):
            d = s[(s["representation"] == rep) & (s["split"] == split)].set_index("source")
            values = d.reindex(sources)[metric].to_numpy()
            errors = d.reindex(sources)[err].to_numpy()
            bars = ax.bar(x + (i - 1) * width, values, width - 0.03, yerr=errors,
                          color=SERIES[i], label=split, error_kw={"lw": 0.8, "ecolor": TEXT_SECONDARY})
            bar_labels(ax, bars, values)
        ax.set_xticks(x, ["ToN-IoT", "CIC-IDS2018", "BoT-IoT"])
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].legend(loc="lower left", fontsize=7, ncols=3, bbox_to_anchor=(0, 1.08))
    fig.text(0.01, -0.03, "Mean ± std over 5 seeds; test flows re-weighted to each source's "
             "real attack share. BoT-IoT is 99.7% attacks, so its F1 is near 1 regardless.",
             fontsize=7, color=TEXT_SECONDARY)
    save(fig, "fig2_split_leakage")


# ----------------------------------------------------------------------
# Figure 3 - context features under the episode split
# ----------------------------------------------------------------------

def figure_context():
    s = pd.read_csv(RESULTS / "part4_graph_summary.csv")
    t = pd.read_csv(RESULTS / "part4_graph_significance.csv")
    t = t[t["metric"] == "TPR@FPR1pct"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, source in zip(axes, MAIN_SOURCES):
        d = s[s["source"] == source].set_index("representation").reindex(REPRESENTATIONS)
        x = np.arange(len(REPRESENTATIONS))
        bars = ax.bar(x, d["TPR@FPR1pct_mean"], 0.62, yerr=d["TPR@FPR1pct_std"],
                      color=SERIES, error_kw={"lw": 0.8, "ecolor": TEXT_SECONDARY})
        for i, (bar, rep) in enumerate(zip(bars, REPRESENTATIONS)):
            label = f"{d.loc[rep, 'TPR@FPR1pct_mean']:.3f}"
            if rep != "flow":
                p = t[(t["source"] == source) & (t["representation"] == rep)]["wilcoxon_p"]
                label += f"\n{stars(p.iloc[0]) if len(p) else ''}"
            top = d.loc[rep, "TPR@FPR1pct_mean"] + d.loc[rep, "TPR@FPR1pct_std"]
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.015, label,
                    ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)
        ax.set_xticks(x, ["flow", "+ neigh.", "+ host", "+ neigh.\n+ host"])
        ax.set_ylim(0.4, 1.08)
        ax.set_title(source, fontsize=9, loc="left")
    axes[0].set_ylabel("Detection rate @ 1% FPR")
    fig.text(0.01, -0.04, "Mean ± std over 10 seeds, attack-episode split. "
             "Stars: paired Wilcoxon vs flow (* p<0.05, ** p<0.01).",
             fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, "fig3_context_features")


# ----------------------------------------------------------------------
# Figure 4 - per-family change in detection rate
# ----------------------------------------------------------------------

def figure_family_delta():
    r = pd.read_csv(RESULTS / "part4_graph_raw.csv")
    rows = []
    for source in MAIN_SOURCES:
        g = r[r["source"] == source]
        cols = [c for c in g.columns if c.startswith("TPR@FPR1pct_") and g[c].notna().any()]
        means = g.groupby("representation")[cols].mean()
        for c in cols:
            fam = c.replace("TPR@FPR1pct_", "")
            short = "CIC" if source == "CSE-CIC-IDS2018" else "ToN"
            row = {"label": f"{short}: {clean(fam)}"}
            for rep in REPRESENTATIONS[1:]:
                row[rep] = means.loc[rep, c] - means.loc["flow", c]
            rows.append(row)
    d = pd.DataFrame(rows).set_index("label")

    fig, ax = plt.subplots(figsize=(4.6, 0.24 * len(d) + 1.0))
    limit = np.nanmax(np.abs(d.to_numpy()))
    im = ax.imshow(d.to_numpy(), cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(3), ["+ neigh.", "+ host", "+ both"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)), d.index, fontsize=7)
    ax.grid(False)
    for i in range(len(d)):
        for j in range(3):
            v = d.iloc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Change in detection @ 1% FPR vs flow", fontsize=7)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "fig4_family_delta")


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def write_table(df, name, caption, label):
    TABLES.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=True, float_format="%.3f",
                        caption=caption, label=label, position="t")
    latex = latex.replace("\\begin{table}[t]", "\\begin{table}[t]\n\\centering\\small")
    (TABLES / f"{name}.tex").write_text(latex)
    df.to_csv(TABLES / f"{name}.csv", index=False)


def tables():
    src = pd.read_csv(RESULTS / "label_ambiguity_by_source.csv")
    write_table(
        src.rename(columns={
            "source": "Source", "flows": "Flows", "attack_share": "Attack share",
            "distinct_vectors": "Distinct vectors", "irreducible_error": "Irreducible error",
            "attack_flows_in_ambiguous_vectors": "Attack flows in ambiguous vectors"}),
        "table1_sources", "BigFlow-NIDS-V2 by source dataset (all 46M flows).", "tab:sources")

    p1 = pd.read_csv(RESULTS / "part1_shortcuts_raw.csv")
    sid = p1[p1["task"] == "identify source from behaviour"]["balanced_accuracy"]
    pooled = p1[p1["task"] == "pooled attack detection"].groupby("Model")[["F1", "PR_AUC", "ROC_AUC"]].mean()
    pooled = pooled.reset_index().rename(columns={"Model": "Input"})
    pooled.loc[len(pooled)] = [f"Source identification (balanced acc. {sid.mean():.3f})",
                               np.nan, np.nan, np.nan]
    write_table(pooled, "table2_shortcuts",
                "Shortcut features on pooled BigFlow (random split, 3 seeds).", "tab:shortcuts")

    s3 = pd.read_csv(RESULTS / "part3_splits_summary.csv")
    s3 = s3[s3["representation"] == "behaviour"][[
        "source", "split", "TPR@FPR1pct_mean", "TPR@FPR1pct_std", "PR_AUC_w_mean",
        "F1_w_mean", "dup_overlap_mean"]]
    write_table(s3, "table3_splits",
                "Effect of the split strategy (behaviour features, HistGB, 5 seeds).", "tab:splits")

    s4 = pd.read_csv(RESULTS / "part4_graph_summary.csv")
    t4 = pd.read_csv(RESULTS / "part4_graph_significance.csv")
    t4 = t4[t4["metric"] == "TPR@FPR1pct"][["source", "representation", "wilcoxon_p", "wins"]]
    s4 = s4[["source", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
             "PR_AUC_w_mean", "F1_w_mean"]].merge(t4, on=["source", "representation"], how="left")
    write_table(s4, "table4_context_features",
                "Context features under the attack-episode split (HistGB, 10 seeds; "
                "Wilcoxon vs flow).", "tab:context")

    s5 = pd.read_csv(RESULTS / "part5_two_branch_summary.csv")
    write_table(s5[["source", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                    "PR_AUC_w_mean", "F1_w_mean"]],
                "table5_two_branch", "Family-gated detector vs baselines (10 seeds).", "tab:gated")


# ----------------------------------------------------------------------
# Figure 5 - context features across model families
# ----------------------------------------------------------------------

def context_by_model():
    """Flow vs flow + neighbourhood + host for HistGB (part 4), RF and MLP (robustness A2)."""
    hgb = pd.read_csv(RESULTS / "part4_graph_summary.csv").assign(model="HistGB")
    hgb_sig = pd.read_csv(RESULTS / "part4_graph_significance.csv").assign(model="HistGB")
    other = pd.read_csv(RESULTS / "robustA2_features_models_summary.csv")
    other_sig = pd.read_csv(RESULTS / "robustA2_features_models_significance.csv")
    summary = pd.concat([hgb, other], ignore_index=True)
    significance = pd.concat([hgb_sig, other_sig], ignore_index=True)
    return summary, significance[significance["metric"] == "TPR@FPR1pct"]


def figure_models():
    summary, significance = context_by_model()
    models = ["HistGB", "RandomForest", "MLP"]
    reps = ["flow", "flow + neighbourhood + host"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    width = 0.36
    for ax, source in zip(axes, MAIN_SOURCES):
        x = np.arange(len(models))
        for i, rep in enumerate(reps):
            d = summary[(summary["source"] == source) & (summary["representation"] == rep)]
            d = d.set_index("model").reindex(models)
            bars = ax.bar(x + (i - 0.5) * width, d["TPR@FPR1pct_mean"], width - 0.03,
                          yerr=d["TPR@FPR1pct_std"], color=SERIES[i],
                          label="flow" if i == 0 else "flow + neighbourhood + host",
                          error_kw={"lw": 0.8, "ecolor": TEXT_SECONDARY})
            for j, (bar, model) in enumerate(zip(bars, models)):
                label = f"{d.loc[model, 'TPR@FPR1pct_mean']:.2f}".lstrip("0")
                if i == 1:
                    p = significance[(significance["source"] == source)
                                     & (significance["model"] == model)
                                     & (significance["representation"] == rep)]["wilcoxon_p"]
                    label += f"\n{stars(p.iloc[0]) if len(p) else ''}"
                top = d.loc[model, "TPR@FPR1pct_mean"] + d.loc[model, "TPR@FPR1pct_std"]
                ax.text(bar.get_x() + bar.get_width() / 2, top + 0.015, label,
                        ha="center", va="bottom", fontsize=6.5, color=TEXT_SECONDARY)
        ax.set_xticks(x, ["Gradient\nboosting", "Random\nforest", "MLP"])
        ax.set_ylim(0.3, 1.12)
        ax.set_title(source, fontsize=9, loc="left")
    axes[0].set_ylabel("Detection rate @ 1% FPR")
    axes[0].legend(loc="lower left", fontsize=7, ncols=2, bbox_to_anchor=(0, 1.08))
    fig.text(0.01, -0.04, "Mean ± std over 10 seeds, attack-episode split. "
             "Stars: paired Wilcoxon vs flow (* p<0.05, ** p<0.01).",
             fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, "fig5_context_by_model")


def robustness_tables():
    # Split inflation per model family (ToN-IoT and CIC-IDS2018).
    hgb = pd.read_csv(RESULTS / "part3_splits_summary.csv")
    hgb = hgb[hgb["representation"] == "behaviour"].assign(model="HistGB")
    a1 = pd.read_csv(RESULTS / "robustA1_splits_models_summary.csv")
    parts = [hgb, a1]
    eg_path = RESULTS / "egraphsage_summary.csv"
    if eg_path.exists():
        eg = pd.read_csv(eg_path)
        parts.append(eg[eg["model"].str.startswith("E-GraphSAGE")])
    splits = pd.concat(parts, ignore_index=True)
    splits = splits[splits["source"].isin(MAIN_SOURCES)
                    & splits["split"].isin(["random rows", "attack episodes"])]
    wide = splits.pivot_table(index=["source", "model"], columns="split", values="F1_w_mean")
    wide["inflation"] = wide["random rows"] - wide["attack episodes"]
    write_table(wide.reset_index(), "table6_split_inflation_by_model",
                "Weighted F1 under random vs attack-episode splits, per model family.",
                "tab:split-models")

    summary, significance = context_by_model()
    context = summary[summary["representation"].isin(REPRESENTATIONS)][[
        "source", "model", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
        "PR_AUC_w_mean"]].merge(
        significance[["source", "model", "representation", "wilcoxon_p", "wins"]],
        on=["source", "model", "representation"], how="left")
    write_table(context, "table7_context_by_model",
                "Context features across model families (attack-episode split, 10 seeds).",
                "tab:context-models")

    if eg_path.exists():
        eg = pd.read_csv(eg_path)
        sig = pd.read_csv(RESULTS / "egraphsage_significance.csv")
        sig = sig[sig["metric"] == "TPR@FPR1pct"][["source", "split", "model", "wilcoxon_p", "wins", "seeds"]]
        eg = eg[["source", "split", "model", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                 "PR_AUC_w_mean", "F1_w_mean"]].merge(sig, on=["source", "split", "model"], how="left")
        write_table(eg, "table8_egraphsage",
                    "E-GraphSAGE with three endpoint definitions vs a flow-only HistGB "
                    "(Wilcoxon vs HistGB).", "tab:egraphsage")

    gap_path = RESULTS / "robustB_gap_summary.csv"
    if gap_path.exists():
        gap = pd.read_csv(gap_path)
        sig = pd.read_csv(RESULTS / "robustB_gap_significance.csv")
        sig = sig[sig["metric"] == "TPR@FPR1pct"][["source", "gap_min", "representation", "wilcoxon_p", "wins", "seeds"]]
        gap = gap[["source", "gap_min", "representation", "TPR@FPR1pct_mean", "TPR@FPR1pct_std",
                   "PR_AUC_w_mean", "F1_w_mean"]].merge(
            sig, on=["source", "gap_min", "representation"], how="left")
        write_table(gap, "table9_gap_sensitivity",
                    "Sensitivity to the episode gap (HistGB; Wilcoxon vs flow).", "tab:gap")


# ----------------------------------------------------------------------
# Figure 0 - evaluation protocol overview
# ----------------------------------------------------------------------

def figure_protocol():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    stages = [
        ("Data", "BigFlow-NIDS\n66.9M flows\n4 sources"),
        ("Sources", "time period\n+ IP ranges"),
        ("Features", "no timestamps,\nIPs, source port"),
        ("Split", "random rows\n10-min blocks\nattack episodes"),
        ("Models", "GB, RF, MLP,\nE-GraphSAGE\n+ context"),
        ("Metrics", "TPR @ 1% FPR\nweighted PR-AUC\nWilcoxon tests"),
    ]
    audits = ["Finding 1:\nshortcuts", "Finding 2:\nlabel ambiguity",
              "Finding 3:\nsplit leakage", "Finding 4:\ncontext value"]
    audit_at = [1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.grid(False)

    w, h, y = 0.86, 0.95, 1.35
    for i, (title, body) in enumerate(stages):
        x = i + (1 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#eef4fc", ec=SERIES[0], lw=1.0))
        ax.text(i + 0.5, y + h - 0.17, title, ha="center", va="center", fontsize=8,
                fontweight="bold", color=TEXT)
        ax.text(i + 0.5, y + 0.38, body, ha="center", va="center", fontsize=6.4,
                color=TEXT_SECONDARY, linespacing=1.25)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((i + 0.5 + w / 2, y + h / 2), (i + 1.5 - w / 2, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=8, lw=1.0,
                                         color=TEXT_SECONDARY))

    for label, i in zip(audits, audit_at):
        ax.add_patch(FancyArrowPatch((i + 0.5, y - 0.03), (i + 0.5, 0.78), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=SERIES[1], linestyle="--"))
        ax.add_patch(FancyBboxPatch((i + 0.12, 0.12), 0.76, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#fdf0ea", ec=SERIES[1], lw=0.9))
        ax.text(i + 0.5, 0.43, label, ha="center", va="center", fontsize=6.6, color=TEXT,
                linespacing=1.25)

    save(fig, "fig0_protocol")


# ----------------------------------------------------------------------
# Figure 6 - detailed workflow pipeline
# ----------------------------------------------------------------------

def figure_workflow():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.axis("off")
    ax.grid(False)

    stage_fill, stage_edge = "#eef4fc", SERIES[0]
    find_fill, find_edge = "#fdf0ea", SERIES[1]
    lane_fill = "#f3f3f1"

    def box(x, y, w, h, title, body="", fill=stage_fill, edge=stage_edge, title_size=7.2,
            body_size=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=0.9))
        if body:
            ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=title_size,
                    fontweight="bold", color=TEXT)
            ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=body_size,
                    color=TEXT_SECONDARY, linespacing=1.3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                    fontweight="bold", color=TEXT)

    def arrow(x0, y0, x1, y1, color=TEXT_SECONDARY, dashed=False):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.9, color=color, linestyle="--" if dashed else "-",
                                     shrinkA=0, shrinkB=0))

    def finding(y, h, title, body):
        box(84, y, 15.5, h, title, body, fill=find_fill, edge=find_edge, title_size=6.6,
            body_size=5.6)

    lanes = [  # (label, y_bottom, height)
        ("1  Data", 83, 17),
        ("2  Features", 62, 18),
        ("3  Splits", 45, 14),
        ("4  Models", 23, 19),
        ("5  Evaluation", 2, 18),
    ]
    for label, y, h in lanes:
        ax.add_patch(FancyBboxPatch((0.5, y), 81.5, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc=lane_fill, ec="none"))
        ax.text(2.2, y + h / 2, label, rotation=90, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color=TEXT_SECONDARY)

    # Lane 1 - data
    box(6, 85.5, 22, 12.5, "BigFlow-NIDS", "66,935,021 flows\n67 Parquet files\n53 NetFlow-v3 features")
    box(32.5, 85.5, 22, 12.5, "Source recovery", "timestamp period\n+ IP-range check\n4 source datasets")
    box(59, 85.5, 21, 12.5, "Behaviour features", "drop timestamps, IPs,\nsource port, DNS id\n47 features")
    arrow(28.3, 91.7, 32.2, 91.7)
    arrow(54.8, 91.7, 58.7, 91.7)

    # Lane 2 - features
    box(6, 64.5, 17, 13.5, "Working sample", "859,697 flows\ncapped per class\nprevalence weights")
    box(27, 64.5, 15, 13.5, "Flow", "47 behaviour\nfeatures")
    box(45, 64.5, 17, 13.5, "Neighbourhood", "k-NN, k = 20\ntrain labels only\n+ LOF, distances")
    box(65, 64.5, 15, 13.5, "Host", "1/10/60 s windows\nsrc, dst, pair\n78 features")
    arrow(23.3, 71.2, 26.7, 71.2)
    arrow(69.5, 85.2, 14.5, 78.3)

    # Lane 3 - splits
    box(6, 47.5, 22, 9.5, "Random rows", "70 / 15 / 15, stratified")
    box(32.5, 47.5, 22, 9.5, "10-min time blocks", "whole blocks, random")
    box(59, 47.5, 21, 9.5, "Attack episodes", "5-min gap; chronological\nfallback if < 3 episodes")
    arrow(43.5, 64.2, 43.5, 57.3)

    # Lane 4 - models
    box(6, 33.5, 15, 7.5, "Gradient\nboosting", title_size=6.6)
    box(23, 33.5, 13, 7.5, "Random\nforest", title_size=6.6)
    box(38, 33.5, 10, 7.5, "MLP", title_size=6.6)
    box(50, 33.5, 30, 7.5, "E-GraphSAGE\nIP | IP:port | random control", title_size=6.6)
    box(6, 24.2, 74, 7.3, "Family-gated detector",
        "router picks the flow, neighbourhood or host expert per attack family",
        title_size=6.6, body_size=6.0)
    arrow(43.5, 47.2, 43.5, 41.3)

    # Lane 5 - evaluation
    box(6, 4.5, 23, 13, "Operational metrics", "TPR @ 1% and 0.1% FPR\nthreshold from\nvalidation benign")
    box(32, 4.5, 23, 13, "Real prevalence", "weighted PR-AUC\nand F1")
    box(58, 4.5, 22, 13, "Statistics", "10 seeds, paired\nWilcoxon tests")
    arrow(43.5, 23.9, 43.5, 17.8)

    # Findings column
    ax.text(91.75, 101, "Findings", ha="center", va="center", fontsize=7.2, fontweight="bold",
            color=find_edge)
    finding(88, 11, "1 Shortcuts", "source identified\nat 99.4%")
    finding(74, 11, "2 Ambiguity", "56% of ToN-IoT\nattacks = benign")
    finding(47, 10, "3 Split leakage", "only episodes\nremove the shortcut")
    finding(32.5, 10, "4 Context value", "helps CIC-IDS2018,\nnot ToN-IoT")
    finding(20.5, 10, "5 Gated", "matches the best\nfeature set")

    arrow(80.4, 91.7, 83.6, 93.5, color=find_edge, dashed=True)
    arrow(80.4, 89.5, 83.6, 79.5, color=find_edge, dashed=True)
    arrow(80.4, 52.2, 83.6, 52.2, color=find_edge, dashed=True)
    arrow(80.4, 37.2, 83.6, 37.2, color=find_edge, dashed=True)
    arrow(80.4, 26.5, 83.6, 26.5, color=find_edge, dashed=True)

    save(fig, "fig6_workflow")


if __name__ == "__main__":
    figure_protocol()
    figure_workflow()
    figure_ambiguity()
    figure_splits()
    figure_context()
    figure_family_delta()
    figure_models()
    tables()
    robustness_tables()
    print("Figures:", sorted(p.name for p in FIGURES.glob("*.png")))
    print("Tables :", sorted(p.name for p in TABLES.glob("*.tex")))
