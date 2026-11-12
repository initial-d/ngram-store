"""Figures for the NGramStore technical report.

Reads numbers from evidence/ (paths below); values only published in a README
are listed inline with their source. Run from the repository root:

    python docs/report/make_figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Validated categorical slots 1-3 (dataviz reference palette, all-pairs pass)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
W = 6.5  # text width, inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- Figure 2
def fig_tradeoff():
    """Table saving vs lookup slowdown, local AG News reproduction."""
    zero = load(EV / "host_query_profile_20261112/results/zero-init.json")
    uni = load(EV / "host_query_profile_20261112/results/uniform-init.json")

    def points(rep, layouts):
        src, t = rep["source_fp32_bytes"], rep["timings"]
        raw = t["raw-mmap per-doc"]["p50_us"]
        return [(t[f"{k} direct per-doc"]["p50_us"] / raw, 100 * (1 - rep["artifact_bytes"][k] / src), lab)
                for k, lab in layouts]

    zp = points(zero, [("dedup-raw", "dedup"), ("dedup-zstd-4", "dedup+Zstd, 4-row"),
                       ("dedup-zstd-16", "dedup+Zstd, 16-row"), ("dedup-zstd-256", "dedup+Zstd, 256-row"),
                       ("generic-zstd-256", "Zstd, 256-row")])
    up = points(uni, [("dedup-raw", "dedup"), ("generic-zstd-256", "Zstd, 256-row")])

    fig, ax = plt.subplots(figsize=(W * 0.52, 2.35))
    ax.axvspan(0.8, 2, color="#f1f0ec", zorder=0, lw=0)
    ax.text(1.27, 45, "near raw speed", color=MUTED, fontsize=6.3, rotation=90, ha="center", va="center")
    ax.scatter([1], [0], s=26, color=MUTED, zorder=3, edgecolor="white", lw=0.8)
    ax.annotate("raw FP32", (1, 0), (0, 7), textcoords="offset points", color=INK2, fontsize=6.3, ha="left")
    for pts, color, name in ((zp, BLUE, "zero init"), (up, ORANGE, "uniform init")):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.scatter(xs, ys, s=30, color=color, zorder=4, edgecolor="white", lw=0.8, label=name)
    short = {"dedup": "dedup, uncompressed", "dedup+Zstd, 4-row": "4", "dedup+Zstd, 16-row": "16",
             "dedup+Zstd, 256-row": "256", "Zstd, 256-row": "Zstd only"}
    for x, y, lab in zp:
        if lab == "dedup":
            ax.annotate(short[lab], (x, y), (7, -2), textcoords="offset points", fontsize=6.3, color=INK2, va="center")
        else:
            ax.annotate(short[lab], (x, y), (0, 6), textcoords="offset points", fontsize=6.3, color=INK2, ha="center")
    ax.annotate("dedup + Zstd, rows per block", (60, 86.2), (0, 16), textcoords="offset points",
                fontsize=6.3, color=INK2, ha="center")
    for x, y, lab in up:
        if lab == "dedup":
            ax.annotate("dedup, uncompressed", (x, y), (7, -6), textcoords="offset points", fontsize=6.3, color=INK2)
        else:
            ax.annotate("Zstd only", (x, y), (0, 6), textcoords="offset points", fontsize=6.3, color=INK2, ha="center")
    ax.set_xscale("log"); ax.set_xlim(0.8, 700); ax.set_ylim(-14, 108)
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1×", "10×", "100×"])
    ax.set_xlabel("Lookup p50 relative to raw mmap (log)")
    ax.set_ylabel("Table size saving (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.55, 0.12), handletextpad=0.2)
    ax.grid(axis="x", visible=False)
    save(fig, "tradeoff")


# ---------------------------------------------------------------- Figure 3
def fig_identity():
    ident = load(EV / "ngram_identity_evidence_20261107/final-analysis.json")["exact_vs_control"]
    get = lambda m, e, c: next(r for r in ident if r["model"] == m and r["edits"] == e and r["control"] == c)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38})

    # (a) 20,877 edits: full-key branch vs independent controls.
    # Head constants and parameter-matched adapter: evidence/memory_application_evidence_20261110/README.md
    controls = ["Head constants", "Matched adapter", "Hash, native OOV", "Hash, matched coverage"]
    readme = {("qwen06", "Head constants"): (22.94, 22.57, 23.29), ("qwen35", "Head constants"): (26.63, 26.27, 27.01),
              ("qwen06", "Matched adapter"): (7.02, 6.64, 7.42), ("qwen35", "Matched adapter"): (5.39, 4.90, 5.92)}
    ctl_key = {"Hash, native OOV": "hash-native-oov", "Hash, matched coverage": "hash"}
    y = np.arange(len(controls))[::-1]
    for model, color, dy, name in (("qwen06", BLUE, 0.13, "Qwen3-0.6B"), ("qwen35", ORANGE, -0.13, "Qwen3.5-4B")):
        vals = []
        for c in controls:
            if c in ctl_key:
                r = get(model, 20877, ctl_key[c]); lo, hi = r["harmonic_delta_interval95"]
                vals.append((100 * r["harmonic_delta"], 100 * lo, 100 * hi))
            else:
                vals.append(readme[(model, c)])
        v = np.array(vals)
        a.errorbar(v[:, 0], y + dy, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=4.2, color=color,
                   ecolor=color, elinewidth=1.2, capsize=0, mec="white", mew=0.6, label=name)
    a.axvline(0, color=INK2, lw=0.6)
    a.set_yticks(y); a.set_yticklabels(controls); a.grid(axis="y", visible=False)
    a.set_xscale("symlog", linthresh=2); a.set_xlim(-0.5, 40)
    a.set_xticks([0, 1, 2, 5, 10, 20]); a.set_xticklabels(["0", "1", "2", "5", "10", "20"])
    a.set_xlabel("Advantage of distinct rows, H (pp)")
    a.set_title("a  Knowledge editing, 20,877 edits", fontsize=8)
    a.legend(loc="lower right", handletextpad=0.1)

    # (b) capacity pressure: delta vs hash at 1,024 and 20,877 edits.
    xs = np.array([0, 1])
    for model, color, name in (("qwen06", BLUE, "Qwen3-0.6B"), ("qwen35", ORANGE, "Qwen3.5-4B")):
        for ctl, marker, ls, off in (("hash", "o", "-", -0.04), ("hash-native-oov", "s", (0, (1, 0)), 0.04)):
            rs = [get(model, e, ctl) for e in (1024, 20877)]
            m = [100 * r["harmonic_delta"] for r in rs]
            lo = [100 * r["harmonic_delta_interval95"][0] for r in rs]
            hi = [100 * r["harmonic_delta_interval95"][1] for r in rs]
            b.errorbar(xs + off, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], fmt=marker + "-", ms=3.8,
                       color=color, lw=1.0, elinewidth=1.0, capsize=0, mec="white", mew=0.6,
                       alpha=1.0 if ctl == "hash" else 0.55)
    b.axhline(0, color=INK2, lw=0.6)
    b.set_xticks(xs); b.set_xticklabels(["1,024", "20,877"]); b.set_xlim(-0.3, 1.45)
    b.set_xlabel("Edits stored in the memory"); b.set_ylabel("H vs hash buckets (pp)")
    b.grid(axis="x", visible=False)
    b.set_title("b  More facts, larger gap", fontsize=8)
    b.text(1.11, 4.05, "4B, native OOV", fontsize=6.2, color=INK2, va="center")
    b.text(1.11, 2.25, "0.6B, native OOV", fontsize=6.2, color=INK2, va="center")
    b.text(1.11, 1.30, "4B, matched", fontsize=6.2, color=INK2, va="center")
    b.text(1.11, 0.40, "0.6B, matched", fontsize=6.2, color=INK2, va="center")
    save(fig, "identity")


# ---------------------------------------------------------------- Figure 4
def fig_redundancy():
    plans = {Path(p).stem: load(p) for p in glob.glob(str(EV / "host_query_profile_20261112/original-tables/plan/*.json"))}
    struct = load(EV / "structure_dedup_20261113/results.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.32})

    groups = [("AG News", "ag_news"), ("DBpedia", "dbpedia_14")]
    x = np.arange(len(groups)); w = 0.34
    for i, (init, color) in enumerate((("zero", BLUE), ("uniform", ORANGE))):
        vals = [100 * np.mean([p["duplicate_fraction"] for k, p in plans.items() if k.startswith(ds) and f"-{init}-" in k])
                for _, ds in groups]
        bars = a.bar(x + (i - 0.5) * (w + 0.03), vals, width=w, color=color, label=f"{init} init", zorder=3)
        for xb, v in zip(x + (i - 0.5) * (w + 0.03), vals):
            a.text(xb, v + 2, f"{v:.1f}%", ha="center", fontsize=6.5, color=INK2)
    a.set_xticks(x); a.set_xticklabels([g[0] for g in groups]); a.set_ylim(0, 112)
    a.set_ylabel("Rows that are byte duplicates (%)"); a.grid(axis="x", visible=False)
    a.set_title("a  Classifier tables (7.0M / 34.1M rows)", fontsize=8)
    a.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    steps = [2000, 8000, 32000]
    per_seed = {}
    for r in struct:
        seed = int(r["point"].split("seed-")[1].split("-")[0]); step = int(r["point"].rsplit("-", 1)[1])
        per_seed.setdefault(seed, {})[step] = 100 * (1 - r["unique_rows_joint"] / r["rows"])
    for seed, d in sorted(per_seed.items()):
        b.plot(steps, [d[s] for s in steps], "-", color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    med = [np.median([d[s] for d in per_seed.values()]) for s in steps]
    b.plot(steps, med, "o-", color=BLUE, lw=1.6, ms=4.5, mec="white", mew=0.7, zorder=3)
    # Update-to-initialization norm ratio (median, 3 seeds): tree-node study README.
    for s, m in zip(steps, med):
        b.annotate(f"{m:.0f}%", (s, m), (-4, 6), textcoords="offset points", ha="right", fontsize=6.5, color=INK)
    b.set_xscale("log"); b.set_xticks(steps); b.set_xticklabels(["2k", "8k", "32k"]); b.minorticks_off()
    b.set_xlim(1500, 45000); b.set_ylim(0, 55)
    b.set_xlabel("SGD steps"); b.set_ylabel("Rows merged into a prefix/suffix\nrepresentative (%)")
    b.grid(axis="x", visible=False)
    b.set_title("b  Character n-grams, order 12, cosine ≥ .99", fontsize=8)
    save(fig, "redundancy")


# ---------------------------------------------------------------- Figure 5
def fig_decode():
    zero = load(EV / "host_query_profile_20261112/results/zero-init.json")
    t = zero["timings"]; ids = zero["ids_per_doc_mean"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.34})

    layouts = [("dedup-zstd-256", 256), ("dedup-zstd-16", 16), ("dedup-zstd-4", 4)]
    kib = [r * 512 / 1024 for _, r in layouts]
    p50 = [t[f"{k} direct per-doc"]["p50_us"] for k, _ in layouts]
    a.plot(kib, p50, "o-", color=BLUE, lw=1.4, ms=4.5, mec="white", mew=0.7, zorder=3, label="dedup + Zstd")
    for k, r, y in zip(kib, [r for _, r in layouts], p50):
        a.annotate(f"{r} rows", (k, y), (4, -10) if r == 4 else (-5, 4), textcoords="offset points", fontsize=6.2,
                   color=INK2, ha="left" if r == 4 else "right")
    unc, raw = t["dedup-raw direct per-doc"]["p50_us"], t["raw-mmap per-doc"]["p50_us"]
    a.axhline(unc, color=AQUA, lw=1.4, zorder=2); a.axhline(raw, color=MUTED, lw=1.0, zorder=2)
    a.text(1.08, unc * 1.22, f"dedup, uncompressed: {unc:.0f} µs", fontsize=6.3, color=INK2)
    a.text(1.08, raw * 0.66, f"raw mmap: {raw:.0f} µs", fontsize=6.3, color=INK2)
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xticks([2, 8, 32, 128]); a.set_xticklabels(["2", "8", "32", "128"]); a.minorticks_off()
    a.set_xlim(1, 200); a.set_ylim(6, 12000)
    a.set_yticks([10, 100, 1000, 10000]); a.set_yticklabels(["10", "100", "1,000", "10,000"])
    a.set_xlabel("Decoded block size (KiB)"); a.set_ylabel("Lookup p50 per document (µs)")
    a.set_title(f"a  One document = {ids:.0f} scattered rows", fontsize=8)
    a.grid(axis="x", visible=False)

    gpu = {Path(p).stem: load(p) for p in glob.glob(str(EV / "host_query_profile_20261112/original-tables/gpu/*.json"))}
    order = [("ag_news-exact-zero-s3-lr10", "AG News\nzero"), ("ag_news-exact-uniform-s3-lr10", "AG News\nuniform"),
             ("dbpedia_14-exact-zero-s3-lr10", "DBpedia\nzero"), ("dbpedia_14-exact-uniform-s3-lr3", "DBpedia\nuniform")]
    x = np.arange(len(order)); w = 0.34
    b.axhspan(29, 79, color="#f1f0ec", lw=0, zorder=0)
    b.text(3.55, 48, "compressed layouts\nin the deployment grid\n(29–79×)", fontsize=6.3, color=INK2, ha="right", va="center")
    for i, (bs, color) in enumerate((("b1", BLUE), ("b32", ORANGE))):
        vals = [gpu[m]["results"][f"dedup-raw gather_array {bs}"]["x_raw_p50"] for m, _ in order]
        b.bar(x + (i - 0.5) * (w + 0.03), vals, width=w, color=color, zorder=3,
              label="batch 1" if bs == "b1" else "batch 32")
    b.set_yscale("log"); b.set_ylim(0.8, 120); b.set_yticks([1, 2, 5, 10, 20, 50, 100])
    b.set_yticklabels(["1×", "2×", "5×", "10×", "20×", "50×", "100×"]); b.minorticks_off()
    b.set_xticks(x); b.set_xticklabels([o[1] for o in order], fontsize=6.6)
    b.set_ylabel("End-to-end p50 vs raw (log)"); b.grid(axis="x", visible=False)
    b.set_title("b  Dedup, uncompressed, A800 host path", fontsize=8)
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1)
    save(fig, "decode")


if __name__ == "__main__":
    fig_tradeoff(); fig_identity(); fig_redundancy(); fig_decode()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
