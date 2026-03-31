#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


DIRECTIONS_ALL = [
    "smiles_to_iupac",
    "iupac_to_smiles",
    "iupac_to_graph",
    "graph_to_iupac",
    "smiles_to_graph",
    "graph_to_smiles",
]
DIRECTION_IO = {
    "smiles_to_iupac": ("smiles", "iupac"),
    "iupac_to_smiles": ("iupac", "smiles"),
    "iupac_to_graph": ("iupac", "graph"),
    "graph_to_iupac": ("graph", "iupac"),
    "smiles_to_graph": ("smiles", "graph"),
    "graph_to_smiles": ("graph", "smiles"),
}
DIRECTION_LABEL_FULL = {
    "smiles_to_iupac": "SMILES -> IUPAC",
    "iupac_to_smiles": "IUPAC -> SMILES",
    "iupac_to_graph": "IUPAC -> MolJSON",
    "graph_to_iupac": "MolJSON -> IUPAC",
    "smiles_to_graph": "SMILES -> MolJSON",
    "graph_to_smiles": "MolJSON -> SMILES",
}
OUTPUT_COLOR = {"smiles": "#E69F00", "iupac": "#56B4E9", "graph": "#009E73"}
INPUT_LINESTYLE = {"smiles": "--", "iupac": "-", "graph": ":"}
AXIS_LABEL_FS = 21
X_AXIS_LABEL_PAD = 16
X_TICK_LABEL_PAD = 8
Y_AXIS_MIN = 0.0
LEGEND_ORDER = [
    "iupac_to_graph",
    "smiles_to_graph",
    "smiles_to_iupac",
    "graph_to_iupac",
    "iupac_to_smiles",
    "graph_to_smiles",
]
MODEL_FILES = [
    ("gpt-5-nano", "gpt-5-nano-low_checked.csv", "fig-gpt5nano-error-modes-three-panel-abc-a5bins.png"),
    ("gpt-5-mini", "gpt-5-mini-low_checked.csv", "fig-gpt5mini-error-modes-three-panel-abc-a5bins.png"),
    ("gpt-5", "gpt-5-low_checked.csv", "fig-gpt5-error-modes-three-panel-abc-a5bins.png"),
]
HEAVY_BINS_5 = [
    (10, 14, 0, "10-14"),
    (15, 18, 1, "15-18"),
    (19, 22, 2, "19-22"),
    (23, 26, 3, "23-26"),
    (27, 30, 4, "27-30"),
]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _wilson_ci_pct(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1.0 + (z**2 / total)
    center = (p + (z**2) / (2 * total)) / denom
    rad = z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total * total))) / denom
    lo = max(0.0, center - rad)
    hi = min(1.0, center + rad)
    return (100.0 * lo, 100.0 * hi)


def _direction_from_io(inp: str, out: str) -> str | None:
    key = f"{inp}_to_{out}"
    return key if key in DIRECTION_IO else None


def _smiles_to_counts(smiles: str) -> tuple[int | None, int | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return int(mol.GetNumHeavyAtoms()), int(mol.GetRingInfo().NumRings())


def _load_checked(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need = {
        "uuid",
        "category",
        "dataset",
        "input_format",
        "output_format",
        "is_correct",
        "smiles",
        "model",
        "effort",
    }
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required columns {sorted(missing)}")

    x = df[(df["category"] == "translation") & (df["dataset"] == "translation_large")].copy()
    x = x[x["input_format"] != x["output_format"]].copy()
    x["direction"] = [_direction_from_io(i, o) for i, o in zip(x["input_format"], x["output_format"])]
    x = x[x["direction"].notna()].copy()
    x["is_correct_bool"] = x["is_correct"].map(_to_bool)

    counts = x["smiles"].map(_smiles_to_counts)
    x["gt_heavy_count"] = [c[0] for c in counts]
    x["gt_ring_count"] = [c[1] for c in counts]
    x["gt_smiles"] = x["smiles"]
    x = x.dropna(subset=["gt_heavy_count", "gt_ring_count", "gt_smiles"]).copy()
    x["gt_heavy_count"] = x["gt_heavy_count"].astype(int)
    x["gt_ring_count"] = x["gt_ring_count"].astype(int)
    return x


def _classify_fused_spiro(smiles: str) -> bool | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bond_rings = [set(r) for r in mol.GetRingInfo().BondRings()]
    has_fused = any(bool(bond_rings[i] & bond_rings[j]) for i in range(len(bond_rings)) for j in range(i + 1, len(bond_rings)))
    has_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0
    return bool(has_fused or has_spiro)


def _build_panel_a(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    d0 = df[df["gt_ring_count"] == 0].copy()
    for direction in DIRECTIONS_ALL:
        sd = d0[d0["direction"] == direction]
        for lo_h, hi_h, idx, label in HEAVY_BINS_5:
            b = sd[(sd["gt_heavy_count"] >= lo_h) & (sd["gt_heavy_count"] <= hi_h)]
            n = int(len(b))
            k = int(b["is_correct_bool"].sum())
            acc = (100.0 * k / n) if n else float("nan")
            lo, hi = _wilson_ci_pct(k, n)
            rows.append(
                {
                    "model": str(df["model"].iloc[0]),
                    "effort": str(df["effort"].iloc[0]),
                    "direction": direction,
                    "heavy_bin_idx": int(idx),
                    "heavy_bin_label": label,
                    "n_questions": n,
                    "n_correct": k,
                    "accuracy_pct": acc,
                    "ci_low_pct": lo,
                    "ci_high_pct": hi,
                }
            )
    return pd.DataFrame(rows)


def _build_panel_b(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for direction in DIRECTIONS_ALL:
        sd = df[df["direction"] == direction]
        for ring in [0, 1, 2, 3]:
            b = sd[sd["gt_ring_count"] == ring]
            n = int(len(b))
            k = int(b["is_correct_bool"].sum())
            acc = (100.0 * k / n) if n else float("nan")
            lo, hi = _wilson_ci_pct(k, n)
            rows.append(
                {
                    "model": str(df["model"].iloc[0]),
                    "effort": str(df["effort"].iloc[0]),
                    "direction": direction,
                    "ring_count": int(ring),
                    "n_questions": n,
                    "n_correct": k,
                    "accuracy_pct": acc,
                    "ci_low_pct": lo,
                    "ci_high_pct": hi,
                }
            )
    return pd.DataFrame(rows)


def _build_panel_c(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["gt_ring_count"].isin([2, 3])].copy()
    cache: dict[str, bool | None] = {}

    def _cached(s: object) -> bool | None:
        key = str(s) if s is not None else ""
        if key in cache:
            return cache[key]
        value = _classify_fused_spiro(key)
        cache[key] = value
        return value

    sub["has_fused_spiro"] = sub["gt_smiles"].map(_cached)
    sub = sub.dropna(subset=["has_fused_spiro"]).copy()
    sub["has_fused_spiro"] = sub["has_fused_spiro"].astype(bool)

    out: list[dict[str, object]] = []
    for direction in DIRECTIONS_ALL:
        sd = sub[sub["direction"] == direction]
        sep = sd[~sd["has_fused_spiro"]]
        fs = sd[sd["has_fused_spiro"]]
        n_sep = int(len(sep))
        k_sep = int(sep["is_correct_bool"].sum())
        n_fs = int(len(fs))
        k_fs = int(fs["is_correct_bool"].sum())
        acc_sep = (100.0 * k_sep / n_sep) if n_sep else float("nan")
        acc_fs = (100.0 * k_fs / n_fs) if n_fs else float("nan")
        sep_lo, sep_hi = _wilson_ci_pct(k_sep, n_sep)
        fs_lo, fs_hi = _wilson_ci_pct(k_fs, n_fs)
        inp, out_fmt = DIRECTION_IO[direction]
        out.append(
            {
                "model": str(df["model"].iloc[0]),
                "effort": str(df["effort"].iloc[0]),
                "direction": direction,
                "direction_label_full": DIRECTION_LABEL_FULL[direction],
                "direction_label_short": direction.replace("_to_", "->"),
                "input_format": inp,
                "output_format": out_fmt,
                "n_sep": n_sep,
                "k_sep": k_sep,
                "acc_sep_pct": acc_sep,
                "ci_sep_low_pct": sep_lo,
                "ci_sep_high_pct": sep_hi,
                "n_fused_spiro": n_fs,
                "k_fused_spiro": k_fs,
                "acc_fused_spiro_pct": acc_fs,
                "ci_fs_low_pct": fs_lo,
                "ci_fs_high_pct": fs_hi,
                "penalty_pp": (acc_sep - acc_fs) if (not math.isnan(acc_sep) and not math.isnan(acc_fs)) else float("nan"),
            }
        )
    return pd.DataFrame(out)


def _plot_line_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    x_order: list[int],
    x_labels: list[str],
    panel_label: str,
    subset_note: str,
    x_label: str,
) -> None:
    x = np.arange(len(x_order), dtype=float)
    xmap = {v: i for i, v in enumerate(x_order)}
    for direction in DIRECTIONS_ALL:
        s = df[df["direction"] == direction].copy()
        if s.empty:
            continue
        s["_x"] = s[x_col].map(xmap)
        s = s.dropna(subset=["_x"]).sort_values("_x")
        xx = s["_x"].to_numpy(dtype=float)
        y = s["accuracy_pct"].to_numpy(dtype=float)
        lo = s["ci_low_pct"].to_numpy(dtype=float)
        hi = s["ci_high_pct"].to_numpy(dtype=float)
        inp, out = DIRECTION_IO[direction]
        ax.plot(xx, y, color=OUTPUT_COLOR[out], linestyle=INPUT_LINESTYLE[inp], linewidth=2.2, zorder=3)
        ax.fill_between(xx, lo, hi, color=OUTPUT_COLOR[out], alpha=0.16, linewidth=0.0, zorder=2)

    ax.set_box_aspect(1.0)
    ax.set_title(panel_label, loc="left", fontsize=AXIS_LABEL_FS, pad=10)
    ax.text(
        0.02,
        0.02,
        subset_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.9},
    )
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FS, labelpad=X_AXIS_LABEL_PAD)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FS)
    ax.set_ylim(Y_AXIS_MIN, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=21)
    ax.tick_params(axis="y", labelsize=21)
    ax.tick_params(axis="x", pad=X_TICK_LABEL_PAD)
    ax.grid(alpha=0.25, linewidth=0.7, zorder=0)


def _plot_slope_panel(ax: plt.Axes, df: pd.DataFrame, panel_label: str) -> None:
    d = df.set_index("direction").reindex(LEGEND_ORDER).dropna(subset=["acc_sep_pct", "acc_fused_spiro_pct"]).reset_index()
    x_sep = -0.30
    x_fs = 1.30
    offsets = [-0.04, -0.024, -0.008, 0.008, 0.024, 0.04]
    for i, row in d.iterrows():
        direction = str(row["direction"])
        inp, out = DIRECTION_IO[direction]
        color = OUTPUT_COLOR[out]
        ls = INPUT_LINESTYLE[inp]
        y_sep = float(row["acc_sep_pct"])
        y_fs = float(row["acc_fused_spiro_pct"])
        sep_lo = float(row["ci_sep_low_pct"])
        sep_hi = float(row["ci_sep_high_pct"])
        fs_lo = float(row["ci_fs_low_pct"])
        fs_hi = float(row["ci_fs_high_pct"])
        j = offsets[i]
        xs = np.array([x_sep + j, x_fs + j], dtype=float)
        ys = np.array([y_sep, y_fs], dtype=float)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.5, zorder=3)
        sep_yerr = np.maximum(0.0, np.array([[y_sep - sep_lo], [sep_hi - y_sep]], dtype=float))
        fs_yerr = np.maximum(0.0, np.array([[y_fs - fs_lo], [fs_hi - y_fs]], dtype=float))
        ax.errorbar([x_sep + j], [y_sep], yerr=sep_yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=2.4, zorder=2)
        ax.errorbar([x_fs + j], [y_fs], yerr=fs_yerr, fmt="none", ecolor=color, elinewidth=1.0, capsize=2.4, zorder=2)

    ax.set_box_aspect(1.0)
    ax.set_title(panel_label, loc="left", fontsize=AXIS_LABEL_FS, pad=10)
    ax.text(
        0.02,
        0.02,
        "2 or 3 rings",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.9},
    )
    ax.set_xlabel("Ring Topology", fontsize=AXIS_LABEL_FS, labelpad=X_AXIS_LABEL_PAD)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FS)
    ax.set_xticks([x_sep, x_fs])
    ax.set_xticklabels(["non-fused", "fused/spiro"], fontsize=21)
    ax.set_xlim(-0.70, 1.70)
    ax.set_ylim(Y_AXIS_MIN, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=21)
    ax.tick_params(axis="x", pad=X_TICK_LABEL_PAD)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7, zorder=0)


def _legend_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for d in LEGEND_ORDER:
        inp, out = DIRECTION_IO[d]
        handles.append(Line2D([0], [0], color=OUTPUT_COLOR[out], linestyle=INPUT_LINESTYLE[inp], linewidth=2.2))
        labels.append(DIRECTION_LABEL_FULL[d])
    return handles, labels


def _plot_model_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, out_fig: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 8.4), dpi=260, sharey=True)
    heavy_bins = panel_a[["heavy_bin_idx", "heavy_bin_label"]].drop_duplicates().sort_values("heavy_bin_idx")
    _plot_line_panel(
        axes[0],
        panel_a,
        x_col="heavy_bin_idx",
        x_order=heavy_bins["heavy_bin_idx"].astype(int).tolist(),
        x_labels=heavy_bins["heavy_bin_label"].astype(str).tolist(),
        panel_label="(a)",
        subset_note="0 rings only",
        x_label="Number of Heavy Atoms",
    )
    _plot_line_panel(
        axes[1],
        panel_b,
        x_col="ring_count",
        x_order=[0, 1, 2, 3],
        x_labels=["0", "1", "2", "3"],
        panel_label="(b)",
        subset_note="All molecules",
        x_label="Number of Rings",
    )
    _plot_slope_panel(axes[2], panel_c, "(c)")

    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    handles, labels = _legend_handles()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=20,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        handlelength=3.0,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.005, 0.16, 1.0, 0.98])
    fig.subplots_adjust(wspace=0.10)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(description="Generate 3x error-mode panel figures (nano/mini/full) + summary tables.")
    parser.add_argument(
        "--checked-dir",
        default=str(package_root / "model_responses" / "checked"),
    )
    parser.add_argument(
        "--out-tables-dir",
        default=str(package_root / "analysis_outputs" / "tables" / "error_mode_panel"),
    )
    parser.add_argument(
        "--out-figs-dir",
        default=str(package_root / "analysis_outputs" / "plots"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked_dir = Path(args.checked_dir)
    out_tables = Path(args.out_tables_dir)
    out_figs = Path(args.out_figs_dir)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    panel_a_all: list[pd.DataFrame] = []
    panel_b_all: list[pd.DataFrame] = []
    panel_c_all: list[pd.DataFrame] = []

    for model_key, filename, out_fig_name in MODEL_FILES:
        path = checked_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing checked file: {path}")
        rows = _load_checked(path)
        if rows.empty:
            raise ValueError(f"No rows left for {path.name}")
        pa = _build_panel_a(rows)
        pb = _build_panel_b(rows)
        pc = _build_panel_c(rows)
        panel_a_all.append(pa)
        panel_b_all.append(pb)
        panel_c_all.append(pc)
        _plot_model_figure(pa, pb, pc, out_figs / out_fig_name)
        print(f"Wrote {out_figs / out_fig_name}")

    panel_a_df = pd.concat(panel_a_all, ignore_index=True)
    panel_b_df = pd.concat(panel_b_all, ignore_index=True)
    panel_c_df = pd.concat(panel_c_all, ignore_index=True)

    p1 = out_tables / "panel_a_zero_ring_heavy_bins5.csv"
    p2 = out_tables / "panel_b_all_molecules_ringcount.csv"
    p3 = out_tables / "panel_c_topology_2or3rings_nonfused_vs_fusedspiro.csv"
    panel_a_df.sort_values(["model", "direction", "heavy_bin_idx"]).to_csv(p1, index=False)
    panel_b_df.sort_values(["model", "direction", "ring_count"]).to_csv(p2, index=False)
    panel_c_df.sort_values(["model", "direction"]).to_csv(p3, index=False)

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
