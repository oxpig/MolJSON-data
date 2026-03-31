#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _load_checked(path: Path) -> pd.DataFrame:
    from rdkit import Chem

    df = pd.read_csv(path, low_memory=False)
    need = {
        "uuid",
        "category",
        "dataset",
        "input_format",
        "output_format",
        "is_correct",
        "inchi",
        "smiles",
        "model",
        "effort",
    }
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required columns {sorted(missing)}")
    out = df.copy()
    out["is_correct_bool"] = out["is_correct"].map(_to_bool)
    mols = out["smiles"].astype(str).map(Chem.MolFromSmiles)
    if mols.isna().any():
        bad = out.loc[mols.isna(), "uuid"].head(5).tolist()
        raise ValueError(f"{path}: failed to parse ground-truth SMILES for UUIDs {bad}")
    out["gt_heavy_count"] = mols.map(lambda m: int(m.GetNumHeavyAtoms()))
    out["gt_ring_count"] = mols.map(lambda m: int(m.GetRingInfo().NumRings()))
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    order = ["smiles", "iupac", "graph"]
    x = df[(df["category"] == "translation") & (df["dataset"] == "translation_large")].copy()
    x = x[x["input_format"].isin(order) & x["output_format"].isin(order)].copy()
    x = x[x["input_format"] != x["output_format"]].copy()
    return x


def _direction_key(inp: str, out: str) -> str:
    return f"{inp}_to_{out}"


def _load_translation_meta_from_questions(questions_dir: Path) -> pd.DataFrame:
    from rdkit import Chem

    rows: list[dict[str, object]] = []
    for name in ("translation_small.jsonl", "translation_large.jsonl"):
        path = questions_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing question file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                meta = row.get("meta") or {}
                mol = meta.get("molecule") or {}
                smiles = mol.get("smiles")
                inchi = mol.get("inchi")
                if not isinstance(smiles, str) or not smiles.strip():
                    continue
                rd_mol = Chem.MolFromSmiles(smiles)
                if rd_mol is None:
                    raise ValueError(f"{path}: failed to parse molecule SMILES for uuid {row.get('uuid', '')}")
                rows.append(
                    {
                        "smiles": smiles,
                        "inchi": inchi if isinstance(inchi, str) else "",
                        "n_rings": int(rd_mol.GetRingInfo().NumRings()),
                        "heavy_atoms": int(rd_mol.GetNumHeavyAtoms()),
                        "datasets": str(meta.get("dataset", "")),
                    }
                )

    meta_df = pd.DataFrame(rows)
    if meta_df.empty:
        raise ValueError(f"No translation metadata rows found in {questions_dir}")
    meta_df = meta_df.drop_duplicates(subset=["inchi", "smiles"], keep="first").copy()
    return meta_df


def _summarize(rows: pd.DataFrame, model_label: str) -> pd.DataFrame:
    g = (
        rows.groupby(["effort", "input_format", "output_format", "gt_ring_count", "gt_heavy_count"], as_index=False)
        .agg(n_questions=("is_correct_bool", "size"), n_correct=("is_correct_bool", "sum"))
        .copy()
    )
    g["model"] = model_label
    g["direction"] = [_direction_key(i, o) for i, o in zip(g["input_format"], g["output_format"])]
    g["accuracy_pct"] = 100.0 * g["n_correct"] / g["n_questions"]
    ci = g.apply(lambda r: _wilson_ci_pct(int(r["n_correct"]), int(r["n_questions"])), axis=1)
    g["ci_low_pct"] = [x[0] for x in ci]
    g["ci_high_pct"] = [x[1] for x in ci]
    cols = [
        "model",
        "effort",
        "direction",
        "input_format",
        "output_format",
        "gt_ring_count",
        "gt_heavy_count",
        "n_questions",
        "n_correct",
        "accuracy_pct",
        "ci_low_pct",
        "ci_high_pct",
    ]
    return g[cols]


def _plot_translation_direction_lines(
    df_checked: pd.DataFrame,
    *,
    meta_df: pd.DataFrame,
    ring_filter: int,
    title: str | None,
    ax: plt.Axes,
    show_ylabel: bool,
    show_xlabel: bool,
) -> dict[str, pd.DataFrame]:
    required = {"input_format", "output_format", "dataset", "inchi", "smiles", "is_correct_bool"}
    missing = required - set(df_checked.columns)
    if missing:
        raise KeyError(f"df_checked missing required columns: {sorted(missing)}")

    df = df_checked[df_checked["dataset"] == "translation_large"].copy()
    if df.empty:
        raise ValueError("No rows available for the requested translation subset.")

    merge_on = "inchi" if "inchi" in meta_df.columns and df["inchi"].notna().any() else "smiles"
    if merge_on not in meta_df.columns:
        raise KeyError(f"meta_df missing required merge column: {merge_on}")
    if "heavy_atoms" not in meta_df.columns or "n_rings" not in meta_df.columns:
        raise KeyError("meta_df missing required columns: ['heavy_atoms', 'n_rings']")

    df = df.merge(
        meta_df[[merge_on, "heavy_atoms", "n_rings"]],
        on=merge_on,
        how="left",
        suffixes=("", "_meta"),
    )
    df = df.dropna(subset=["n_rings", "heavy_atoms"]).copy()
    df["n_rings"] = df["n_rings"].astype(int)
    df["heavy_atoms"] = df["heavy_atoms"].astype(int)
    df = df[df["n_rings"] == ring_filter]

    directions = [
        ("smiles", "iupac"),
        ("iupac", "smiles"),
        ("iupac", "graph"),
        ("graph", "iupac"),
        ("smiles", "graph"),
        ("graph", "smiles"),
    ]
    output_color = {
        "smiles": "#E69F00",
        "iupac": "#56B4E9",
        "graph": "#009E73",
    }
    input_linestyle = {
        "smiles": "--",
        "iupac": "-",
        "graph": ":",
    }

    stats: dict[str, pd.DataFrame] = {}
    for inp, out_fmt in directions:
        sub = df[(df["input_format"] == inp) & (df["output_format"] == out_fmt)]
        if sub.empty:
            continue

        grp = (
            sub.assign(is_correct_num=sub["is_correct_bool"].astype(int))
            .groupby("heavy_atoms")["is_correct_num"]
            .agg(correct="sum", total="size")
            .sort_index()
        )
        grp["p"] = grp["correct"] / grp["total"]
        wilson = grp.apply(lambda row: _wilson_ci_pct(int(row["correct"]), int(row["total"])), axis=1)
        grp["ci_low_pct"] = [x[0] for x in wilson]
        grp["ci_high_pct"] = [x[1] for x in wilson]
        stats[f"{inp}->{out_fmt}"] = grp

        x = grp.index.to_numpy()
        y = grp["p"].to_numpy() * 100.0
        lo = grp["ci_low_pct"].to_numpy(dtype=float)
        hi = grp["ci_high_pct"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            label=f"{inp}→{out_fmt}",
            color=output_color[out_fmt],
            linestyle=input_linestyle[inp],
            linewidth=1.5,
            marker="",
            markersize=3,
        )
        ax.fill_between(
            x,
            lo,
            hi,
            color=output_color[out_fmt],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Heavy atom count" if show_xlabel else "", fontsize=16)
    ax.set_ylabel("Accuracy (%)" if show_ylabel else "", fontsize=16)
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.grid(True, color="#b0b0b0", alpha=0.4, linewidth=0.6)

    return stats


def _make_figure(
    *,
    df_nano: pd.DataFrame,
    df_mini: pd.DataFrame,
    df_full: pd.DataFrame,
    meta_df: pd.DataFrame,
    out_path: Path,
) -> None:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    rings = [0, 1, 2, 3]
    titles = ("GPT-5-Nano", "GPT-5-Mini", "GPT-5")

    prop_df = meta_df.copy()
    if "datasets" in prop_df.columns:
        prop_df = prop_df[prop_df["datasets"].astype(str).str.contains("translation_large", na=False)].copy()
    for col in ("n_rings", "heavy_atoms"):
        if col in prop_df.columns:
            prop_df[col] = pd.to_numeric(prop_df[col], errors="coerce")
    prop_df = prop_df.dropna(subset=["smiles", "n_rings", "heavy_atoms"]).copy()
    prop_df["n_rings"] = prop_df["n_rings"].astype(int)
    prop_df["heavy_atoms"] = prop_df["heavy_atoms"].astype(int)
    dedup_col = "inchi" if "inchi" in prop_df.columns else "smiles"
    prop_df = prop_df.drop_duplicates(subset=[dedup_col], keep="first").copy()

    fs_cache: dict[str, bool | None] = {}

    def _has_fused_or_spiro(smiles: str) -> bool | None:
        s = str(smiles).strip()
        if not s:
            return None
        if s in fs_cache:
            return fs_cache[s]
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fs_cache[s] = None
            return None
        bond_rings = [set(r) for r in mol.GetRingInfo().BondRings()]
        has_fused = any(bool(bond_rings[i] & bond_rings[j]) for i in range(len(bond_rings)) for j in range(i + 1, len(bond_rings)))
        has_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0
        out = bool(has_fused or has_spiro)
        fs_cache[s] = out
        return out

    prop_df["fused_or_spiro"] = prop_df["smiles"].map(_has_fused_or_spiro)
    prop_df = prop_df.dropna(subset=["fused_or_spiro"]).copy()
    prop_df["fused_or_spiro"] = prop_df["fused_or_spiro"].astype(bool)
    fused_spiro_prop = (
        prop_df.groupby(["n_rings", "heavy_atoms"], as_index=False)["fused_or_spiro"]
        .mean()
        .rename(columns={"fused_or_spiro": "fused_spiro_prop"})
    )

    fig, axes = plt.subplots(
        len(rings),
        3,
        figsize=(8.8, 13.2),
        dpi=200,
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.02, "hspace": 0.12},
    )

    model_dfs = (df_nano, df_mini, df_full)
    for r_idx, ring in enumerate(rings):
        for c_idx, (df_checked, title) in enumerate(zip(model_dfs, titles)):
            ax = axes[r_idx, c_idx]
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1)
            if ring in {2, 3}:
                psub = fused_spiro_prop[fused_spiro_prop["n_rings"] == int(ring)].sort_values("heavy_atoms")
                if not psub.empty:
                    ax.plot(
                        psub["heavy_atoms"].to_numpy(dtype=float),
                        (100.0 * psub["fused_spiro_prop"]).to_numpy(dtype=float),
                        color="#969696",
                        linestyle=(0, (2.0, 2.4)),
                        linewidth=1.35,
                        alpha=0.64,
                        zorder=0,
                    )
            plot_df = df_checked.copy()
            _plot_translation_direction_lines(
                plot_df,
                meta_df=meta_df,
                ring_filter=ring,
                title=title if r_idx == 0 else None,
                ax=ax,
                show_ylabel=(c_idx == 0),
                show_xlabel=(r_idx == len(rings) - 1),
            )
            ax.tick_params(axis="x", labelbottom=(r_idx == len(rings) - 1))
            ax.tick_params(axis="y", labelleft=(c_idx == 0))

        axes[r_idx, 0].text(
            -0.32,
            0.5,
            f"{ring} rings" if ring != 1 else "1 ring",
            transform=axes[r_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=16,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        handle_by_label = {lbl: h for h, lbl in zip(handles, labels)}
        display_map = {
            "smiles→graph": "SMILES -> MolJSON",
            "iupac→graph": "IUPAC -> MolJSON",
            "smiles→iupac": "SMILES -> IUPAC",
            "graph→iupac": "MolJSON -> IUPAC",
            "graph→smiles": "MolJSON -> SMILES",
            "iupac→smiles": "IUPAC -> SMILES",
        }
        legend_input_order = [
            "iupac→graph",
            "smiles→graph",
            "__fused_spiro__",
            "smiles→iupac",
            "graph→iupac",
            "iupac→smiles",
            "graph→smiles",
        ]
        fused_handle = Line2D([0], [0], color="#969696", linestyle=(0, (2.0, 2.4)), linewidth=1.35, alpha=0.64)
        ordered_handles = []
        ordered_display = []
        for key in legend_input_order:
            if key == "__fused_spiro__":
                ordered_handles.append(fused_handle)
                ordered_display.append("% fused/spiro rings")
                continue
            if key in handle_by_label:
                ordered_handles.append(handle_by_label[key])
                ordered_display.append(display_map.get(key, key))
        fig.legend(
            ordered_handles,
            ordered_display,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=14,
            bbox_to_anchor=(0.5, 0.03),
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.18, wspace=0.02, hspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.14, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    parser = argparse.ArgumentParser(description="Generate translation ring/atom-count accuracy figure + table.")
    parser.add_argument(
        "--checked-dir",
        default=str(package_root / "model_responses" / "checked"),
    )
    parser.add_argument(
        "--questions-dir",
        default=str(package_root / "questions"),
    )
    parser.add_argument(
        "--out-table",
        default=str(package_root / "analysis_outputs" / "tables" / "accuracy_by_direction_ring_heavy_model.csv"),
    )
    parser.add_argument(
        "--out-fig",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-translation-direction-lines-ring-grid-low.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checked_dir = Path(args.checked_dir)
    questions_dir = Path(args.questions_dir)
    out_table = Path(args.out_table)
    out_fig = Path(args.out_fig)

    files = [
        ("gpt-5-nano", checked_dir / "gpt-5-nano-low_checked.csv"),
        ("gpt-5-mini", checked_dir / "gpt-5-mini-low_checked.csv"),
        ("gpt-5", checked_dir / "gpt-5-low_checked.csv"),
    ]
    loaded: dict[str, pd.DataFrame] = {}
    summary_parts: list[pd.DataFrame] = []
    for model_label, path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing checked file: {path}")
        raw = _load_checked(path)
        sub = _prepare(raw)
        if sub.empty:
            raise ValueError(f"No rows left after filtering for {path.name}")
        loaded[model_label] = sub
        summary_parts.append(_summarize(sub, model_label))

    out_table.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.sort_values(
        ["model", "effort", "direction", "gt_ring_count", "gt_heavy_count"],
    ).to_csv(out_table, index=False)

    meta_df = _load_translation_meta_from_questions(questions_dir)
    _make_figure(
        df_nano=loaded["gpt-5-nano"],
        df_mini=loaded["gpt-5-mini"],
        df_full=loaded["gpt-5"],
        meta_df=meta_df,
        out_path=out_fig,
    )

    print(f"Wrote {out_table}")
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
