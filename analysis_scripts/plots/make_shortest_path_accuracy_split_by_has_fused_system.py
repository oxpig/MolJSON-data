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
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(tempfile.gettempdir()) / "xdg_cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem

INPUT_ORDER = ["smiles", "iupac", "graph"]
INPUT_LABEL = {"smiles": "SMILES", "iupac": "IUPAC", "graph": "MolJSON"}
INPUT_COLOR = {"smiles": "#E69F00", "iupac": "#56B4E9", "graph": "#009E73"}
AXIS_LABEL_FS = 19
TICK_LABEL_FS = 19
ANNOT_FS = TICK_LABEL_FS
LEGEND_FS = 18
X_LABEL_PAD = 16


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    p = argparse.ArgumentParser(
        description=(
            "Plot GPT-5 shortest-path accuracy split by whether the molecule "
            "contains any fused ring system."
        )
    )
    p.add_argument("--model", default="gpt-5")
    p.add_argument(
        "--results",
        default=str(package_root / "model_responses" / "checked" / "gpt-5-low_checked.csv"),
    )
    p.add_argument(
        "--questions-file",
        default=str(package_root / "questions" / "shortest_path_questions.jsonl"),
    )
    p.add_argument(
        "--out-summary",
        default=str(package_root / "analysis_outputs" / "tables" / "gpt5_shortest_path_accuracy_split_by_has_fused_system_summary.csv"),
    )
    p.add_argument(
        "--out-molecule-classes",
        default=str(package_root / "analysis_outputs" / "tables" / "gpt5_shortest_path_molecule_has_fused_system_classes.csv"),
    )
    p.add_argument(
        "--out-figure",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-gpt5-shortest-path-accuracy-split-fused-system.png"),
    )
    p.add_argument("--fig-width", type=float, default=9.6)
    p.add_argument("--fig-height", type=float, default=5.8)
    return p.parse_args()


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


def _fused_ring_components(mol: Chem.Mol) -> list[set[int]]:
    bond_rings = [set(r) for r in mol.GetRingInfo().BondRings()]
    n = len(bond_rings)
    if n < 2:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if bond_rings[i].intersection(bond_rings[j]):
                union(i, j)

    comp_members: dict[int, list[int]] = {}
    for i in range(n):
        comp_members.setdefault(find(i), []).append(i)

    return [set(int(i) for i in idxs) for idxs in comp_members.values() if len(idxs) >= 2]


def _molecule_has_fused_system(smiles: str) -> bool | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return len(_fused_ring_components(mol)) > 0


def _canonicalize_smiles(smiles: object) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _load_shortest_path_answers(questions_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with questions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("category", "")).strip().lower() != "shortest_path":
                continue
            rows.append(
                {
                    "uuid": str(row.get("uuid", "")).strip(),
                    "answer_num": pd.to_numeric(row.get("answer"), errors="coerce"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No shortest_path rows found in {questions_path}")
    out = out.dropna(subset=["answer_num"]).copy()
    out["answer_num"] = out["answer_num"].astype(int)
    return out


def _load_rows(path: Path, questions_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    sub = df[
        (df["model"].astype(str).str.lower() == str(model_name).strip().lower())
        & (df["category"] == "shortest_path")
    ].copy()
    sub = sub[sub["input_format"].astype(str).str.lower().isin(INPUT_ORDER)].copy()
    sub = sub[sub["output_format"].astype(str).str.lower() == "integer"].copy()
    sub["_nonempty"] = sub["model_answer"].notna() & (sub["model_answer"].astype(str).str.strip() != "")

    keys = ["uuid", "category", "input_format", "output_format"]
    sub = sub.sort_values("_nonempty", ascending=False).drop_duplicates(subset=keys, keep="first")
    sub["input_format"] = sub["input_format"].astype(str).str.lower()
    sub["is_correct_bool"] = sub["is_correct"].map(_to_bool).astype(bool)
    sub["smiles_canon"] = sub["smiles"].map(_canonicalize_smiles)

    answers = _load_shortest_path_answers(questions_path)
    sub = sub.merge(answers, on="uuid", how="left")
    sub = sub.dropna(subset=["answer_num"]).copy()
    sub["answer_num"] = sub["answer_num"].astype(int)
    return sub


def _molecule_classes(rows: pd.DataFrame) -> pd.DataFrame:
    mol = rows[["smiles_canon", "smiles", "answer_num"]].dropna(subset=["smiles_canon"]).copy()
    mol = mol.drop_duplicates(subset=["smiles_canon"], keep="first")
    mol["has_fused_system"] = mol["smiles_canon"].map(_molecule_has_fused_system)
    mol["class_ok"] = mol["has_fused_system"].notna()
    return mol[mol["class_ok"]].copy()


def _summary(rows: pd.DataFrame, mol_classes: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(mol_classes[["smiles_canon", "has_fused_system"]], on="smiles_canon", how="inner")
    out_rows: list[dict[str, object]] = []
    for has_fused in [False, True]:
        part = merged[merged["has_fused_system"] == has_fused]
        for fmt in INPUT_ORDER:
            s = part[part["input_format"] == fmt]
            if s.empty:
                continue
            grp = (
                s.groupby("answer_num", as_index=False)["is_correct_bool"]
                .agg(n_correct="sum", n_total="size")
                .sort_values("answer_num")
            )
            for _, row in grp.iterrows():
                n_correct = int(row["n_correct"])
                n_total = int(row["n_total"])
                lo, hi = _wilson_ci_pct(n_correct, n_total)
                out_rows.append(
                    {
                        "has_fused_system": bool(has_fused),
                        "subset_label": "has_fused_system" if has_fused else "no_fused_system",
                        "input_format": fmt,
                        "answer_num": int(row["answer_num"]),
                        "n_total": n_total,
                        "n_correct": n_correct,
                        "accuracy_pct": (100.0 * n_correct / n_total) if n_total else float("nan"),
                        "ci_low_pct": lo,
                        "ci_high_pct": hi,
                    }
                )
    return pd.DataFrame(out_rows)


def _legend_handles() -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for fmt in INPUT_ORDER:
        handles.append(Line2D([0], [0], color=INPUT_COLOR[fmt], linewidth=2.2))
        labels.append(INPUT_LABEL[fmt])
    return handles, labels


def _plot(summary: pd.DataFrame, out_path: Path, fig_width: float, fig_height: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=260, sharey=True)
    panel_defs = [
        (False, "(a) No fused ring system"),
        (True, "(b) Has fused ring system"),
    ]

    for ax, (has_fused, title) in zip(axes, panel_defs):
        block = summary[summary["has_fused_system"] == has_fused].copy()
        for fmt in INPUT_ORDER:
            s = block[block["input_format"] == fmt].sort_values("answer_num")
            if s.empty:
                continue
            x = s["answer_num"].to_numpy(dtype=int)
            y = s["accuracy_pct"].to_numpy(dtype=float)
            lo = s["ci_low_pct"].to_numpy(dtype=float)
            hi = s["ci_high_pct"].to_numpy(dtype=float)
            ax.plot(x, y, color=INPUT_COLOR[fmt], linewidth=2.2, zorder=3)
            ax.fill_between(x, lo, hi, color=INPUT_COLOR[fmt], alpha=0.16, zorder=2)

        xticks = sorted(block["answer_num"].unique().tolist())
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) if i % 2 == 0 else "" for i, x in enumerate(xticks)])
        ax.set_box_aspect(1.0)
        ax.set_title(title, fontsize=ANNOT_FS, color="#222222", pad=10)
        ax.set_xlabel("Shortest path length", fontsize=AXIS_LABEL_FS, labelpad=X_LABEL_PAD)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Accuracy (%)", fontsize=AXIS_LABEL_FS)
    handles, labels = _legend_handles()
    fig.legend(handles, labels, frameon=False, fontsize=LEGEND_FS, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0.01, 0.15, 1.0, 0.98])
    fig.subplots_adjust(wspace=0.16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.results), Path(args.questions_file), args.model)
    mol_classes = _molecule_classes(rows)
    summary = _summary(rows, mol_classes)

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    out_classes = Path(args.out_molecule_classes)
    out_classes.parent.mkdir(parents=True, exist_ok=True)
    mol_classes.to_csv(out_classes, index=False)

    _plot(summary, Path(args.out_figure), args.fig_width, args.fig_height)

    n_has = int((mol_classes["has_fused_system"] == True).sum())
    n_no = int((mol_classes["has_fused_system"] == False).sum())
    print(f"rows_used={len(rows)}")
    print(f"unique_molecules_classified={len(mol_classes)}")
    print(f"molecules_has_fused_system={n_has}")
    print(f"molecules_no_fused_system={n_no}")
    print(f"wrote_summary={out_summary.resolve()}")
    print(f"wrote_molecule_classes={out_classes.resolve()}")
    print(f"wrote_figure={Path(args.out_figure).resolve()}")


if __name__ == "__main__":
    main()
