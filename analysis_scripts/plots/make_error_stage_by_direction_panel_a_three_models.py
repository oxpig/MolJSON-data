#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DIRECTION_ORDER = [
    "iupac->graph",
    "iupac->smiles",
    "smiles->graph",
    "smiles->iupac",
    "graph->smiles",
    "graph->iupac",
]

LABEL_MAP = {
    "iupac->graph": "IUPAC -> MolJSON",
    "iupac->smiles": "IUPAC -> SMILES",
    "smiles->graph": "SMILES -> MolJSON",
    "smiles->iupac": "SMILES -> IUPAC",
    "graph->smiles": "MolJSON -> SMILES",
    "graph->iupac": "MolJSON -> IUPAC",
}

STAGE_ORDER_BASE = [
    "correct",
    "parsed_but_wrong_molecule",
    "fundamental_parse_failure",
    "no_output_extracted",
]

COLORS = {
    "correct": "#2f855a",
    "parsed_but_wrong_molecule": "#4e79a7",
    "fundamental_parse_failure": "#f28e2b",
    "no_output_extracted": "#e15759",
}

STAGE_LABEL = {
    "correct": "Correct",
    "parsed_but_wrong_molecule": "RDKit valid but incorrect molecule",
    "fundamental_parse_failure": "Non-empty response",
    "no_output_extracted": "Empty response",
}

MODEL_SPECS = [
    ("GPT-5-nano", "gpt-5-nano-low_checked.csv", "gpt5nano"),
    ("GPT-5-mini", "gpt-5-mini-low_checked.csv", "gpt5mini"),
    ("GPT-5", "gpt-5-low_checked.csv", "gpt5"),
]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _parse_obj(value: object) -> object | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _extract_raw_output_value(model_answer: object, output_format: str) -> object | None:
    obj = _parse_obj(model_answer)
    if not isinstance(obj, dict):
        return None

    if output_format == "graph":
        if "atoms" in obj and "bonds" in obj:
            return obj
        return None

    return obj.get(output_format)


def classify_non_smiles_error_stage(error_detail: str, output_format: str, model_answer: object) -> str:
    e = (error_detail or "").strip()
    raw_value = _extract_raw_output_value(model_answer, output_format)
    if isinstance(raw_value, str) and raw_value == "":
        return "no_output_extracted"
    if output_format == "graph" and isinstance(raw_value, dict):
        atoms = raw_value.get("atoms")
        if isinstance(atoms, list) and len(atoms) == 0:
            return "no_output_extracted"
    if output_format == "graph":
        if e == "graph_smiles_mismatch":
            return "parsed_but_wrong_molecule"
        return "fundamental_parse_failure"
    if output_format == "iupac":
        if e == "iupac_smiles_mismatch":
            return "parsed_but_wrong_molecule"
        return "fundamental_parse_failure"
    return "fundamental_parse_failure"


def classify_smiles_error_stage(error_detail: str, model_answer: object) -> str:
    e = (error_detail or "").strip()
    raw_value = _extract_raw_output_value(model_answer, "smiles")
    if isinstance(raw_value, str) and raw_value == "":
        return "no_output_extracted"
    if e == "smiles_mismatch":
        return "parsed_but_wrong_molecule"
    return "fundamental_parse_failure"


def _load_summary(path: Path, model_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need = {
        "category",
        "dataset",
        "input_format",
        "output_format",
        "is_correct",
        "error_detail",
        "model_answer",
    }
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing required columns {sorted(missing)}")

    df = df[(df["category"] == "translation") & (df["dataset"] == "translation_large")].copy()
    df = df[df["input_format"] != df["output_format"]].copy()
    df["input_format"] = df["input_format"].astype(str).str.lower()
    df["output_format"] = df["output_format"].astype(str).str.lower()
    df["direction"] = df["input_format"] + "->" + df["output_format"]
    df = df[df["direction"].isin(DIRECTION_ORDER)].copy()
    df["is_correct_bool"] = df["is_correct"].map(_to_bool)

    df["stage"] = "correct"
    bad = ~df["is_correct_bool"]

    mask_smiles = bad & df["output_format"].eq("smiles")
    df.loc[mask_smiles, "stage"] = df.loc[mask_smiles].apply(
        lambda r: classify_smiles_error_stage(r.get("error_detail", ""), r.get("model_answer", "")),
        axis=1,
    )

    mask_non = bad & ~df["output_format"].eq("smiles")
    df.loc[mask_non, "stage"] = df.loc[mask_non].apply(
        lambda r: classify_non_smiles_error_stage(r.get("error_detail", ""), r["output_format"], r.get("model_answer", "")),
        axis=1,
    )

    summary = (
        df.groupby(["direction", "stage"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby("direction").size().rename("n_total").reset_index()
    summary = summary.merge(totals, on="direction", how="left")
    summary["pct_total"] = 100.0 * summary["count"] / summary["n_total"]
    summary["model_label"] = model_label
    return summary


def _panel(ax: plt.Axes, df: pd.DataFrame, panel_title: str, stage_order: list[str]) -> None:
    x = np.arange(len(DIRECTION_ORDER), dtype=float)
    bottom = np.zeros(len(DIRECTION_ORDER), dtype=float)

    for st in stage_order:
        vals: list[float] = []
        for d in DIRECTION_ORDER:
            rr = df[(df["direction"] == d) & (df["stage"] == st)]
            vals.append(float(rr.iloc[0]["pct_total"]) if not rr.empty else 0.0)

        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            color=COLORS[st],
            edgecolor="white",
            linewidth=0.9,
        )

        if st in {"correct", "parsed_but_wrong_molecule", "fundamental_parse_failure", "no_output_extracted"}:
            for i, (b, v) in enumerate(zip(bars, vals)):
                if v >= 7:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        bottom[i] + v / 2,
                        f"{v:.1f}",
                        ha="center",
                        va="center",
                        fontsize=13.5,
                        color="white",
                    )
        bottom += np.asarray(vals)

    ax.set_title(panel_title, fontsize=26)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_MAP[d] for d in DIRECTION_ORDER], rotation=25, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1.0)


def parse_args() -> argparse.Namespace:
    package_root = _package_root()
    p = argparse.ArgumentParser(
        description="Plot 3-panel error-stage-by-direction panel A for GPT-5-nano, GPT-5-mini, and GPT-5."
    )
    p.add_argument(
        "--checked-dir",
        default=str(package_root / "model_responses" / "checked"),
    )
    p.add_argument(
        "--out-summary-dir",
        default=str(package_root / "analysis_outputs" / "tables" / "error_stage_by_direction"),
    )
    p.add_argument(
        "--out-fig",
        default=str(package_root / "analysis_outputs" / "plots" / "fig-error-stage-by-direction-panel-a-three-models.png"),
    )
    p.add_argument("--fig-width", type=float, default=17.2)
    p.add_argument("--fig-height", type=float, default=8.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checked_dir = Path(args.checked_dir)
    out_summary_dir = Path(args.out_summary_dir)
    out_fig = Path(args.out_fig)
    out_summary_dir.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    for model_label, filename, stem in MODEL_SPECS:
        summary = _load_summary(checked_dir / filename, model_label)
        summaries.append(summary)
        summary.to_csv(out_summary_dir / f"{stem}_error_stage_by_direction_summary.csv", index=False)

    d_nano, d_mini, d_full = summaries

    stage_order = list(STAGE_ORDER_BASE)

    fig, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_height), dpi=260, sharey=True)
    _panel(axes[0], d_nano, "GPT-5-nano", stage_order)
    _panel(axes[1], d_mini, "GPT-5-mini", stage_order)
    _panel(axes[2], d_full, "GPT-5", stage_order)

    axes[0].set_ylabel("Share of responses (%)", fontsize=19)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    handles = [Patch(facecolor=COLORS[s], edgecolor="white", label=STAGE_LABEL[s]) for s in stage_order]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(stage_order),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=16,
    )

    fig.tight_layout(rect=[0.01, 0.10, 1.0, 0.98])
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out_fig, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
