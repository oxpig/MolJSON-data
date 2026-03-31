from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df_checked missing required columns: {sorted(missing)}")


def _ordered(values: Iterable[str], preferred: list[str] | None) -> list[str]:
    existing = list(values)
    if preferred is None:
        return sorted(existing)
    ordered = list(preferred)
    ordered += [v for v in existing if v not in ordered]
    return ordered


def _label_map(label_map: dict[str, str] | None, default_map: dict[str, str]) -> dict[str, str]:
    merged = dict(default_map)
    if label_map:
        merged.update(label_map)
    return merged


def _pivot_accuracy(
    df: pd.DataFrame,
    *,
    fill_value: float | None,
) -> pd.DataFrame:
    return (
        df.assign(is_correct_num=df["is_correct"].astype(bool).astype(int))
        .pivot_table(
            index="input_format",
            columns="output_format",
            values="is_correct_num",
            aggfunc="mean",
            fill_value=fill_value,
        )
        .mul(100.0)
    )


def _pivot_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["input_format", "output_format"]).size().unstack(fill_value=0)


def plot_translation_matrix(
    df_checked: pd.DataFrame,
    *,
    input_order: list[str] | None = None,
    output_order: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    dpi: int = 300,
    cell: float = 1.6,
    vmin: float = 0,
    vmax: float = 100,
    text_size: int = 13,
    title: str = "All results",
    mask_diagonal: bool = False,
    show: bool = False,
    show_counts: bool = False,
    count_text_size: int = 9,
    count_color: str = "#333333",
    count_fmt: str = "n={n}",
    high_value_text_threshold: float = 70.0,
    high_value_text_color: str = "white",
    default_text_color: str = "black",
    ax: plt.Axes | None = None,
):
    required = {"input_format", "output_format", "is_correct"}
    _ensure_columns(df_checked, required)

    mat = _pivot_accuracy(df_checked, fill_value=0.0)
    counts = _pivot_counts(df_checked) if show_counts else None

    rows = _ordered(mat.index.tolist(), input_order)
    cols = _ordered(mat.columns.tolist(), output_order)
    mat = mat.reindex(index=rows, columns=cols)
    if counts is not None:
        counts = counts.reindex(index=rows, columns=cols, fill_value=0)

    if mask_diagonal:
        for fmt in set(rows).intersection(cols):
            mat.loc[fmt, fmt] = np.nan

    lm = _label_map(label_map, {"V2000_MOLBLOCK": "molfile"})
    row_labels = [lm.get(r, r) for r in rows]
    col_labels = [lm.get(c, c) for c in cols]

    cmap = plt.cm.Greens.copy()
    cmap.set_bad(color="white")

    fig_w = cell * max(1, len(cols))
    fig_h = cell * max(1, len(rows))
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Output Representation", fontsize=14)
    ax.set_ylabel("Input Representation", fontsize=14)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=12)

    for i in range(len(rows)):
        for j in range(len(cols)):
            value = mat.iat[i, j]
            if np.isfinite(value):
                text_color = high_value_text_color if value > high_value_text_threshold else default_text_color
                ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    color=text_color,
                )
            if counts is not None:
                n = int(counts.iat[i, j])
                if n > 0:
                    count_text_color = high_value_text_color if np.isfinite(value) and value > high_value_text_threshold else count_color
                    ax.text(
                        j + 0.48,
                        i + 0.48,
                        count_fmt.format(n=n),
                        ha="right",
                        va="bottom",
                        fontsize=count_text_size,
                        color=count_text_color,
                    )

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

    return mat, fig, ax
