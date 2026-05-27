"""Result visualization that re-reads pixels from source rasters.

Nothing on disk except the FAISS store — to render results we open each
source raster and pull the relevant window via SourceReader at display time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .chips import ChipSpec
from .io import SourceReader


def show_results(
    scored_specs: List[Tuple[float, ChipSpec]],
    query_image: Optional[str] = None,
    max_display: int = 10,
    cols: int = 5,
) -> None:
    """Plot query thumbnail (top) + result grid below, reading windows on demand."""
    n = min(max_display, len(scored_specs))
    cols = max(1, min(cols, n)) if n > 0 else 1
    rows = int(np.ceil(n / cols)) if n > 0 else 0

    text_color = "#1a1a1a"
    muted_color = "#6b6b6b"
    border_color = "#d8d8d8"
    bg_color = "#fafafa"

    fig_w = min(2.2 * cols, 13.5)
    fig_h = min(2.0 * rows + 3.0, 8.5)
    fig = plt.figure(
        figsize=(fig_w, fig_h),
        dpi=90,
        constrained_layout=True,
        facecolor=bg_color,
    )
    gs = fig.add_gridspec(
        rows + 1,
        cols,
        height_ratios=[1.5] + [1.0] * rows if rows else [1.0],
    )

    def _style(ax, lw: float = 0.8) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(bg_color)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(border_color)
            s.set_linewidth(lw)

    ax_q = fig.add_subplot(gs[0, :])
    if query_image is not None:
        ax_q.imshow(Image.open(query_image).convert("RGB"))
    ax_q.set_title("QUERY", fontsize=10, fontweight="bold", color=muted_color, loc="left", pad=10)
    _style(ax_q, lw=1.2)

    # Group by source so each raster is opened once
    by_source: dict = {}
    for i, (score, spec) in enumerate(scored_specs[:n]):
        by_source.setdefault(spec.source, []).append((i, score, spec))

    panels: List[Tuple[int, float, ChipSpec, Image.Image]] = []
    for source, items in by_source.items():
        with SourceReader(source) as reader:
            for i, score, spec in items:
                panels.append((i, score, spec, reader.read(spec.x, spec.y, spec.window)))
    panels.sort(key=lambda t: t[0])

    for i, score, spec, img in panels:
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        ax.imshow(img)
        ax.set_title(
            f"#{i + 1:02d}",
            fontsize=11,
            fontweight="bold",
            color=text_color,
            loc="left",
            pad=6,
        )
        ax.set_xlabel(
            f"score {score:.3f}   W={spec.window}",
            fontsize=9,
            color=muted_color,
            labelpad=6,
        )
        _style(ax)

    fig.suptitle(
        "Similarity Search Results",
        fontsize=18,
        fontweight="600",
        color=text_color,
        x=0.02,
        ha="left",
    )
    plt.show()
