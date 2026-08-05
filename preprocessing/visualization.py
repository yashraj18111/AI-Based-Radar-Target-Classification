"""
Visualization utilities for radar Range-Doppler and Range-Angle maps.

Renders single frames or batches with optional bounding-box overlays,
suitable for debugging, reports, and publication figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import config
from preprocessing.loader import RadarSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _setup_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _draw_bounding_boxes(
    ax: plt.Axes,
    sample: RadarSample,
) -> None:
    """Overlay CARRADA-style bounding boxes on the radar map."""
    for bbox in sample.annotations:
        x_min, y_min, x_max, y_max = bbox.as_xyxy()
        width = x_max - x_min
        height = y_max - y_min
        color = config.BBOX_COLORS.get(bbox.class_name, "#FFFFFF")

        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x_min,
            y_min - 2,
            bbox.class_name,
            color=color,
            fontsize=8,
            fontweight="bold",
            clip_on=True,
        )


def _axis_labels_for_modality(modality: str) -> tuple[str, str]:
    if "angle" in modality:
        return "Angle bin", "Range bin"
    return "Doppler bin", "Range bin"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_sample(
    sample: RadarSample,
    show_boxes: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple[float, float] = config.VIS_FIGSIZE,
    cmap: str = config.VIS_COLORMAP,
) -> plt.Figure:
    """
    Visualize a single radar sample.

    Parameters
    ----------
    sample : RadarSample
        Loaded radar frame from :class:`RadarDataset`.
    show_boxes : bool
        Draw annotation bounding boxes when available.
    save_path : path-like, optional
        If provided, save the figure to this location.
    show : bool
        Call ``plt.show()`` when True.
    figsize : tuple
        Matplotlib figure size in inches.
    cmap : str
        Colormap name for the radar heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=config.VIS_DPI)

    display_map = sample.radar_map.squeeze()
    xlabel, ylabel = _axis_labels_for_modality(sample.modality)

    im = ax.imshow(
        display_map,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized power")

    title = (
        f"{sample.sequence_id} | frame {sample.frame_id} | "
        f"{sample.modality} | objects: {sample.num_objects}"
    )
    _setup_axes(ax, title, xlabel, ylabel)

    if show_boxes and sample.annotations:
        _draw_bounding_boxes(ax, sample)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=config.VIS_DPI)
        logger.info("Saved visualization to %s", save_path)

    if show:
        plt.show()

    return fig


def visualize_batch(
    samples: List[RadarSample],
    show_boxes: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    max_cols: int = 4,
    cmap: str = config.VIS_COLORMAP,
) -> plt.Figure:
    """
    Visualize multiple radar samples in a grid layout.

    Parameters
    ----------
    samples : list of RadarSample
        Frames to display.
    show_boxes : bool
        Draw bounding boxes on each subplot.
    save_path : path-like, optional
        Output file path.
    show : bool
        Display the figure interactively.
    max_cols : int
        Maximum number of columns in the grid.
    cmap : str
        Colormap for heatmaps.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not samples:
        raise ValueError("visualize_batch requires at least one sample")

    n = len(samples)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3 * nrows),
        dpi=config.VIS_DPI,
        squeeze=False,
    )

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        display_map = sample.radar_map.squeeze()
        xlabel, ylabel = _axis_labels_for_modality(sample.modality)

        im = ax.imshow(
            display_map,
            aspect="auto",
            origin="upper",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_title(f"{sample.frame_id} ({sample.num_objects} obj)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)

        if show_boxes and sample.annotations:
            _draw_bounding_boxes(ax, sample)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    fig.suptitle(
        f"Radar batch — {samples[0].modality} ({n} frames)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=config.VIS_DPI)
        logger.info("Saved batch visualization to %s", save_path)

    if show:
        plt.show()

    return fig


def save_sample_preview(
    sample: RadarSample,
    output_dir: Union[str, Path] = config.FIGURE_OUTPUT_DIR,
    prefix: str = "preview",
) -> Path:
    """
    Convenience helper: save a single-sample preview to the figures directory.

    Returns the path of the saved PNG file.
    """
    output_dir = Path(output_dir)
    filename = f"{prefix}_{sample.sequence_id}_{sample.frame_id}.png"
    save_path = output_dir / filename
    visualize_sample(sample, save_path=save_path, show=False)
    plt.close()
    return save_path
