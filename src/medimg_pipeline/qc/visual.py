"""Automatic visual QC: overlay a predicted (and optionally ground-truth)
segmentation mask on representative slices of the source image, saved as
a PNG. Never requires a GUI or a display -- uses Matplotlib's non-
interactive Agg backend explicitly so this also works headless in CI/
Docker.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _representative_slice_indices(mask: np.ndarray, axis: int, n_slices: int) -> list[int]:
    """Pick slice indices along `axis` that actually contain foreground,
    spread evenly through the foreground region, so QC figures show the
    organ rather than empty background slices.
    """

    foreground_present = np.any(mask > 0, axis=tuple(a for a in range(mask.ndim) if a != axis))
    indices = np.where(foreground_present)[0]
    if len(indices) == 0:
        # No foreground anywhere: fall back to evenly-spaced slices through
        # the whole volume so the QC figure is still produced.
        size = mask.shape[axis]
        return list(np.linspace(0, size - 1, num=min(n_slices, size), dtype=int))
    return list(
        np.linspace(indices.min(), indices.max(), num=min(n_slices, len(indices)), dtype=int)
    )


def generate_overlay_figure(
    image: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    *,
    ground_truth: np.ndarray | None = None,
    n_slices: int = 4,
    axis: int = 2,
    title: str = "Segmentation QC",
) -> Path:
    """Save a PNG with `n_slices` axial (or `axis`-oriented) slices showing
    the image with the predicted mask overlaid in red, and (if given) the
    ground-truth mask outlined in green for comparison.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = _representative_slice_indices(prediction, axis, n_slices)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices, strict=True):
        image_slice = np.take(image, idx, axis=axis)
        pred_slice = np.take(prediction, idx, axis=axis)

        ax.imshow(image_slice.T, cmap="gray", origin="lower")
        pred_overlay = np.ma.masked_where(pred_slice.T == 0, pred_slice.T)
        ax.imshow(pred_overlay, cmap="autumn", alpha=0.5, origin="lower")

        if ground_truth is not None:
            gt_slice = np.take(ground_truth, idx, axis=axis)
            ax.contour(gt_slice.T, levels=[0.5], colors="lime", linewidths=1.0, origin="lower")

        ax.set_title(f"slice {idx}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
