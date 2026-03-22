from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

MPLCONFIGDIR = Path("/tmp/qrt_mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator


QRT_COLOR = "#8CA377"
QRT_RAW_ALPHA = 0.14
QRT_BAND_ALPHA = 0.20
QRT_LINE_ALPHA = 0.98
HP_LINE_W = 4.0
HP_SPINE_W = 2.2
HP_TICK_FS = 18
HP_GRID_W = 1.2

METRICS = (
    ("Robust Accuracy", "qrt_rob_accs"),
    ("Clean Accuracy", "qrt_clean_accs"),
    ("Clean Loss", "qrt_clean_losses"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    plot_dir = root / "plot_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-npz", default=str(plot_dir / "plot_data_clean_loss_only_bs5_steps120000.npz"))
    parser.add_argument("--robust-npz", default=str(plot_dir / "plot_data_robust_loss_only_bs5_steps120000.npz"))
    parser.add_argument("--combined-npz", default=str(plot_dir / "plot_data_combined_loss_alpha0.50_bs5_steps120000.npz"))
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--bins", type=int, default=150)
    parser.add_argument("--raw-points", type=int, default=240)
    parser.add_argument("--q-low", type=float, default=0.20)
    parser.add_argument("--q-high", type=float, default=0.80)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    return parser.parse_args()


def sanitize_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    keep = x >= 0.0
    x = x[keep]
    y = y[keep]
    if x.size > 1:
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    return x, y


def sample_raw_points(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points, dtype=int)
    idx = np.unique(idx)
    return x[idx], y[idx]


def safe_savgol(y: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    _ = polyorder
    if y.size < 5:
        return y
    win = min(window, y.size if y.size % 2 == 1 else y.size - 1)
    if win < 5:
        return y
    kernel = np.ones(win, dtype=float) / float(win)
    padded = np.pad(y, (win // 2, win // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: y.size]


def binned_summary(
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    q_low: float,
    q_high: float,
    smooth_window: int,
    smooth_polyorder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    centers: list[float] = []
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for idx in range(bins):
        if idx < bins - 1:
            mask = (x >= edges[idx]) & (x < edges[idx + 1])
        else:
            mask = (x >= edges[idx]) & (x <= edges[idx + 1])
        if not np.any(mask):
            continue
        x_bin = x[mask]
        y_bin = y[mask]
        centers.append(float(np.median(x_bin)))
        medians.append(float(np.median(y_bin)))
        lows.append(float(np.quantile(y_bin, q_low)))
        highs.append(float(np.quantile(y_bin, q_high)))
    xc = np.asarray(centers, dtype=float)
    ym = np.asarray(medians, dtype=float)
    yl = np.asarray(lows, dtype=float)
    yh = np.asarray(highs, dtype=float)
    return (
        xc,
        safe_savgol(ym, smooth_window, smooth_polyorder),
        safe_savgol(yl, smooth_window, smooth_polyorder),
        safe_savgol(yh, smooth_window, smooth_polyorder),
    )


def format_step_ticks(value: float, _pos: int) -> str:
    if abs(value) >= 1000:
        return f"{int(round(value / 1000.0))}k"
    if abs(value) >= 1:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def style_axis(ax: plt.Axes) -> None:
    ax.grid(which="major", color="gray", alpha=0.28, linewidth=HP_GRID_W)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", labelsize=HP_TICK_FS, width=2.2, length=7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(format_step_ticks))
    for spine in ax.spines.values():
        spine.set_linewidth(HP_SPINE_W)
        spine.set_color("black")


def plot_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    bins: int,
    q_low: float,
    q_high: float,
    raw_points: int,
    smooth_window: int,
    smooth_polyorder: int,
) -> None:
    x, y = sanitize_x(x, y)
    xr, yr = sample_raw_points(x, y, raw_points)
    xb, ym, yl, yh = binned_summary(
        x=x,
        y=y,
        bins=bins,
        q_low=q_low,
        q_high=q_high,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
    )
    ax.scatter(xr, yr, s=14, color=QRT_COLOR, alpha=QRT_RAW_ALPHA, linewidths=0, zorder=1)
    ax.fill_between(xb, yl, yh, color=QRT_COLOR, alpha=QRT_BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(xb, ym, color=QRT_COLOR, linewidth=HP_LINE_W, alpha=QRT_LINE_ALPHA, zorder=3)
    style_axis(ax)


def load_row(npz_path: Path) -> dict[str, object]:
    with np.load(npz_path, allow_pickle=True) as data:
        batch_size = int(np.asarray(data["batch_size"]).item()) if "batch_size" in data else -1
        total_steps = int(np.asarray(data["total_steps"]).item()) if "total_steps" in data else -1
        loss_alpha = float(np.asarray(data["loss_alpha"]).item()) if "loss_alpha" in data.files else None
        arrays = {
            "x": np.asarray(data["qrt_steps"], dtype=float),
            "qrt_rob_accs": np.asarray(data["qrt_rob_accs"], dtype=float),
            "qrt_clean_accs": np.asarray(data["qrt_clean_accs"], dtype=float),
            "qrt_clean_losses": np.asarray(data["qrt_clean_losses"], dtype=float),
        }
    name = npz_path.name
    if "clean_loss_only" in name:
        row_label = "Mode: Clean-only training"
    elif "robust_loss_only" in name:
        row_label = "Mode: Robust-only training"
    else:
        alpha_text = f"{loss_alpha:.2f}" if loss_alpha is not None else "0.50"
        row_label = f"Mode: Combined training (alpha={alpha_text})"
    return {
        "path": npz_path,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "row_label": row_label,
        **arrays,
    }


def infer_base_name(rows: list[dict[str, object]]) -> str:
    batch_size = rows[0]["batch_size"]
    total_steps = rows[0]["total_steps"]
    if all(row["batch_size"] == batch_size for row in rows) and all(row["total_steps"] == total_steps for row in rows):
        return f"qrt_only_nine_panel_bs{batch_size}_steps{total_steps}"
    return "qrt_only_nine_panel_summary"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        load_row(Path(args.clean_npz)),
        load_row(Path(args.robust_npz)),
        load_row(Path(args.combined_npz)),
    ]
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1.0, 1.0, 0.16, 1.0], hspace=0.30, wspace=0.16)
    axes: list[list[plt.Axes]] = []
    for row_idx in range(3):
        axes.append([fig.add_subplot(gs[row_idx, 0]), fig.add_subplot(gs[row_idx, 1]), fig.add_subplot(gs[row_idx, 3])])
    for row_idx, row in enumerate(rows):
        x = np.asarray(row["x"], dtype=float)
        for col_idx, (title, array_key) in enumerate(METRICS):
            plot_panel(
                ax=axes[row_idx][col_idx],
                x=x,
                y=np.asarray(row[array_key], dtype=float),
                bins=args.bins,
                q_low=args.q_low,
                q_high=args.q_high,
                raw_points=args.raw_points,
                smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
            )
            if row_idx == 0:
                axes[row_idx][col_idx].set_title(title, fontsize=22, pad=12)
        axes[row_idx][0].text(
            0.03,
            0.96,
            str(row["row_label"]),
            transform=axes[row_idx][0].transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=4,
        )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.09)
    first_col_pos = axes[1][0].get_position()
    second_col_pos = axes[1][1].get_position()
    third_col_pos = axes[1][2].get_position()
    loss_label_x = ((second_col_pos.x1 + third_col_pos.x0) / 2.0) - 0.01
    fig.text(0.035, 0.52, "Accuracy", rotation=90, va="center", ha="center", fontsize=26)
    fig.text(loss_label_x, 0.52, "Loss", rotation=90, va="center", ha="center", fontsize=26)
    fig.text(0.52, 0.035, "Training Step", ha="center", va="center", fontsize=26)
    base_name = infer_base_name(rows)
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
