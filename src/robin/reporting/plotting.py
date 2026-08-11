"""
plotting.py

This module contains functions for creating plots used in the PDF report.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpath_effects
import matplotlib.ticker as mticker
import io
import math
import textwrap
import matplotlib.font_manager as fm
import os
from robin.gui import fonts

import natsort

from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

import logging
from typing import Iterator, List, Optional, Sequence, Tuple, Dict, Any

logger = logging.getLogger(__name__)

from robin.reference_contigs import is_visible_contig
from robin.cnv_plot_style import (
    CNV_CHROMOSOME_AXIS_LOG2,
    horizontal_lane_height,
    CNV_PLOIDY_BASELINE,
    horizontal_label_placements,
    CNV_LOG2_MAX_AXIS_SPAN,
    CNV_LOG2_MIN_AXIS_SPAN,
    cnv_chromosome_axis_window,
    clamp_band_for_markers,
    cnv_arm_mean_spans,
    gene_bin_window,
    gene_crosses_cutoff,
    cnv_axis_tick_spec,
    cnv_reference_levels,
    cnv_segment_spans,
    robust_gene_value,
    snap_axis_window_to_ticks,
)

# Define consistent color scheme and style
MODERN_COLORS = {
    "primary": "#2C3E50",  # Dark blue-grey (matching report text)
    "secondary": "#E2E8F0",  # Light grey (matching table grid)
    "background": "#F8FAFC",  # Light background (matching table alternate rows)
    "accent": "#3498DB",  # Blue accent
    "grid": "#E2E8F0",  # Grid color
}

CNV_COLORS = {
    "points": "#5B7C9D",
    "median": "#1E3A5F",
    "diploid": "#3D5A80",
    "reference": "#C5CDD6",
    "gain_fill": "#D4EDDA",
    "loss_fill": "#F8D7DA",
    "gain_edge": "#3D7A4A",
    "loss_edge": "#A94442",
    "gene": "#5C6BC0",
    # Report convention (PDF only; the live GUI keeps its own palette):
    # gains blue, losses red, clinical trial targets purple whichever way they went.
    "plot_gain": "#1D4ED8",
    "plot_loss": "#C81E1E",
    "plot_neutral": "#9CA3AF",
    "plot_trial": "#7E22CE",
    "cutoff": "#B45309",
    # Darker than the plain grid: reviewers need the chromosome boundary and
    # the p/q split to be readable at a glance on a 24in genome panel.
    "centromere": "#64748B",
    "contig_boundary": "#475569",
}

# Scatter alpha for dense CNV tracks (lower = clearer overplotting).
CNV_POINT_ALPHA_NEUTRAL = 0.38
CNV_POINT_ALPHA_CALLED = 0.72
CNV_POINT_ALPHA_DEFAULT = 0.45

_CNV_GENE_LABEL_PATH_EFFECTS = [
    mpath_effects.withStroke(linewidth=2.6, foreground="white", alpha=0.95),
]
CNV_FONT = {
    "title": 11,
    "subtitle": 8,
    "axis": 9.5,
    "tick": 7.5,
    "annotation": 6.5,
}

CNV_TEXT = {
    "primary": MODERN_COLORS["primary"],
    "muted": "#5A6B7D",
}

CNV_CHROMOSOME_FIG_WIDTH = 7.5
CNV_CHROMOSOME_PAD_INCHES = 0.12
# Gene-label lane sizing: label height x line spacing, as a share of panel height.
CNV_ROTATED_LABEL_MARKER_CLEARANCE = 0.025
# A rotated name runs along the y-axis, so its extent scales with how long it is
# and with the label size. Used only to decide which side of the marker has room.
CNV_ROTATED_LABEL_EXTENT_PER_CHAR = 0.035
CNV_ROTATED_LABEL_EXTENT_MAX = 0.45
CNV_CHROMOSOME_SCATTER_SIZE = 6
CNV_CHROMOSOME_PLOTS_PER_PAGE = 4
CNV_CHROMOSOME_PLOT_SPACER_PT = 6
# ReportLab's default Frame uses 6pt padding on each side.
CNV_REPORT_FRAME_PADDING_PT = 12

_CNV_FONT_REGULAR: Optional[fm.FontProperties] = None
_CNV_FONT_BOLD: Optional[fm.FontProperties] = None
_CNV_FONTS_READY = False


def _setup_cnv_fonts() -> None:
    """Register report plot fonts with a safe fallback when bundled TTFs are absent."""
    global _CNV_FONT_REGULAR, _CNV_FONT_BOLD, _CNV_FONTS_READY
    if _CNV_FONTS_READY:
        return

    fonts_dir = os.path.dirname(os.path.abspath(fonts.__file__))
    regular_path = os.path.join(fonts_dir, "fira-sans-v16-latin-regular.ttf")
    bold_path = os.path.join(fonts_dir, "fira-sans-v16-latin-700.ttf")
    family = "DejaVu Sans"

    if os.path.isfile(regular_path):
        try:
            fm.fontManager.addfont(regular_path)
            _CNV_FONT_REGULAR = fm.FontProperties(fname=regular_path)
            family = _CNV_FONT_REGULAR.get_name()
        except Exception:
            _CNV_FONT_REGULAR = fm.FontProperties(family=family)
    else:
        _CNV_FONT_REGULAR = fm.FontProperties(family=family)

    if os.path.isfile(bold_path):
        try:
            fm.fontManager.addfont(bold_path)
            _CNV_FONT_BOLD = fm.FontProperties(fname=bold_path)
        except Exception:
            _CNV_FONT_BOLD = fm.FontProperties(family=family, weight="bold")
    else:
        _CNV_FONT_BOLD = fm.FontProperties(family=family, weight="bold")

    plt.rcParams["font.family"] = family
    # Keep math labels on the same sans family as axis text (no Computer Modern mix).
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    _CNV_FONTS_READY = True


def set_modern_style():
    """Set consistent modern style for all plots"""
    _setup_cnv_fonts()

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])

    plt.rcParams.update(
        {
            # Figure settings
            "figure.facecolor": "white",
            "figure.dpi": 300,
            # Axes settings
            "axes.facecolor": "white",
            "axes.edgecolor": MODERN_COLORS["primary"],
            "axes.labelcolor": MODERN_COLORS["primary"],
            "axes.titlecolor": MODERN_COLORS["primary"],
            "axes.grid": True,
            "axes.labelsize": CNV_FONT["axis"],
            "axes.titlesize": CNV_FONT["title"],
            # Grid settings
            "grid.color": MODERN_COLORS["grid"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            # Tick settings
            "xtick.color": MODERN_COLORS["primary"],
            "ytick.color": MODERN_COLORS["primary"],
            "xtick.labelsize": CNV_FONT["tick"],
            "ytick.labelsize": CNV_FONT["tick"],
            # Legend settings
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": MODERN_COLORS["grid"],
            "legend.fontsize": CNV_FONT["tick"],
            # Line settings
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def _apply_cnv_y_tick_ladder(ax, *, use_log: bool) -> None:
    """Fine, mirrored copy-number ticks (WGM/conumee convention).

    Labels are duplicated on the right-hand edge so a point in the middle of a
    wide panel is never far from a tick to read it against, and minor ticks add
    resolution between labels without crowding them.
    """
    y_lo, y_hi = ax.get_ylim()
    ticks = cnv_axis_tick_spec(y_lo, y_hi, use_log=use_log)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(ticks.major))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(ticks.minor))
    # Trim trailing zeros so fine intervals do not read as "0.625" everywhere.
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _pos: f"{value:g}")
    )
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(MODERN_COLORS["grid"])
    ax.yaxis.set_ticks_position("both")
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=True,
        labelleft=True,
        labelright=True,
        colors=CNV_TEXT["primary"],
        labelsize=CNV_FONT["tick"],
    )
    ax.tick_params(axis="y", which="minor", length=2, labelleft=False, labelright=False)
    ax.grid(
        True,
        axis="y",
        which="minor",
        color=MODERN_COLORS["grid"],
        linestyle=":",
        linewidth=0.3,
        alpha=0.35,
    )


def _snap_cnv_ylim(y_min: float, y_max: float, *, use_log: bool) -> Tuple[float, float]:
    """Round CNV axis limits out onto whole ticks so the panel ends on a gridline."""
    ticks = cnv_axis_tick_spec(y_min, y_max, use_log=use_log)
    return snap_axis_window_to_ticks(
        y_min,
        y_max,
        ticks.major,
        clamp_min=None if use_log else 0.0,
    )


def _apply_cnv_chromosome_axes(
    ax,
    x_max_mb: float,
    y_max: float,
    *,
    y_min: float = 0.0,
    xlabel: str,
    ylabel: str,
    show_xlabel: bool = True,
    use_log: bool = False,
) -> None:
    """Pin axes at the origin with no leading x padding."""
    _setup_cnv_fonts()
    ax.set_xlim(0, x_max_mb)
    ax.set_ylim(y_min, y_max)
    ax.margins(x=0, y=0)
    ax.autoscale(enable=False)
    if show_xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=CNV_FONT["axis"],
            color=CNV_TEXT["primary"],
            labelpad=10,
            fontproperties=_CNV_FONT_REGULAR,
        )
    else:
        ax.set_xlabel("")
    ax.set_ylabel(
        ylabel,
        fontsize=CNV_FONT["axis"],
        color=CNV_TEXT["primary"],
        labelpad=10,
        fontproperties=_CNV_FONT_REGULAR,
    )
    ax.spines["left"].set_position(("data", 0))
    if y_min < 0:
        # Keep the Mb axis at the bottom of the panel; log2 reference stays at y=0 inside.
        ax.spines["bottom"].set_position(("axes", 0.0))
    else:
        ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, axis="y", color=MODERN_COLORS["grid"], linestyle="--", linewidth=0.4, alpha=0.55)
    ax.grid(False, axis="x")
    ax.tick_params(colors=CNV_TEXT["primary"], labelsize=CNV_FONT["tick"])
    _apply_cnv_y_tick_ladder(ax, use_log=use_log)


def _chromosome_display_name(contig: str) -> str:
    return contig.replace("chr", "")


def _plot_cnv_track(
    ax,
    cnv_df: pd.DataFrame,
    x_max_mb: float,
    *,
    point_size: float = 4,
    show_trend: bool = True,
    contig: Optional[str] = None,
) -> None:
    """Render CNV bin values as scatter with a rolling median trace."""
    ax.scatter(
        cnv_df["position_mb"],
        cnv_df["ploidy"],
        s=point_size,
        c=CNV_COLORS["points"],
        alpha=CNV_POINT_ALPHA_DEFAULT,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    if show_trend:
        _add_cnv_trend_line(
            ax,
            cnv_df["position_mb"],
            cnv_df["ploidy"],
            split_at=_centromere_split(contig, 1_000_000.0),
        )


def _is_gene_cnv_outlier(
    cnv_val: float,
    *,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    use_log2: bool = True,
    baseline: Optional[float] = None,
    cutoff_override: Optional[float] = None,
) -> bool:
    """True when a panel target crosses the gain/loss cut-off in force.

    Judged against the genome-wide baseline, not the gene's own chromosome: a
    standard-deviation rule taken per chromosome cancels out whole-chromosome and
    arm-level events, so every gene on a gained chromosome looks normal.
    """
    gain_threshold, loss_threshold = _cnv_cutoff_thresholds(
        chromosome, sex_estimate, cutoff_override
    )
    if gain_threshold is None or loss_threshold is None:
        return False
    return gene_crosses_cutoff(
        cnv_val,
        gain_threshold=gain_threshold,
        loss_threshold=loss_threshold,
        use_log=use_log2,
        baseline=baseline,
    )


SIGNIFICANT_CNV_REGION_TYPES = {"GAIN", "LOSS", "HIGH_GAIN", "DEEP_LOSS"}


def _should_label_panel_gene(
    point: Dict[str, Any],
    *,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    use_log2: bool = True,
    baseline: Optional[float] = None,
    cutoff_override: Optional[float] = None,
) -> bool:
    """Label panel genes that cross the gain/loss cut-off in force."""
    return _is_gene_cnv_outlier(
        point["cnv_val"],
        chromosome=chromosome,
        sex_estimate=sex_estimate,
        use_log2=use_log2,
        baseline=baseline,
        cutoff_override=cutoff_override,
    )


def _add_cnv_regions_on_plot(ax, regions: List[Dict[str, Any]], y_max: float) -> None:
    """Shade called regions on the main track with edge guides and top brackets."""
    for region in regions:
        start_mb = region["start_pos"] / 1_000_000
        end_mb = region["end_pos"] / 1_000_000
        is_gain = region["type"] in ("GAIN", "HIGH_GAIN")
        fill_color = CNV_COLORS["gain_fill"] if is_gain else CNV_COLORS["loss_fill"]
        edge_color = CNV_COLORS["gain_edge"] if is_gain else CNV_COLORS["loss_edge"]

        ax.axvspan(start_mb, end_mb, color=fill_color, alpha=0.55, zorder=0, linewidth=0)
        ax.axvline(start_mb, color=edge_color, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
        ax.axvline(end_mb, color=edge_color, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
        _draw_region_bracket(
            ax, start_mb, end_mb, y_max * 0.992, edge_color, height_frac=y_max * 0.028,
        )


PANEL_LABEL_MAX_CHARS = 10
# Genome-wide figure geometry for the standalone PDF download, which is not
# constrained by a page: a wide canvas is the whole advantage of it. Font sizes
# are absolute points, so widening the figure buys horizontal room for 24
# chromosomes of gene labels without shrinking any text — it is the only lever
# that reduces label crowding without touching the labels themselves. Do not
# narrow this to match the report page: the report passes its own frame width
# explicitly, and it is the one that has to live inside a page.
CNV_GENOME_FIG_WIDTH = 24.0
# Height is deliberately not scaled with the width — crowding on this plot is
# horizontal, so the width gives the x-room without a tall figure. Kept flatter
# than 4:1 because a genome profile read at full width looks stretched when the
# panel is deep: the data occupies a narrow band and the rest is empty axis.
CNV_GENOME_FIG_ASPECT = 5.0
# Axis furniture reserved around the genome-wide panel, in inches. These are
# absolute rather than fractions of the figure: matplotlib's fractional defaults
# left the panel using ~77% of the canvas whatever its size, which showed up as
# a wide empty strip down the right-hand side. Sized to the y-axis label plus
# tick labels on the left, and the mirrored tick labels on the right.
# Panel gene marker styling. A gene is plotted at its own copy-number value, so
# on a densely binned chromosome it lands inside a cloud of same-coloured points
# — the white ring and stem halo are what make it findable there.
# Kept small so a run of lollipops on a gene-dense chromosome reads as a row of
# fine stems rather than a line of blobs. The white ring, not the area, is what
# separates a marker from the bins around it — shrinking much below this makes a
# gene disappear into a densely binned cloud, which is what made EGFR on a gained
# chr7 look absent from its own chromosome plot.
CNV_PANEL_MARKER_SIZE = 12.0
CNV_PANEL_MARKER_SIZE_OFFSCALE = 20.0
CNV_PANEL_MARKER_EDGE_WIDTH = 0.7
CNV_PANEL_STEM_WIDTH = 0.9
CNV_PANEL_STEM_ALPHA = 0.85
#: Chromosome boundary and centromere line weights on the CNV panels.
CNV_CONTIG_BOUNDARY_WIDTH = 0.9
CNV_CENTROMERE_WIDTH = 0.8
#: Gap between a chromosome boundary and its name, as a fraction of the
#: whole genome width, so the name never sits on top of the line.
CNV_CONTIG_LABEL_PAD_FRAC = 0.0018
#: Gap in points between a marker's outer edge and the start of its name.
CNV_PANEL_LABEL_GAP_PT = 1.6

CNV_GENOME_MARGIN_LEFT_IN = 0.62
CNV_GENOME_MARGIN_RIGHT_IN = 0.42
CNV_GENOME_MARGIN_TOP_IN = 0.30
CNV_GENOME_MARGIN_BOTTOM_IN = 0.52


def _fill_genome_canvas(fig, width: float, height: float) -> None:
    """Expand the genome panel to the figure edges, less its axis furniture.

    Keeps the reserved space constant in inches, so a wider figure turns into a
    wider *panel* rather than a wider empty margin.
    """
    width = max(float(width), 1e-6)
    height = max(float(height), 1e-6)
    left = min(CNV_GENOME_MARGIN_LEFT_IN / width, 0.4)
    right = max(1.0 - CNV_GENOME_MARGIN_RIGHT_IN / width, left + 0.2)
    bottom = min(CNV_GENOME_MARGIN_BOTTOM_IN / height, 0.4)
    top = max(1.0 - CNV_GENOME_MARGIN_TOP_IN / height, bottom + 0.2)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


# Genome-wide panel width in points, used to judge how wide a horizontal label is
# relative to the axis when deciding which names need stacking.
CNV_DEFAULT_PANEL_WIDTH_PT = (
    CNV_GENOME_FIG_WIDTH - CNV_GENOME_MARGIN_LEFT_IN - CNV_GENOME_MARGIN_RIGHT_IN
) * 72.0
# Fallback panel height in points, used when the caller does not know the figure
# geometry; the real value is passed in so lane spacing adapts to the panel.
CNV_DEFAULT_PANEL_HEIGHT_PT = 108.0
# Default gene label size in points. Overridable per figure via the plotting
# preferences, so the reporting team can tune label density without a code change.
LOLLIPOP_LABEL_FONT_SIZE = 4.5


def _truncate_panel_label(label: str, max_chars: int = PANEL_LABEL_MAX_CHARS) -> str:
    label = str(label).strip()
    if len(label) <= max_chars:
        return label
    return f"{label[: max_chars - 1]}…"


def _wrap_chromosome_status_text(status_text: str, *, width: int = 88) -> str:
    parts = [part.strip() for part in status_text.split(";") if part.strip()]
    if not parts:
        return status_text
    lines: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}; {part}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = textwrap.fill(part, width=width)
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


def _chromosome_figure_margins(
    *,
    has_status: bool,
    compact: bool = False,
) -> Dict[str, float]:
    if compact:
        return {
            "top": 0.70 if has_status else 0.80,
            "bottom": 0.20,
            "right": 0.97,
            "left": 0.13,
        }
    return {
        "top": 0.76 if has_status else 0.86,
        "bottom": 0.18,
        "right": 0.97,
        "left": 0.13,
    }


def _normalise_coverage_to_cnv_axis(
    coverage: float,
    *,
    mean_cov: float,
    scale_mean_cnv: float,
    use_log: bool,
) -> Optional[float]:
    """Map target depth onto the CNV y-axis for visual comparison.

    Linear/ploidy: ``scale_mean_cnv * (cov / mean_cov)`` so mean coverage sits
    at the CNV mean. Log2: ``log2(cov / mean_cov)`` so mean coverage sits at 0.
    """
    if not np.isfinite(coverage) or coverage <= 0:
        return None
    if not np.isfinite(mean_cov) or mean_cov <= 0:
        return None
    ratio = float(coverage) / float(mean_cov)
    if use_log:
        return float(np.log2(ratio))
    if not np.isfinite(scale_mean_cnv):
        return None
    return float(scale_mean_cnv) * ratio


def _panel_label_matches_configured(label: str, configured_genes: Sequence[str]) -> bool:
    """True when a panel target label matches a configured ``[cnv].genes`` symbol."""
    key = str(label).strip().casefold()
    if not key:
        return False
    for gene in configured_genes:
        want = str(gene).strip().casefold()
        if not want:
            continue
        if key == want:
            return True
        if key.startswith(f"{want}_") or key.startswith(f"{want}-") or key.startswith(
            f"{want} "
        ):
            return True
    return False


def _marker_clearance_data_units(y_span: float, panel_height_pt: float) -> float:
    """Vertical gap that clears a panel gene marker, in data units."""
    radius_pt = math.sqrt(CNV_PANEL_MARKER_SIZE / math.pi)
    clearance_pt = radius_pt + CNV_PANEL_MARKER_EDGE_WIDTH + CNV_PANEL_LABEL_GAP_PT
    if panel_height_pt <= 0:
        return 0.0
    return float(y_span) * clearance_pt / float(panel_height_pt)


def _layout_panel_coverage_point_labels(
    coverage_points: List[Dict[str, Any]],
    y_min: float,
    y_max: float,
    x_max: float,
    *,
    x_key: str,
    min_x_spacing: float,
    band_lo: Optional[float] = None,
    band_hi: Optional[float] = None,
    label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    rotated: bool = True,
    panel_width_pt: float = CNV_DEFAULT_PANEL_WIDTH_PT,
    panel_height_pt: float = CNV_DEFAULT_PANEL_HEIGHT_PT,
    reference_levels: Sequence[float] = (),
) -> Dict[tuple[str, float], float]:
    """Position each gene name relative to its own marker.

    Rotated names (the default) are narrow, so each sits just clear of its marker
    and nothing is reserved for them. Horizontal names are as wide as the gene
    symbol, so overlapping ones are stacked into lanes instead — still inside the
    existing axis, which means on a crowded panel they sit over the bin cloud.
    """
    if not coverage_points:
        return {}

    if not rotated:
        y_span = max(y_max - y_min, 1e-6)
        lane_height = horizontal_lane_height(
            y_span=y_span,
            panel_height_pt=panel_height_pt,
            font_size=float(label_size),
        )
        placements = horizontal_label_placements(
            [float(point[x_key]) for point in coverage_points],
            [_truncate_panel_label(point["label"]) for point in coverage_points],
            [
                "below" if point.get("direction") == "loss" else "above"
                for point in coverage_points
            ],
            [float(point["y_norm"]) for point in coverage_points],
            x_span=float(x_max) if x_max else None,
            lane_height=lane_height,
            font_size=float(label_size),
            panel_width_pt=float(panel_width_pt),
            y_lo=y_min + y_span * 0.01,
            y_hi=y_max - y_span * 0.01,
            reference_levels=reference_levels or (),
        )
        return {
            (point["label"], float(point[x_key])): placement["y"]
            for point, placement in zip(coverage_points, placements)
        }
    y_span = max(y_max - y_min, 1e-6)
    # Clear the marker itself, not just a fixed fraction of the axis. The marker
    # carries a white ring so it reads over a dense cloud, and a name anchored
    # inside that ring hides the very marker it is naming.
    offset = max(
        y_span * CNV_ROTATED_LABEL_MARKER_CLEARANCE,
        _marker_clearance_data_units(y_span, panel_height_pt),
    )

    layouts: Dict[tuple[str, float], float] = {}
    for point in coverage_points:
        x_pos = float(point[x_key])
        y_head = float(point["y_norm"])
        label = _truncate_panel_label(point["label"])
        size_scale = float(label_size) / LOLLIPOP_LABEL_FONT_SIZE
        extent = y_span * min(
            len(label) * CNV_ROTATED_LABEL_EXTENT_PER_CHAR * size_scale,
            CNV_ROTATED_LABEL_EXTENT_MAX,
        )
        above = point.get("direction") != "loss"
        # Gains read upwards and losses downwards, unless that side has no room.
        if above and y_head + extent > y_max:
            above = False
        elif not above and y_head - extent < y_min:
            above = True
        label_y = y_head + offset if above else y_head - offset
        layouts[(point["label"], x_pos)] = float(label_y)
    return layouts


def _annotate_significant_panel_points(
    panel_points: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    *,
    use_log2: bool,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    baseline: Optional[float] = None,
    cutoff_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Keep panel targets that cross the gain/loss cut-off in force."""
    significant_points: List[Dict[str, Any]] = []
    for point in panel_points:
        if not _should_label_panel_gene(
            point,
            chromosome=chromosome,
            sex_estimate=sex_estimate,
            use_log2=use_log2,
            baseline=baseline,
            cutoff_override=cutoff_override,
        ):
            continue
        significant_points.append(
            {
                **point,
                "direction": _gene_cnv_direction(point, regions, use_log2=use_log2),
            }
        )
    return significant_points


def _annotate_configured_panel_points(
    panel_points: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    *,
    use_log2: bool,
    configured_genes: Sequence[str],
    outliers_only: bool = False,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    baseline: Optional[float] = None,
    cutoff_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Keep configured ``[cnv].genes`` targets, optionally only those past the cut-off."""
    annotated: List[Dict[str, Any]] = []
    for point in panel_points:
        if not _panel_label_matches_configured(point["label"], configured_genes):
            continue
        if outliers_only and not _should_label_panel_gene(
            point,
            chromosome=chromosome,
            sex_estimate=sex_estimate,
            use_log2=use_log2,
            baseline=baseline,
            cutoff_override=cutoff_override,
        ):
            continue
        annotated.append(
            {
                **point,
                "direction": _gene_cnv_direction(point, regions, use_log2=use_log2),
            }
        )
    return annotated


def _tag_clinical_trial_points(
    points: List[Dict[str, Any]],
    clinical_trial_genes: Sequence[str],
) -> List[Dict[str, Any]]:
    """Mark panel points whose gene is a current clinical trial target."""
    if not clinical_trial_genes:
        return points
    return [
        {
            **point,
            "clinical_trial": _panel_label_matches_configured(
                point.get("label", ""), clinical_trial_genes
            ),
        }
        for point in points
    ]


def _select_panel_coverage_points(
    panel_points: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    *,
    use_log2: bool,
    configured_genes: Sequence[str] = (),
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    baseline: Optional[float] = None,
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    clinical_trial_genes: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Prefer the configured gene list; otherwise keep genes crossing the cut-off."""
    if configured_genes:
        return _tag_clinical_trial_points(
            _annotate_configured_panel_points(
                panel_points,
                regions,
                use_log2=use_log2,
                configured_genes=configured_genes,
                outliers_only=outliers_only,
                chromosome=chromosome,
                sex_estimate=sex_estimate,
                baseline=baseline,
                cutoff_override=cutoff_override,
            ),
            clinical_trial_genes,
        )
    return _tag_clinical_trial_points(
        _annotate_significant_panel_points(
            panel_points,
            regions,
            use_log2=use_log2,
            chromosome=chromosome,
            sex_estimate=sex_estimate,
            baseline=baseline,
            cutoff_override=cutoff_override,
        ),
        clinical_trial_genes,
    )


def _attach_normalised_coverage(
    points: List[Dict[str, Any]],
    *,
    mean_cov: Optional[float],
    scale_mean_cnv: float,
    use_log2: bool,
) -> List[Dict[str, Any]]:
    """Place each panel gene marker on the profile at its own copy-number value.

    Sequencing depth is kept as ``coverage_ratio`` for the caption rather than
    used as the marker height: mapping depth onto the copy-number axis put markers
    far off the profile and forced the axis open to reach them.
    """
    if not points:
        return []
    if mean_cov is None or not np.isfinite(mean_cov) or mean_cov <= 0:
        coverage_vals = np.asarray(
            [
                float(p["coverage_val"])
                for p in points
                if p.get("coverage_val") is not None
            ],
            dtype=float,
        )
        coverage_vals = coverage_vals[np.isfinite(coverage_vals) & (coverage_vals > 0)]
        mean_cov = float(np.mean(coverage_vals)) if len(coverage_vals) else None

    baseline_y = 0.0 if use_log2 else float(scale_mean_cnv)
    placed: List[Dict[str, Any]] = []
    for point in points:
        cnv_val = point.get("cnv_val")
        if cnv_val is None or not np.isfinite(float(cnv_val)):
            continue
        coverage_val = point.get("coverage_val")
        coverage_ratio = None
        if (
            coverage_val is not None
            and np.isfinite(float(coverage_val))
            and mean_cov
        ):
            coverage_ratio = float(coverage_val) / float(mean_cov)
        placed.append(
            {
                **point,
                "y_norm": float(cnv_val),
                "coverage_ratio": coverage_ratio,
                "baseline_y": float(baseline_y),
            }
        )
    return placed


def _expand_ylim_for_coverage_points(
    y_min: float,
    y_max: float,
    coverage_points: Sequence[Dict[str, Any]],
) -> Tuple[float, float]:
    """Widen CNV axis limits toward normalised coverage markers, within bounds.

    Expansion is capped so a deeply deleted panel target cannot flatten the whole
    CNV profile into a couple of ticks; markers past the cap are drawn clamped at
    the panel edge.
    """
    if not coverage_points:
        return y_min, y_max
    ys = [float(p["y_norm"]) for p in coverage_points if np.isfinite(p.get("y_norm", np.nan))]
    if not ys:
        return y_min, y_max
    lo, hi = clamp_band_for_markers(y_min, y_max, ys)
    pad = max((hi - lo) * 0.04, 0.1)
    return lo - pad, hi + pad


def _clamp_coverage_points_to_band(
    coverage_points: Sequence[Dict[str, Any]],
    y_min: float,
    y_max: float,
) -> List[Dict[str, Any]]:
    """Pin off-scale gene markers to the panel edge, and record that we did.

    A marker sitting on the edge would otherwise read as its clamped value — the
    same gene then appears at a different level on the genome-wide panel, whose
    axis is fitted to the data. The flag lets the marker be drawn as an off-scale
    arrow rather than an ordinary dot.
    """
    clamped: List[Dict[str, Any]] = []
    for point in coverage_points:
        y_norm = float(point.get("y_norm", 0.0))
        if not np.isfinite(y_norm):
            clamped.append(dict(point))
            continue
        limited = float(np.clip(y_norm, y_min, y_max))
        clamped.append(
            {
                **point,
                "y_norm": limited,
                "offscale": limited != y_norm,
                "true_value": y_norm,
            }
        )
    return clamped


def _add_panel_coverage_points(
    ax_cnv,
    coverage_points: List[Dict[str, Any]],
    x_max: float,
    *,
    x_key: str,
    min_x_spacing: float,
    y_min: float,
    y_max: float,
    band_lo: Optional[float] = None,
    band_hi: Optional[float] = None,
    label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    rotated: bool = True,
    panel_width_pt: float = CNV_DEFAULT_PANEL_WIDTH_PT,
    panel_height_pt: float = CNV_DEFAULT_PANEL_HEIGHT_PT,
    reference_levels: Sequence[float] = (),
) -> bool:
    """Plot panel genes as coverage normalised onto the shared CNV axis."""
    if not coverage_points:
        return False

    for point in coverage_points:
        x_pos = float(point[x_key])
        y_head = float(point["y_norm"])
        y_base = float(point.get("baseline_y", 0.0))
        color = _panel_point_color(point)
        # White outline under the stem so it stays traceable where it crosses a
        # dense same-coloured scatter cloud — on a fine-binned single chromosome
        # a thin translucent line simply disappears into the points.
        ax_cnv.vlines(
            x_pos,
            min(y_base, y_head),
            max(y_base, y_head),
            colors=color,
            linewidths=CNV_PANEL_STEM_WIDTH,
            alpha=CNV_PANEL_STEM_ALPHA,
            zorder=5,
            clip_on=True,
            path_effects=[
                mpath_effects.Stroke(
                    linewidth=CNV_PANEL_STEM_WIDTH + 1.4, foreground="white"
                ),
                mpath_effects.Normal(),
            ],
        )

    # Grouped by drawn colour so trial targets scatter together whatever their
    # direction. Off-scale markers get an arrow so the panel never implies a gene
    # sits at the axis edge when its real value is beyond it.
    by_style: Dict[tuple, List[Dict[str, Any]]] = {}
    for point in coverage_points:
        marker = "o"
        if point.get("offscale"):
            marker = "^" if float(point["true_value"]) > float(point["y_norm"]) else "v"
        by_style.setdefault((_panel_point_color(point), marker), []).append(point)
    for (color, marker), subset in by_style.items():
        # A wide white ring is what separates a gene marker from the scatter it
        # sits in. Without it a gene plotted correctly above its cluster is
        # indistinguishable from the cluster on a densely binned chromosome.
        ax_cnv.scatter(
            [float(point[x_key]) for point in subset],
            [float(point["y_norm"]) for point in subset],
            s=CNV_PANEL_MARKER_SIZE_OFFSCALE if marker != "o" else CNV_PANEL_MARKER_SIZE,
            marker=marker,
            color=color,
            zorder=6,
            edgecolors="white",
            linewidths=CNV_PANEL_MARKER_EDGE_WIDTH,
            alpha=1.0,
            clip_on=False,
        )

    head_label_y = _layout_panel_coverage_point_labels(
        coverage_points,
        y_min,
        y_max,
        x_max,
        x_key=x_key,
        min_x_spacing=min_x_spacing,
        band_lo=band_lo,
        band_hi=band_hi,
        label_size=label_size,
        rotated=rotated,
        panel_width_pt=panel_width_pt,
        panel_height_pt=panel_height_pt,
        reference_levels=reference_levels,
    )
    for point in coverage_points:
        x_pos = float(point[x_key])
        color = _panel_point_color(point)
        label_y = head_label_y[(point["label"], x_pos)]
        above = label_y >= float(point["y_norm"])
        ax_cnv.text(
            x_pos,
            label_y,
            _truncate_panel_label(point["label"]),
            # Rotated names read away from the baseline, so they stack
            # horizontally rather than eating the height of the panel. At 90
            # degrees the text runs upwards, so it is *ha* that decides whether
            # the name starts at the anchor or straddles it: with ha="center" a
            # rotated name sat centred on its anchor and covered its own marker.
            ha=("left" if above else "right") if rotated else "center",
            va="center" if rotated else ("bottom" if above else "top"),
            rotation=90 if rotated else 0,
            rotation_mode="anchor" if rotated else None,
            fontsize=float(label_size),
            color=color,
            fontweight="bold",
            zorder=8,
            # Not clipped: a rotated name is narrow, and truncating it mid-word
            # is worse than letting it reach a little past the panel edge.
            clip_on=False,
            path_effects=_CNV_GENE_LABEL_PATH_EFFECTS,
        )
    return True


def _panel_point_color(point: Dict[str, Any]) -> str:
    """Colour for a panel gene marker and its label.

    Clinical trial targets are purple whichever way they went, so a reporting
    scientist can pick out the genes with a trial route at a glance; everything
    else follows the report's gain/loss convention.
    """
    if point.get("clinical_trial"):
        return CNV_COLORS["plot_trial"]
    return (
        CNV_COLORS["plot_gain"]
        if point.get("direction") == "gain"
        else CNV_COLORS["plot_loss"]
    )


def _gene_cnv_direction(
    point: Dict[str, Any],
    regions: List[Dict[str, Any]],
    *,
    use_log2: bool,
) -> str:
    """Classify a significant panel gene as gain-like or loss-like for point colour."""
    mid_bp = float(point["mid_mb"]) * 1_000_000
    for region in regions:
        if region.get("type") not in SIGNIFICANT_CNV_REGION_TYPES:
            continue
        start_bp = float(region["start_pos"])
        end_bp = float(region["end_pos"])
        if start_bp <= mid_bp <= end_bp:
            if region["type"] in ("GAIN", "HIGH_GAIN"):
                return "gain"
            if region["type"] in ("LOSS", "DEEP_LOSS"):
                return "loss"
    cnv_val = float(point["cnv_val"])
    if use_log2:
        return "gain" if cnv_val >= 0.0 else "loss"
    return "gain" if cnv_val >= 0.0 else "loss"


def _mean_target_coverage(
    target_coverage_df: Optional[pd.DataFrame],
    contig: Optional[str] = None,
) -> Optional[float]:
    """Mean sequencing coverage across panel targets (optionally one chromosome)."""
    if target_coverage_df is None or target_coverage_df.empty:
        return None
    if "coverage" not in target_coverage_df.columns:
        return None
    subset = target_coverage_df
    if contig is not None and "chrom" in target_coverage_df.columns:
        chrom_subset = target_coverage_df[target_coverage_df["chrom"] == contig]
        if not chrom_subset.empty:
            subset = chrom_subset
    vals = np.asarray(subset["coverage"], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _add_genome_panel_coverage_points(
    ax_cnv,
    panel_points: List[Dict[str, Any]],
    x_max_bp: float,
    *,
    y_min: float,
    y_max: float,
    band_lo: Optional[float] = None,
    band_hi: Optional[float] = None,
    label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    rotated: bool = True,
    panel_width_pt: float = CNV_DEFAULT_PANEL_WIDTH_PT,
    panel_height_pt: float = CNV_DEFAULT_PANEL_HEIGHT_PT,
    reference_levels: Sequence[float] = (),
) -> bool:
    """Plot panel gene coverage markers on the genome-wide summary plot."""
    return _add_panel_coverage_points(
        ax_cnv,
        panel_points,
        x_max_bp,
        x_key="position_bp",
        min_x_spacing=1_800_000.0,
        y_min=y_min,
        y_max=y_max,
        band_lo=band_lo,
        band_hi=band_hi,
        label_size=label_size,
        rotated=rotated,
        panel_width_pt=panel_width_pt,
        panel_height_pt=panel_height_pt,
        reference_levels=reference_levels,
    )


def _add_chromosome_panel_coverage_overlay(
    ax_cnv,
    panel_points: List[Dict[str, Any]],
    x_max_mb: float,
    *,
    y_min: float,
    y_max: float,
    band_lo: Optional[float] = None,
    band_hi: Optional[float] = None,
    label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    rotated: bool = True,
    panel_width_pt: float = CNV_DEFAULT_PANEL_WIDTH_PT,
    panel_height_pt: float = CNV_DEFAULT_PANEL_HEIGHT_PT,
    reference_levels: Sequence[float] = (),
) -> bool:
    """Add coverage-normalised panel markers on the shared chromosome CNV axis."""
    if not panel_points:
        return False
    return _add_panel_coverage_points(
        ax_cnv,
        panel_points,
        float(ax_cnv.get_xlim()[1]) if ax_cnv.get_xlim()[1] > 0 else x_max_mb,
        x_key="mid_mb",
        min_x_spacing=1.8,
        y_min=y_min,
        y_max=y_max,
        band_lo=band_lo,
        band_hi=band_hi,
        label_size=label_size,
        rotated=rotated,
        panel_width_pt=panel_width_pt,
        panel_height_pt=panel_height_pt,
        reference_levels=reference_levels,
    )


def _collect_chromosome_significant_panel_points(
    panel_genes_df: Optional[pd.DataFrame],
    contig: str,
    values_array: np.ndarray,
    analysis_bin_width: int,
    regions: List[Dict[str, Any]],
    target_coverage_df: Optional[pd.DataFrame],
    *,
    use_log2: bool,
    baseline: float,
    sex_estimate: str = "Unknown",
    configured_genes: Sequence[str] = (),
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    clinical_trial_genes: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Collect panel gene markers for one chromosome (configured genes or cut-off crossings)."""
    panel_points = _collect_panel_gene_points(
        panel_genes_df,
        contig,
        values_array,
        analysis_bin_width,
        use_max_abs=use_log2,
        target_coverage_df=target_coverage_df,
        cutoff_override=cutoff_override,
    )
    selected = _select_panel_coverage_points(
        panel_points,
        regions,
        use_log2=use_log2,
        configured_genes=configured_genes,
        chromosome=contig,
        sex_estimate=sex_estimate,
        baseline=baseline,
        cutoff_override=cutoff_override,
        outliers_only=outliers_only,
        clinical_trial_genes=clinical_trial_genes,
    )
    return _attach_normalised_coverage(
        selected,
        # Use whole-panel mean coverage so gene markers match the genome summary.
        mean_cov=_mean_target_coverage(target_coverage_df),
        scale_mean_cnv=baseline,
        use_log2=use_log2,
    )


def _collect_genome_significant_panel_points(
    panel_genes_df: Optional[pd.DataFrame],
    cnv_source: Dict[str, np.ndarray],
    ordered_contigs: List[str],
    chrom_start_offsets: Dict[str, float],
    analysis_bin_width: int,
    significant_regions: Optional[Dict[str, List[Dict[str, Any]]]],
    target_coverage_df: Optional[pd.DataFrame],
    *,
    use_log2: bool,
    scale_mean_cnv: float,
    sex_estimate: str = "Unknown",
    configured_genes: Sequence[str] = (),
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    clinical_trial_genes: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Collect panel gene markers with genome-wide bp positions."""
    if panel_genes_df is None or panel_genes_df.empty:
        return []

    significant_regions = significant_regions or {}
    genome_points: List[Dict[str, Any]] = []
    mean_cov = _mean_target_coverage(target_coverage_df)

    for contig in ordered_contigs:
        if contig not in cnv_source:
            continue
        values_array = np.asarray(cnv_source[contig], dtype=float)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) == 0:
            continue

        regions = significant_regions.get(contig, [])
        panel_points = _collect_panel_gene_points(
            panel_genes_df,
            contig,
            values_array,
            analysis_bin_width,
            use_max_abs=use_log2,
            target_coverage_df=target_coverage_df,
            cutoff_override=cutoff_override,
        )

        chrom_offset = float(chrom_start_offsets.get(contig, 0.0))
        for point in _select_panel_coverage_points(
            panel_points,
            regions,
            use_log2=use_log2,
            configured_genes=configured_genes,
            chromosome=contig,
            sex_estimate=sex_estimate,
            baseline=scale_mean_cnv,
            cutoff_override=cutoff_override,
            outliers_only=outliers_only,
            clinical_trial_genes=clinical_trial_genes,
        ):
            genome_points.append(
                {
                    **point,
                    "position_bp": chrom_offset + float(point["mid_mb"]) * 1_000_000,
                }
            )

    return _attach_normalised_coverage(
        genome_points,
        mean_cov=mean_cov,
        scale_mean_cnv=scale_mean_cnv,
        use_log2=use_log2,
    )


_CNV_REFERENCE_LINE_STYLE = {
    "baseline": {"linestyle": "-", "linewidth": 1.3, "alpha": 0.9},
    # Calling cut-offs stay dashed so they read as guides, but dark and thick
    # enough to be unmissable: these are the lines gains and losses are read against.
    "threshold": {"linestyle": (0, (6, 3)), "linewidth": 1.3, "alpha": 0.95},
    "ploidy": {"linestyle": ":", "linewidth": 0.8, "alpha": 0.6},
}


def _add_clinical_trial_legend(
    fig,
    coverage_points: Sequence[Dict[str, Any]],
    *,
    corner: str = "bottom",
) -> bool:
    """Footnote naming the purple convention, when a trial target is on the plot.

    Drawn on the figure rather than as report body text so it travels with the
    exported PDFs too, and only when a purple gene is actually shown.

    ``corner`` selects the bottom or top edge. The per-chromosome figures in the
    report are only about 2in tall, where the bottom edge already carries the
    x-axis label and the two collide; the top edge is free beside the centred
    title.
    """
    if not any(point.get("clinical_trial") for point in coverage_points):
        return False
    from robin.gui.plotting_preferences import CNV_CLINICAL_TRIAL_LEGEND

    at_top = str(corner) == "top"
    fig.text(
        0.995,
        0.995 if at_top else 0.005,
        CNV_CLINICAL_TRIAL_LEGEND,
        ha="right",
        va="top" if at_top else "bottom",
        fontsize=CNV_FONT["annotation"],
        color=CNV_COLORS["plot_trial"],
        fontproperties=_CNV_FONT_REGULAR,
    )
    return True


def _drawn_reference_levels(
    *,
    use_log: bool,
    y_min: float,
    y_max: float,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    cutoff_override: Optional[float] = None,
) -> List[float]:
    """Y values of the guides drawn on a panel, so labels can avoid them."""
    gain_threshold, loss_threshold = (None, None)
    if use_log:
        gain_threshold, loss_threshold = _cnv_cutoff_thresholds(
            chromosome, sex_estimate, cutoff_override
        )
    return [
        value
        for value, _kind in cnv_reference_levels(
            use_log=use_log,
            gain_threshold=gain_threshold,
            loss_threshold=loss_threshold,
            y_lo=y_min,
            y_hi=y_max,
        )
    ]


def _add_cnv_reference_guides(
    ax,
    x_max: float,
    *,
    use_log: bool,
    y_min: float,
    y_max: float,
    chromosome: str = "chr1",
    sex_estimate: str = "Unknown",
    annotate: bool = True,
    cutoff_override: Optional[float] = None,
) -> None:
    """Draw the no-change baseline plus its supporting guides.

    Log2 panels get the calling gain/loss cut-offs as faint dashed lines, ploidy
    panels get the integer copy-number guides; both anchor the eye to the level a
    point is being judged against.
    """
    gain_threshold: Optional[float] = None
    loss_threshold: Optional[float] = None
    if use_log:
        gain_threshold, loss_threshold = _cnv_cutoff_thresholds(
            chromosome, sex_estimate, cutoff_override
        )

    levels = cnv_reference_levels(
        use_log=use_log,
        gain_threshold=gain_threshold,
        loss_threshold=loss_threshold,
        y_lo=y_min,
        y_hi=y_max,
    )
    for value, kind in levels:
        style = _CNV_REFERENCE_LINE_STYLE.get(kind, _CNV_REFERENCE_LINE_STYLE["ploidy"])
        color = {
            "baseline": CNV_COLORS["diploid"],
            "threshold": CNV_COLORS["cutoff"],
        }.get(kind, CNV_COLORS["reference"])
        ax.axhline(value, color=color, zorder=1, **style)
        if not annotate:
            continue
        if kind == "threshold":
            # Name the cut-off at the right-hand end of its own line.
            ax.text(
                x_max * 0.995,
                value,
                f"{value:+g}",
                fontsize=CNV_FONT["annotation"],
                color=color,
                fontweight="bold",
                va="bottom" if value > 0 else "top",
                ha="right",
                zorder=2,
            )
            continue
        ax.text(
            x_max * 0.008,
            value,
            f"{value:g}",
            fontsize=CNV_FONT["annotation"],
            color=color,
            va="center",
            ha="left",
            zorder=1,
        )


def _draw_region_bracket(
    ax,
    start_mb: float,
    end_mb: float,
    y_level: float,
    color: str,
    height_frac: float = 0.04,
) -> None:
    """Draw a horizontal genomic bracket."""
    y_top = y_level
    y_drop = y_level - height_frac
    ax.plot(
        [start_mb, start_mb, end_mb, end_mb],
        [y_drop, y_top, y_top, y_drop],
        color=color,
        linewidth=1.1,
        solid_capstyle="butt",
        zorder=5,
        clip_on=False,
    )


def _apply_cnv_axes_style(ax, *, xlabel: str, ylabel: str, title: Optional[str] = None) -> None:
    """Apply consistent seaborn-inspired styling to a CNV axes."""
    _setup_cnv_fonts()
    ax.set_facecolor("white")
    ax.set_xlabel(
        xlabel,
        fontsize=CNV_FONT["axis"],
        color=CNV_TEXT["primary"],
        labelpad=10,
        fontproperties=_CNV_FONT_REGULAR,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=CNV_FONT["axis"],
        color=CNV_TEXT["primary"],
        labelpad=10,
        fontproperties=_CNV_FONT_REGULAR,
    )
    if title:
        ax.set_title(
            title,
            fontsize=CNV_FONT["title"],
            color=CNV_TEXT["primary"],
            pad=10,
            fontproperties=_CNV_FONT_BOLD,
        )
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, axis="y", color=MODERN_COLORS["grid"], linestyle="--", linewidth=0.4, alpha=0.55)
    ax.grid(False, axis="x")
    ax.tick_params(colors=CNV_TEXT["primary"], labelsize=CNV_FONT["tick"])


def _apply_cnv_genome_overview_axes(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    x_max_bp: float,
    use_log: bool = False,
) -> None:
    """Genome-wide CNV panel: y-axis at x=0, no bottom axis line, no genomic tick labels."""
    _setup_cnv_fonts()
    ax.set_facecolor("white")
    ax.set_xlim(0, x_max_bp)
    ax.margins(x=0)
    ax.set_xlabel(
        xlabel,
        fontsize=CNV_FONT["axis"],
        color=CNV_TEXT["primary"],
        labelpad=18,
        fontproperties=_CNV_FONT_REGULAR,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=CNV_FONT["axis"],
        color=CNV_TEXT["primary"],
        labelpad=10,
        fontproperties=_CNV_FONT_REGULAR,
    )
    if title:
        ax.set_title(
            title,
            fontsize=CNV_FONT["title"],
            color=CNV_TEXT["primary"],
            pad=10,
            fontproperties=_CNV_FONT_BOLD,
        )
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("none")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.grid(True, axis="y", color=MODERN_COLORS["grid"], linestyle="--", linewidth=0.4, alpha=0.55)
    ax.grid(False, axis="x")
    _apply_cnv_y_tick_ladder(ax, use_log=use_log)


def _chromosome_cnv_dataframe(positions_mb, values) -> pd.DataFrame:
    return pd.DataFrame(
        {"position_mb": positions_mb, "ploidy": pd.Series(values, dtype=float)}
    )


def _add_cnv_reference_lines(ax, mean_cnv: float, std_cnv: float, y_min: float, y_max: float) -> None:
    """Genome-wide reference guides (mean plus optional spread)."""
    ax.axhline(y=mean_cnv, color=CNV_COLORS["reference"], linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
    for offset in (std_cnv, 2 * std_cnv):
        if y_min < mean_cnv + offset <= y_max:
            ax.axhline(y=mean_cnv + offset, color=CNV_COLORS["reference"], linestyle=":", linewidth=0.6, alpha=0.4, zorder=1)
        if y_min <= mean_cnv - offset < y_max:
            ax.axhline(y=mean_cnv - offset, color=CNV_COLORS["reference"], linestyle=":", linewidth=0.6, alpha=0.4, zorder=1)


def _log2_linear_axis_limits(
    log2_values: np.ndarray,
    *,
    percentile: float = 97.5,
    min_span: float = CNV_LOG2_MIN_AXIS_SPAN,
    max_span: float = CNV_LOG2_MAX_AXIS_SPAN,
    pad: float = 0.08,
) -> Tuple[float, float]:
    """Data-driven symmetric linear limits for log2 ratio plots.

    The floor is deliberately tight so the axis hugs the profile; a wide fixed
    minimum is what flattens real gains and losses into the middle of the panel.
    """
    vals = np.asarray(log2_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -min_span, min_span
    span = float(np.percentile(np.abs(vals), percentile)) + pad
    span = max(span, min_span)
    span = min(span, max_span)
    return -span, span


def _cnv_cutoff_thresholds(
    chromosome: str,
    sex_estimate: str,
    cutoff_override: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Gain/loss cut-off in force, honouring a user cut-off override."""
    if cutoff_override is not None:
        magnitude = abs(float(cutoff_override))
        return magnitude, -magnitude
    from robin.classification_config import get_cnv_thresholds

    try:
        return get_cnv_thresholds(chromosome or "chr1", sex_estimate or "Unknown")
    except Exception:
        logger.debug("Could not resolve CNV thresholds for %s", chromosome)
        return None, None


def _cnv_plot_point_state(
    value: float,
    chromosome: str,
    sex_estimate: str,
    cutoff_override: Optional[float] = None,
) -> str:
    """Classify a log2 CNV value as gain, loss, or neutral using the cut-off in force."""
    if not np.isfinite(value):
        return "neutral"
    gain_thr, loss_thr = _cnv_cutoff_thresholds(
        chromosome, sex_estimate, cutoff_override
    )
    if gain_thr is None or loss_thr is None:
        return "neutral"
    if value > gain_thr:
        return "gain"
    if value < loss_thr:
        return "loss"
    return "neutral"


def _add_centromere_line(
    ax, positions, *, linewidth: float = CNV_CENTROMERE_WIDTH
) -> None:
    """Dashed verticals separating the p and q arm of each chromosome."""
    for position in positions:
        if position is None or not np.isfinite(position):
            continue
        ax.axvline(
            float(position),
            color=CNV_COLORS["centromere"],
            linestyle=(0, (4, 4)),
            linewidth=linewidth,
            alpha=0.9,
            # Behind the bins and the segment line.
            zorder=0,
        )


def _centromere_boundaries() -> Dict[str, int]:
    """p/q boundary per chromosome, empty when the resource is unavailable."""
    try:
        from robin.analysis.cnv_regional import load_centromere_boundaries

        return load_centromere_boundaries()
    except Exception:
        logger.debug("Could not load centromere boundaries", exc_info=True)
        return {}


def _full_range_y_limits(
    values,
    *,
    use_log: bool,
    pad_frac: float = 0.06,
) -> Tuple[float, float]:
    """Limits that show every bin at its true value, with no percentile clipping.

    The ordinary limit helpers deliberately clip to a percentile and cap the span,
    which is right for the main view but defeats the purpose of a full-range page:
    a homozygous deletion near log2 -6 would still fall off the axis.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    baseline = 0.0 if use_log else CNV_PLOIDY_BASELINE
    if arr.size == 0:
        return (baseline - 1.0, baseline + 1.0)
    lo = float(min(arr.min(), baseline))
    hi = float(max(arr.max(), baseline))
    span = max(hi - lo, 1e-6)
    lo -= span * pad_frac
    hi += span * pad_frac
    if not use_log:
        lo = max(0.0, lo)
    return lo, hi


def _count_offscale_bins(
    values,
    axis_span: Optional[float],
    *,
    use_log: bool,
) -> int:
    """Number of bins a fixed axis window cannot show."""
    if not axis_span:
        return 0
    lo, hi = cnv_chromosome_axis_window(use_log=use_log, span_log2=float(axis_span))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0
    return int(np.count_nonzero((arr > hi) | (arr < lo)))


def _format_axis_span(axis_span: Optional[float], *, use_log: bool) -> str:
    """Human-readable description of a fixed axis window, for captions."""
    if not axis_span:
        return "the plotted range"
    if use_log:
        return f"±{float(axis_span):g} log2"
    lo, hi = cnv_chromosome_axis_window(use_log=False, span_log2=float(axis_span))
    return f"{lo:.2g}–{hi:.2g} copies"


def _mark_offscale_bins(ax, x_values, y_values, y_min: float, y_max: float) -> int:
    """Flag bins outside the fixed window with a marker at the panel edge.

    The per-chromosome window is fixed so pages can be compared, which means a
    deep event can fall outside it. Flagging keeps that visible instead of the
    point simply disappearing.
    """
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    if xs.size != ys.size or xs.size == 0:
        return 0
    flagged = 0
    for mask, edge, marker, color in (
        (ys > y_max, y_max, "^", CNV_COLORS["plot_gain"]),
        (ys < y_min, y_min, "v", CNV_COLORS["plot_loss"]),
    ):
        mask = mask & np.isfinite(ys)
        if not mask.any():
            continue
        flagged += int(mask.sum())
        ax.plot(
            xs[mask],
            np.full(int(mask.sum()), edge),
            linestyle="none",
            marker=marker,
            markersize=3.5,
            color=color,
            alpha=0.9,
            clip_on=False,
            zorder=7,
        )
    return flagged


def _arm_levels_from_calling_track(
    ploidy_cnv: Optional[Dict[str, Any]],
    analysis_bin_width: Optional[int],
    sex_estimate: str,
) -> Dict[str, Dict[str, float]]:
    """Arm means taken from the same track the arm/whole-chromosome table calls on.

    ``analyze_chromosome_arms`` is called directly, rather than the arm mean being
    recomputed here, so the figure shows the number the table used rather than a
    near neighbour of it. Deriving it from the plotted points moved 13 of 561 arms
    across the cut-off; splitting the calling track at the centromere instead of
    at the cytoband arm boundary still moved 2.
    """
    levels: Dict[str, Dict[str, float]] = {}
    if not ploidy_cnv or not analysis_bin_width:
        return levels
    try:
        from robin.analysis.cnv_analysis import prepare_cnv_calling_track
        from robin.analysis.cnv_classification import analyze_chromosome_arms
        from robin.analysis.cnv_regional import load_cytobands_bed

        calling, calling_bin = prepare_cnv_calling_track(
            ploidy_cnv, int(analysis_bin_width), sex_estimate
        )
        cytobands = load_cytobands_bed()
    except Exception:
        logger.debug("Could not build arm levels for the genome figure", exc_info=True)
        return levels
    if cytobands is None or cytobands.empty:
        return levels

    for contig in calling or {}:
        try:
            p_mean, q_mean, *_ = analyze_chromosome_arms(
                calling, str(contig), calling_bin, sex_estimate, cytobands
            )
        except Exception:
            logger.debug("Could not resolve arm means for %s", contig, exc_info=True)
            continue
        arms = {}
        if p_mean is not None and np.isfinite(p_mean):
            arms["p"] = float(p_mean)
        if q_mean is not None and np.isfinite(q_mean):
            arms["q"] = float(q_mean)
        if arms:
            levels[str(contig)] = arms
    return levels


def _centromere_split(contig: Optional[str], scale: float) -> List[float]:
    """Centromere position for ``contig`` in the plot's x units, or empty.

    ``scale`` converts base pairs to the axis units (1 for bp, 1e6 for Mb).
    """
    if not contig:
        return []
    centre = _centromere_boundaries().get(str(contig))
    if not centre:
        return []
    return [float(centre) / float(scale)]


def _add_cnv_trend_line(
    ax,
    x_values,
    y_values,
    *,
    linewidth: float = 1.4,
    split_at: Sequence[float] = (),
) -> None:
    """Overlay the segment level as detached horizontal bars, as on array CNV plots.

    Drawn as separate bars rather than a connected step: a step line has to riser
    between levels, so a short unmappable run forming its own segment shows up as a
    full-height spike off the line.
    """
    spans = cnv_segment_spans(x_values, y_values, split_at=split_at)
    if not spans:
        return
    ax.hlines(
        [level for _start, _end, level in spans],
        [start for start, _end, _level in spans],
        [end for _start, end, _level in spans],
        colors=CNV_COLORS["median"],
        linewidth=linewidth,
        alpha=0.95,
        zorder=4,
    )


def _scatter_cnv_chromosome_points(
    ax,
    cnv_df: pd.DataFrame,
    *,
    color_by_state: bool,
    point_size: float = CNV_CHROMOSOME_SCATTER_SIZE,
    show_trend: bool = True,
    contig: Optional[str] = None,
) -> None:
    """Scatter per-chromosome CNV points, optionally coloured by threshold state."""
    if color_by_state and "state" in cnv_df.columns:
        for state, color, zorder in (
            ("neutral", CNV_COLORS["plot_neutral"], 1),
            ("loss", CNV_COLORS["plot_loss"], 2),
            ("gain", CNV_COLORS["plot_gain"], 2),
        ):
            subset = cnv_df[cnv_df["state"] == state]
            if subset.empty:
                continue
            ax.scatter(
                subset["position_mb"],
                subset["ploidy"],
                c=color,
                s=point_size,
                alpha=(
                    CNV_POINT_ALPHA_NEUTRAL
                    if state == "neutral"
                    else CNV_POINT_ALPHA_CALLED
                ),
                linewidth=0,
                edgecolors="none",
                rasterized=True,
                zorder=zorder,
            )
        if show_trend:
            _add_cnv_trend_line(
                ax,
                cnv_df["position_mb"],
                cnv_df["ploidy"],
                split_at=_centromere_split(contig, 1_000_000.0),
            )
        return

    _plot_cnv_track(
        ax,
        cnv_df,
        float(cnv_df["position_mb"].max()),
        point_size=point_size,
        show_trend=show_trend,
        contig=contig,
    )


def _scatter_cnv_genome_points(
    ax,
    df: pd.DataFrame,
    *,
    color_by_state: bool,
    show_trend: bool = True,
    arm_levels: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Scatter genome-wide CNV points, optionally coloured by threshold state."""
    if show_trend and "contig" in df.columns:
        # One trace per chromosome so the trend never bridges a chromosome boundary.
        for _contig, subset in df.groupby("contig", sort=False):
            ordered = subset.sort_values("position_bp")
            # One level per arm, at the arm mean, so the genome-wide figure and
            # the arm / whole-chromosome table cannot disagree about the same
            # arm. Finer segmentation here tracks coverage structure rather than
            # copy number; the per-chromosome pages keep the detailed line.
            spans = cnv_arm_mean_spans(
                np.asarray(ordered["position_bp"], dtype=float),
                np.asarray(ordered["ploidy"], dtype=float),
                split_at=_centromere_split(_contig, 1.0),
            )
            # Replace each arm's level with the value the table called on.
            called = (arm_levels or {}).get(str(_contig))
            if called and spans:
                names = ["p", "q"] if len(spans) == 2 else ["q"]
                spans = [
                    (s, e, called.get(name, level))
                    for (s, e, level), name in zip(spans, names)
                ]
            if spans:
                ax.hlines(
                    [level for _s, _e, level in spans],
                    [s for s, _e, _l in spans],
                    [e for _s, e, _l in spans],
                    colors=CNV_COLORS["median"],
                    linewidth=1.1,
                    zorder=3,
                )
    if color_by_state and "state" in df.columns:
        for state, color, zorder in (
            ("neutral", CNV_COLORS["plot_neutral"], 1),
            ("loss", CNV_COLORS["plot_loss"], 2),
            ("gain", CNV_COLORS["plot_gain"], 2),
        ):
            subset = df[df["state"] == state]
            if subset.empty:
                continue
            ax.scatter(
                subset["position_bp"],
                subset["ploidy"],
                c=color,
                s=4,
                alpha=(
                    CNV_POINT_ALPHA_NEUTRAL
                    if state == "neutral"
                    else CNV_POINT_ALPHA_CALLED
                ),
                linewidth=0,
                edgecolors="none",
                rasterized=True,
                zorder=zorder,
            )
        return

    palette = sns.color_palette("muted", n_colors=max(df["contig"].nunique(), 3))
    contig_palette = dict(zip(sorted(df["contig"].unique()), palette))
    for contig, color in contig_palette.items():
        subset = df[df["contig"] == contig]
        ax.scatter(
            subset["position_bp"],
            subset["ploidy"],
            c=[color],
            s=4,
            alpha=CNV_POINT_ALPHA_DEFAULT,
            linewidth=0,
            edgecolors="none",
            rasterized=True,
        )


def target_distribution_plot(df):
    """
    Creates a target distribution plot.

    Args:
        df (pd.DataFrame): DataFrame containing the target distribution data.

    Returns:
        io.BytesIO: Buffer containing the plot image.
    """
    set_modern_style()

    df["chrom"] = pd.Categorical(
        df["chrom"], categories=natsort.natsorted(df["chrom"].unique()), ordered=True
    )
    df = df.sort_values("chrom")

    # Generate the plot
    plt.figure(figsize=(16, 8))
    boxplot = sns.boxplot(
        x="chrom",
        y="coverage",
        data=df,
        color=MODERN_COLORS["accent"],
        flierprops={"marker": "o", "markerfacecolor": MODERN_COLORS["primary"]},
    )

    plt.title(
        "Distribution of Target Coverage on Each Chromosome",
        fontsize=12,
        color=MODERN_COLORS["primary"],
        pad=20,
    )
    plt.xlabel("Chromosome", color=MODERN_COLORS["primary"])
    plt.ylabel("Coverage", color=MODERN_COLORS["primary"])
    plt.xticks(rotation=45)

    # Identify and annotate outliers
    def annotate_outliers(df, boxplot):
        # Calculate quartiles and IQR
        for chrom in df["chrom"].unique():
            chrom_data = df[df["chrom"] == chrom]
            Q1 = chrom_data["coverage"].quantile(0.25)
            Q3 = chrom_data["coverage"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Find outliers
            outliers = chrom_data[
                (chrom_data["coverage"] < lower_bound)
                | (chrom_data["coverage"] > upper_bound)
            ]

            for idx in outliers.index:
                outlier = outliers.loc[idx]
                boxplot.annotate(
                    outlier["name"],
                    xy=(df.loc[idx, "chrom"], df.loc[idx, "coverage"]),
                    xytext=(10, 10),  # Offset the text more from the point
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="red",
                )  # Increase font size

    annotate_outliers(df, boxplot)

    plt.tight_layout()
    # Save the plot as a JPG file
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi=300)
    buf.seek(0)
    return buf


def _create_empty_cnv_buffer():
    """Create a minimal valid JPEG buffer for empty CNV plots."""
    try:
        plt.figure(figsize=(16, 4))
        plt.text(0.5, 0.5, "No CNV data available", 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, color='gray')
        plt.title("Copy Number Changes")
        plt.axis('off')
        
        buf = io.BytesIO()
        fig = plt.gcf()
        plt.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        
        # Validate buffer contains data
        if buf.getvalue():
            return buf
        else:
            raise ValueError("Empty buffer")
    except Exception:
        # If even this fails, create a minimal valid JPEG programmatically
        try:
            # Create a minimal 1x1 white JPEG
            from PIL import Image as PILImage
            img = PILImage.new('RGB', (1, 1), color='white')
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            buf.seek(0)
            return buf
        except Exception:
            # Last resort: return a minimal valid JPEG binary directly
            # This is a valid 1x1 white JPEG
            jpeg_bytes = bytes([
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
                0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
                0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
                0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
                0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
                0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
                0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
                0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
                0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x08, 0xFF, 0xC4, 0x00, 0x14, 0x10, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
                0xD2, 0xCF, 0x20, 0xFF, 0xD9
            ])
            buf = io.BytesIO(jpeg_bytes)
            buf.seek(0)
            return buf


def build_CNV_genome_figure(
    result,
    cnv_dict,
    normalized_cnv=None,
    *,
    use_normalized_difference: bool = False,
    plot_bin_width: Optional[int] = None,
    sex_estimate: str = "Unknown",
    panel_genes_df: Optional[pd.DataFrame] = None,
    target_coverage_df: Optional[pd.DataFrame] = None,
    significant_regions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    reference_contig_scope: Optional[str] = None,
    configured_genes: Sequence[str] = (),
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    show_trend: bool = True,
    gene_label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    gene_labels_rotated: bool = True,
    clinical_trial_genes: Sequence[str] = (),
    fig_width: float = CNV_GENOME_FIG_WIDTH,
    fig_height: Optional[float] = None,
    fixed_axis_log2: Optional[float] = None,
):
    """
    Builds the genome-wide CNV matplotlib figure (returns the Figure, or None).

    Kept separate from the buffer wrappers so the same figure can be written as a
    report JPEG or exported as a vector PDF.

    Args:
        result (Result): CNV result object.
        cnv_dict (dict): Dictionary containing CNV data.
        normalized_cnv (dict, optional): Per-chromosome log2(ploidy / expected copy number)
            values derived from the absolute CNV track.
        use_normalized_difference (bool): When True, plot log2(ploidy / expected) instead of
            absolute ploidy for the genome-wide summary chart.
        plot_bin_width (int, optional): Display bin width in bp for genome-wide plot.
            Defaults to 1 Mb. Values below the analysis bin width are ignored.
        sex_estimate: Sex estimate for per-chromosome log2 calling thresholds.
        configured_genes: Optional ``[cnv].genes`` list. When non-empty, those panel
            targets are marked; otherwise >3 SD CNV outliers are used.
        fixed_axis_log2: Symmetric log2 half-span for the genome-wide axis. None
            (the default) fits the axis to the data.

    Returns:
        matplotlib.figure.Figure | None: The figure, or None when it cannot be built.
    """
    try:
        from robin.analysis.cnv_analysis import (
            downsample_cnv_for_plot,
            resolve_cnv_plot_bin_width,
            resolve_cnv_report_genome_plot_bin_width,
        )
        from robin.gui.plotting_preferences import (
            CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
            CNV_REPORT_SCALE_PLOIDY,
            cnv_report_genome_ylabel_mathtext,
        )

        set_modern_style()

        cnv_source = result.cnv if hasattr(result, "cnv") else None
        plot_normalized = use_normalized_difference and normalized_cnv
        if use_normalized_difference and not normalized_cnv:
            logger.warning(
                "Log2 CNV summary requested but no relative data available; "
                "falling back to absolute ploidy"
            )
        if plot_normalized:
            cnv_source = normalized_cnv
        scale = (
            CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
            if plot_normalized
            else CNV_REPORT_SCALE_PLOIDY
        )

        # Check if result has CNV data
        if not cnv_source:
            logger.warning("No CNV data available for plotting")
            return None

        # Prepare data for plotting
        analysis_bin_width = int(cnv_dict["bin_width"])
        # An explicit width (the GUI selector) is honoured as given; only the
        # default is derived from the analysis bin, since that varies with depth.
        display_bin_width = resolve_cnv_plot_bin_width(
            analysis_bin_width,
            plot_bin_width
            or resolve_cnv_report_genome_plot_bin_width(analysis_bin_width),
        )

        plot_rows = []
        offset_bp = 0.0
        contig_centers = {}
        contig_boundaries = []
        chrom_start_offsets: Dict[str, float] = {}
        log2_values_for_limits: List[float] = []
        ordered_contigs = [
            contig
            for contig in natsort.natsorted(cnv_source.keys())
            if is_visible_contig(contig, reference_contig_scope)
        ]

        for contig in ordered_contigs:
            values = np.asarray(cnv_source[contig], dtype=float)
            chrom_start_offsets[contig] = offset_bp
            chrom_span_bp = len(values) * analysis_bin_width
            x_local, plot_values = downsample_cnv_for_plot(
                values, analysis_bin_width, display_bin_width
            )
            x_global = offset_bp + x_local
            for position_bp, y_value in zip(x_global, plot_values):
                y_value = float(y_value)
                if plot_normalized and not np.isfinite(y_value):
                    continue
                if plot_normalized:
                    log2_values_for_limits.append(y_value)
                row = {
                    "contig": contig,
                    "position_bp": float(position_bp),
                    "ploidy": y_value,
                }
                if plot_normalized:
                    row["state"] = _cnv_plot_point_state(
                        y_value, contig, sex_estimate, cutoff_override
                    )
                plot_rows.append(row)
            contig_centers[contig] = offset_bp + (chrom_span_bp / 2)
            offset_bp += chrom_span_bp
            contig_boundaries.append(offset_bp)

        if not plot_rows:
            logger.warning("No plot data available for CNV plot")
            return None

        df = pd.DataFrame(plot_rows)
        mean_value = float(df["ploidy"].mean())
        std_value = float(df["ploidy"].std())
        if plot_normalized:
            if fixed_axis_log2:
                # An explicit genome Y-range from the admin Plotting settings.
                # Defaults to None (fit the data), because unlike the
                # per-chromosome panels there is nothing to compare against.
                span = abs(float(fixed_axis_log2))
                y_min, y_max = -span, span
            else:
                y_min, y_max = _log2_linear_axis_limits(
                    np.asarray(log2_values_for_limits)
                )
        else:
            y_min = max(0.0, float(df["ploidy"].min()) - 0.25)
            y_max = max(mean_value + (4 * std_value), mean_value * 1.35, 2.5)

        # Rendered close to the size it is placed at, so gene labels are not
        # scaled down to illegibility on the page.
        width = float(fig_width)
        height = float(fig_height) if fig_height else width / CNV_GENOME_FIG_ASPECT
        fig, ax = plt.subplots(figsize=(width, height))
        # Matplotlib's default subplot margins are fractions of the figure, so
        # the panel used only ~77% of the canvas and left a wide empty strip on
        # the right. Reserve the axis furniture in absolute inches instead, so
        # the plot reaches the edge at any figure width.
        _fill_genome_canvas(fig, width, height)
        genome_panel_height_pt = float(
            fig.get_figheight() * ax.get_position().height * 72.0
        )
        genome_panel_width_pt = float(
            fig.get_figwidth() * ax.get_position().width * 72.0
        )
        genome_panel_points = _collect_genome_significant_panel_points(
            panel_genes_df,
            cnv_source,
            ordered_contigs,
            chrom_start_offsets,
            analysis_bin_width,
            significant_regions,
            target_coverage_df,
            use_log2=plot_normalized,
            scale_mean_cnv=mean_value,
            sex_estimate=sex_estimate,
            configured_genes=configured_genes,
            cutoff_override=cutoff_override,
            outliers_only=outliers_only,
            clinical_trial_genes=clinical_trial_genes,
        )
        has_lollipops = bool(genome_panel_points)
        if has_lollipops and not (plot_normalized and fixed_axis_log2):
            # A chosen window is honoured exactly; only a fitted one is widened
            # to reach its gene markers. Markers outside a chosen window are
            # flagged at the panel edge, as they are on the chromosome plots.
            y_min, y_max = _expand_ylim_for_coverage_points(
                y_min, y_max, genome_panel_points
            )

        _scatter_cnv_genome_points(
            ax,
            df,
            color_by_state=plot_normalized,
            show_trend=show_trend,
            arm_levels=_arm_levels_from_calling_track(
                getattr(result, "cnv", None), analysis_bin_width, sex_estimate
            ),
        )

        for boundary in contig_boundaries[:-1]:
            ax.axvline(
                boundary,
                color=CNV_COLORS["contig_boundary"],
                linewidth=CNV_CONTIG_BOUNDARY_WIDTH,
                linestyle="-",
                alpha=0.85,
                zorder=0,
            )

        centromeres = _centromere_boundaries()
        _add_centromere_line(
            ax,
            [
                chrom_start_offsets[contig] + centromeres[contig]
                for contig in ordered_contigs
                if contig in centromeres and contig in chrom_start_offsets
            ],
            linewidth=CNV_CENTROMERE_WIDTH,
        )

        # Data band before label gutters are reserved, so gene badges can be
        # parked outside it instead of over the bins.
        band_lo, band_hi = y_min, y_max
        if not (plot_normalized and fixed_axis_log2):
            # An explicitly chosen window is used as given; only a fitted one is
            # rounded out onto whole ticks.
            y_min, y_max = _snap_cnv_ylim(y_min, y_max, use_log=plot_normalized)

        _add_cnv_reference_guides(
            ax,
            offset_bp,
            use_log=plot_normalized,
            y_min=y_min,
            y_max=y_max,
            sex_estimate=sex_estimate,
            annotate=False,
            cutoff_override=cutoff_override,
        )
        if not plot_normalized:
            _add_cnv_reference_lines(ax, mean_value, std_value, y_min, y_max)

        # Anchored to the start of each chromosome rather than its centre, so a
        # name reads against the boundary that opens it — on a 24in panel a
        # centred name is a long way from either edge of its own chromosome.
        label_y = y_min + (y_max - y_min) * 0.03
        label_pad_bp = float(offset_bp) * CNV_CONTIG_LABEL_PAD_FRAC
        for contig, start_bp in chrom_start_offsets.items():
            ax.text(
                float(start_bp) + label_pad_bp,
                label_y,
                _chromosome_display_name(contig),
                fontsize=CNV_FONT["tick"],
                ha="left",
                va="bottom",
                rotation=0,
                color=CNV_TEXT["primary"],
                fontproperties=_CNV_FONT_REGULAR,
                clip_on=False,
            )

        ax.set_ylim(y_min, y_max)
        _apply_cnv_genome_overview_axes(
            ax,
            xlabel="Chromosome",
            ylabel=cnv_report_genome_ylabel_mathtext(scale),
            title="Copy number variation across chromosomes",
            x_max_bp=offset_bp,
            use_log=plot_normalized,
        )
        if has_lollipops:
            _add_genome_panel_coverage_points(
                ax,
                _clamp_coverage_points_to_band(genome_panel_points, band_lo, band_hi),
                offset_bp,
                y_min=y_min,
                y_max=y_max,
                band_lo=band_lo,
                band_hi=band_hi,
                label_size=gene_label_size,
                rotated=gene_labels_rotated,
                panel_width_pt=genome_panel_width_pt,
                panel_height_pt=genome_panel_height_pt,
                reference_levels=_drawn_reference_levels(
                    use_log=plot_normalized,
                    y_min=y_min,
                    y_max=y_max,
                    sex_estimate=sex_estimate,
                    cutoff_override=cutoff_override,
                ),
            )
            _add_clinical_trial_legend(fig, genome_panel_points)

        return fig
    except Exception as e:
        logger.error(f"Error creating CNV plot: {str(e)}")
        plt.close("all")
        return None


def create_CNV_plot(*args, **kwargs):
    """Genome-wide CNV summary as a JPEG buffer for the PDF report."""
    fig = build_CNV_genome_figure(*args, **kwargs)
    if fig is None:
        return _create_empty_cnv_buffer()
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="jpg", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        buf.seek(0)

        buf_data = buf.getvalue()
        if not buf_data:
            logger.warning("Empty buffer for CNV plot")
            return _create_empty_cnv_buffer()
        if len(buf_data) < 2 or buf_data[:2] != b"\xff\xd8":
            logger.warning("Invalid JPEG data for CNV plot")
            return _create_empty_cnv_buffer()

        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"Error rendering CNV plot: {str(e)}")
        plt.close("all")
        return _create_empty_cnv_buffer()


REPORTABLE_CHROMOSOMES = ["chr" + str(i) for i in range(0, 23)] + ["chrX", "chrY"]
PANEL_Y_EXPANSION_FACTOR = 6.0


def cnv_chromosome_fig_height_for_page(
    page_height_inch: float,
    *,
    plots_per_page: int = CNV_CHROMOSOME_PLOTS_PER_PAGE,
    spacer_pt: float = CNV_CHROMOSOME_PLOT_SPACER_PT,
    frame_padding_pt: float = CNV_REPORT_FRAME_PADDING_PT,
) -> float:
    """Matplotlib figure height so ``plots_per_page`` chromosome plots fit one PDF page.

    ``page_height_inch`` should be ``SimpleDocTemplate.height`` in inches. ReportLab
    frames reserve ``frame_padding_pt`` (default 12) of vertical space for padding.
    """
    usable_inch = page_height_inch - frame_padding_pt / 72.0
    spacer_inch = max(plots_per_page - 1, 0) * spacer_pt / 72.0
    return (usable_inch - spacer_inch) / plots_per_page


def cnv_chromosome_fig_width_for_page(
    page_width_inch: float,
    *,
    frame_padding_pt: float = CNV_REPORT_FRAME_PADDING_PT,
) -> float:
    """Matplotlib figure width that fits the printable ReportLab frame."""
    return page_width_inch - frame_padding_pt / 72.0


def _panel_target_label(gene_row) -> str:
    """Return the display label for a target-panel region."""
    for key in ("gene", "name", "target"):
        if key in gene_row.index and pd.notna(gene_row[key]):
            raw = str(gene_row[key]).strip()
            if raw and raw.lower() != "nan":
                return raw.split(",")[0].strip()
    return ""


def _coverage_for_panel_target(
    gene_row: pd.Series,
    contig: str,
    target_coverage_df: Optional[pd.DataFrame],
) -> Optional[float]:
    """Resolve target coverage for a panel BED interval."""
    if target_coverage_df is None or target_coverage_df.empty:
        return None
    chrom_matches = target_coverage_df[target_coverage_df["chrom"] == contig]
    if chrom_matches.empty:
        return None

    start_bp = float(gene_row["start_pos"])
    end_bp = float(gene_row["end_pos"])
    overlapping = chrom_matches[
        (chrom_matches["endpos"] >= start_bp) & (chrom_matches["startpos"] <= end_bp)
    ]
    if not overlapping.empty:
        return float(overlapping["coverage"].max())

    label = _panel_target_label(gene_row)
    if not label:
        return None
    for _, row in chrom_matches.iterrows():
        names = [name.strip() for name in str(row["name"]).split(",") if name.strip()]
        if label in names:
            return float(row["coverage"])
    return None


def _gene_call_cutoff(contig: str, cutoff_override: Optional[float]) -> float:
    """Cut-off a gene value is judged against, for the focal-rescue threshold."""
    if cutoff_override is not None:
        try:
            candidate = abs(float(cutoff_override))
            if np.isfinite(candidate) and candidate > 0:
                return candidate
        except (TypeError, ValueError):
            pass
    from robin.classification_config import get_cnv_thresholds

    return abs(get_cnv_thresholds(contig or "chr1", "Unknown")[0])


def _collect_panel_gene_points(
    panel_genes_df: Optional[pd.DataFrame],
    contig: str,
    values,
    bin_width: int,
    *,
    use_max_abs: bool = False,
    target_coverage_df: Optional[pd.DataFrame] = None,
    cutoff_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Collect panel target positions and a representative CNV per target."""
    if panel_genes_df is None or panel_genes_df.empty:
        return []

    values_array = np.array(values)
    genes = panel_genes_df[panel_genes_df["chrom"] == contig].sort_values("start_pos")
    points: List[Dict[str, Any]] = []

    for _, gene_row in genes.iterrows():
        label_text = _panel_target_label(gene_row)
        if not label_text:
            continue

        start_bp = float(gene_row["start_pos"])
        end_bp = float(gene_row["end_pos"])
        mid_bp = (start_bp + end_bp) / 2.0
        mid_mb = mid_bp / 1_000_000
        # Same window rule as the gene table: widened to a minimum number of
        # bins, so a target shorter than one bin is not judged on one bin.
        start_bin, stop_bin = gene_bin_window(
            int(start_bp // bin_width),
            int(end_bp // bin_width) + 1,
            len(values_array),
        )
        end_bin = stop_bin - 1
        if end_bin < start_bin:
            cnv_val = 0.0
        else:
            region_vals = values_array[start_bin : end_bin + 1]
            if len(region_vals):
                if use_max_abs:
                    # Averaged across the gene, with a focal event reported at
                    # its own depth. Taking the peak bin outright made a neutral
                    # gene spanning five bins call about half the time.
                    cnv_val = robust_gene_value(
                        region_vals, cutoff=_gene_call_cutoff(contig, cutoff_override)
                    )
                else:
                    cnv_val = float(np.nanmax(region_vals))
            else:
                cnv_val = 0.0

        coverage_val = _coverage_for_panel_target(gene_row, contig, target_coverage_df)

        points.append(
            {
                "label": label_text,
                "mid_mb": mid_mb,
                "cnv_val": cnv_val,
                "coverage_val": coverage_val,
            }
        )

    # A gene may have several panel intervals (ALK, GNAQ, MGMT and others have
    # two or three). They collapse to one marker per gene, and the winning
    # interval must be chosen the same way as the value within an interval:
    # by magnitude when use_max_abs is set, otherwise by signed maximum. Picking
    # the signed maximum here while extracting by magnitude above would let a
    # small gain in one interval discard a deep loss in another, flipping the
    # reported direction of the gene.
    def _is_stronger(candidate: float, incumbent: float) -> bool:
        if use_max_abs:
            return abs(candidate) > abs(incumbent)
        return candidate > incumbent

    merged: Dict[str, Dict[str, Any]] = {}
    for point in points:
        existing = merged.get(point["label"])
        if existing is None:
            merged[point["label"]] = dict(point)
            continue
        if _is_stronger(point["cnv_val"], existing["cnv_val"]):
            # Take the winning interval's position too, so the marker sits at
            # the locus the value actually came from.
            existing["cnv_val"] = point["cnv_val"]
            existing["mid_mb"] = point["mid_mb"]
        if point.get("coverage_val") is not None:
            prev = existing.get("coverage_val")
            existing["coverage_val"] = max(prev or 0.0, point["coverage_val"])

    return list(merged.values())


def _compute_cnv_y_limits(
    values_array: np.ndarray,
    panel_points: List[Dict[str, Any]],
) -> tuple[float, float, bool, float, float]:
    """
    Choose y-axis limits that preserve bulk CNV detail while surfacing
    amplified panel targets when feasible.
    """
    mean_cnv = float(np.mean(values_array))
    std_cnv = float(np.std(values_array))
    baseline_max = max(mean_cnv + (2 * std_cnv), mean_cnv * 1.4, 2.5)

    if not panel_points:
        return 0.0, baseline_max, False, mean_cnv, std_cnv

    panel_max = max(point["cnv_val"] for point in panel_points)
    if panel_max <= baseline_max * 1.05:
        return 0.0, baseline_max, False, mean_cnv, std_cnv

    if panel_max <= baseline_max * PANEL_Y_EXPANSION_FACTOR:
        return 0.0, panel_max * 1.08, False, mean_cnv, std_cnv

    return 0.0, baseline_max, True, mean_cnv, std_cnv


def _log2_chromosome_axis_limits(
    log2_values: np.ndarray,
    *,
    percentile: float = 97.5,
    min_span: float = CNV_LOG2_MIN_AXIS_SPAN,
    max_span: float = CNV_LOG2_MAX_AXIS_SPAN,
    pad: float = 0.08,
) -> Tuple[float, float]:
    """Log2 limits for one chromosome: data-driven, always containing zero.

    Unlike the genome-wide panel these are not forced symmetric. A chromosome
    carrying only a loss would otherwise spend its whole upper half on empty
    space, halving the resolution of the change actually being read.
    """
    vals = np.asarray(log2_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return -min_span / 2.0, min_span / 2.0
    lo = float(np.percentile(vals, 100.0 - percentile)) - pad
    hi = float(np.percentile(vals, percentile)) + pad
    # The no-change baseline must stay on the panel, with a little room each side.
    lo = min(lo, -min_span / 2.0)
    hi = max(hi, min_span / 2.0)
    lo = max(lo, -max_span)
    hi = min(hi, max_span)
    return lo, hi


def _compute_log2_y_limits(
    values_array: np.ndarray,
    panel_points: List[Dict[str, Any]],
) -> tuple[float, float, bool, float, float]:
    """Log2 ratio limits for per-chromosome difference plots."""
    mean_cnv = float(np.mean(values_array))
    std_cnv = float(np.std(values_array))
    y_min, y_max = _log2_chromosome_axis_limits(values_array)
    off_scale_mode = False
    if panel_points:
        panel_extreme = max(abs(point["cnv_val"]) for point in panel_points)
        if panel_extreme > y_max * 0.97:
            off_scale_mode = True
    return y_min, y_max, off_scale_mode, mean_cnv, std_cnv


def iter_CNV_chromosome_figures(
    result,
    cnv_dict,
    significant_regions=None,
    chromosomes: Optional[List[str]] = None,
    panel_genes_df: Optional[pd.DataFrame] = None,
    chromosome_status: Optional[Dict[str, str]] = None,
    normalized_cnv: Optional[Dict[str, np.ndarray]] = None,
    target_coverage_df: Optional[pd.DataFrame] = None,
    *,
    use_log2_ratio: bool = False,
    plot_bin_width: Optional[int] = None,
    sex_estimate: str = "Unknown",
    fig_height: Optional[float] = None,
    fig_width: Optional[float] = None,
    reference_contig_scope: Optional[str] = None,
    configured_genes: Sequence[str] = (),
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    show_trend: bool = True,
    fixed_axis_log2: Optional[float] = CNV_CHROMOSOME_AXIS_LOG2,
    gene_label_size: float = LOLLIPOP_LABEL_FONT_SIZE,
    gene_labels_rotated: bool = True,
    full_range_pages: bool = False,
    full_range_axis: bool = False,
    figure_legend: bool = True,
    clinical_trial_genes: Sequence[str] = (),
):
    """Yields ``(chromosome, Figure)`` for each per-chromosome CNV plot.

    Callers own the figures and must close them. Kept separate from the buffer
    wrapper so the same figures can be written as report JPEGs or as a vector PDF.

    Args:
        result (Result): CNV result object.
        cnv_dict (dict): Dictionary containing CNV data.
        significant_regions (dict): Dictionary mapping chromosomes to lists of significant regions.
        chromosomes (list, optional): Ordered chromosome names to plot.
        panel_genes_df (pd.DataFrame, optional): Target panel genes.
        chromosome_status (dict, optional): Status text keyed by chromosome.
        normalized_cnv (dict, optional): Per-chromosome log2(ploidy / expected) values.
        sex_estimate (str): Sample sex estimate for log2 threshold colouring.
        use_log2_ratio (bool): When True, plot log2(ploidy / expected) instead of absolute ploidy.
        plot_bin_width (int, optional): Display bin width in bp. Defaults to the
            sample analysis bin width (finest resolution available).
        fig_height (float, optional): Matplotlib figure height in inches. Defaults
            to a height that fits four plots per PDF page.
        fig_width (float, optional): Matplotlib figure width in inches. Defaults
            to ``CNV_CHROMOSOME_FIG_WIDTH``.
        configured_genes: Optional ``[cnv].genes`` list. When non-empty, those panel
            targets are marked; otherwise genes crossing the cut-off are used.
        fixed_axis_log2: Symmetric log2 half-span applied to every chromosome so the
            plots are comparable. Bins outside it are flagged at the panel edge.
            Pass None for a data-driven axis per chromosome.
        full_range_axis: When True the chromosome's own page is drawn on the true
            data range rather than a percentile fit, so a deep event is shown at
            its real depth instead of being clipped to an edge marker.
        figure_legend: When False the clinical-trial legend is not drawn on the
            figure. The report stacks four plots to a page and states it once in
            the body text instead, where it cannot collide with a title.
        full_range_pages: When True, a chromosome with bins outside the fixed
            window yields a second figure fitted to the data, so a deep event is
            never reduced to an edge marker with no magnitude. A chromosome may
            therefore yield twice, which is why callers that key results by
            chromosome leave this off.

    Yields:
        Tuple[str, matplotlib.figure.Figure]: Chromosome name and its figure.
    """
    chromosome_status = chromosome_status or {}
    try:
        from robin.analysis.cnv_analysis import (
            downsample_cnv_chromosome_track,
            resolve_cnv_plot_bin_width,
        )
        from robin.gui.plotting_preferences import cnv_report_genome_ylabel_mathtext

        set_modern_style()

        plot_log2 = use_log2_ratio and normalized_cnv
        if use_log2_ratio and not normalized_cnv:
            logger.warning(
                "Log2 per-chromosome CNV plots requested but no relative data available; "
                "falling back to absolute ploidy"
            )

        cnv_source = normalized_cnv if plot_log2 else (result.cnv if hasattr(result, "cnv") else None)
        if not cnv_source:
            logger.warning("No CNV data available for per-chromosome plotting")
            return

        if chromosomes is None:
            chromosomes = [
                contig
                for contig in natsort.natsorted(cnv_source.keys())
                if is_visible_contig(contig, reference_contig_scope)
            ]

        scale = "normalized_difference" if plot_log2 else "ploidy"
        ylabel = cnv_report_genome_ylabel_mathtext(scale)
        analysis_bin_width = int(cnv_dict["bin_width"])
        report_plot_bin_width = resolve_cnv_plot_bin_width(
            analysis_bin_width,
            plot_bin_width,
        )
        compact_layout = True
        total_height = fig_height or cnv_chromosome_fig_height_for_page(9.34)
        total_width = fig_width or CNV_CHROMOSOME_FIG_WIDTH

        for contig in chromosomes:
            if contig not in cnv_source:
                continue
            values = cnv_source[contig]

            values_array = np.asarray(values, dtype=float)
            finite_values = values_array[np.isfinite(values_array)]
            if len(finite_values) == 0:
                continue

            positions_mb, plot_values, x_max_mb = downsample_cnv_chromosome_track(
                values_array,
                analysis_bin_width,
                report_plot_bin_width,
            )
            plot_mask = np.isfinite(plot_values)
            positions_mb = positions_mb[plot_mask]
            plot_values = plot_values[plot_mask]
            if len(plot_values) == 0:
                continue
            # Axis-independent work, so a chromosome drawn twice does it once.
            if plot_log2:
                data_lo, data_hi, _, mean_cnv, std_cnv = _compute_log2_y_limits(
                    finite_values, [],
                )
            else:
                data_lo, data_hi, _, mean_cnv, std_cnv = _compute_cnv_y_limits(
                    finite_values, [],
                )
                data_lo = 0.0
            regions = (significant_regions or {}).get(contig, [])
            significant_panel_points = _collect_chromosome_significant_panel_points(
                panel_genes_df,
                contig,
                values_array,
                analysis_bin_width,
                regions,
                target_coverage_df,
                use_log2=plot_log2,
                baseline=0.0 if plot_log2 else mean_cnv,
                sex_estimate=sex_estimate,
                configured_genes=configured_genes,
                cutoff_override=cutoff_override,
                outliers_only=outliers_only,
                clinical_trial_genes=clinical_trial_genes,
            )
            chrom_name = _chromosome_display_name(contig)
            cnv_df = _chromosome_cnv_dataframe(positions_mb, plot_values)
            if plot_log2:
                cnv_df["state"] = [
                    _cnv_plot_point_state(value, contig, sex_estimate, cutoff_override)
                    for value in cnv_df["ploidy"]
                ]

            def _build_page(
                axis_span, note=None, full_range=False, label_full_range=True
            ):
                """One page for this chromosome on the given axis window."""
                y_min, y_max = data_lo, data_hi
                if full_range:
                    # Every bin at its true value, however deep.
                    y_min, y_max = _full_range_y_limits(
                        plot_values, use_log=bool(plot_log2)
                    )
                elif axis_span:
                    # One window for every chromosome, so pages are comparable.
                    y_min, y_max = cnv_chromosome_axis_window(
                        use_log=bool(plot_log2), span_log2=float(axis_span)
                    )
                elif significant_panel_points:
                    y_min, y_max = _expand_ylim_for_coverage_points(
                        y_min, y_max, significant_panel_points
                    )

                status_text = note or chromosome_status.get(
                    contig, "No significant CNV change"
                )
                has_status = bool(
                    status_text and status_text != "No significant CNV change"
                )
                fig = plt.figure(figsize=(total_width, total_height))
                ax = fig.add_subplot(111)

                margins = _chromosome_figure_margins(
                    has_status=has_status,
                    compact=compact_layout,
                )
                fig.subplots_adjust(**margins)
                fig.suptitle(
                    f"Chromosome {chrom_name}"
                    + (" — full range" if full_range and label_full_range else ""),
                    fontsize=CNV_FONT["title"],
                    color=CNV_TEXT["primary"],
                    y=0.99 if has_status else 0.97,
                    fontproperties=_CNV_FONT_BOLD,
                )
                if has_status:
                    wrapped_status = _wrap_chromosome_status_text(status_text)
                    fig.text(
                        0.5,
                        0.88 if compact_layout else 0.90,
                        wrapped_status,
                        ha="center",
                        va="top",
                        fontsize=CNV_FONT["subtitle"],
                        color=CNV_TEXT["muted"],
                        linespacing=1.2,
                        fontproperties=_CNV_FONT_REGULAR,
                    )

                # Data band before markers are clamped onto it.
                band_lo, band_hi = y_min, y_max
                if not axis_span:
                    y_min, y_max = _snap_cnv_ylim(y_min, y_max, use_log=plot_log2)

                _add_cnv_reference_guides(
                    ax,
                    x_max_mb,
                    use_log=plot_log2,
                    y_min=y_min,
                    y_max=y_max,
                    chromosome=contig,
                    sex_estimate=sex_estimate,
                    cutoff_override=cutoff_override,
                )
                centromere_bp = _centromere_boundaries().get(contig)
                if centromere_bp is not None:
                    _add_centromere_line(ax, [centromere_bp / 1_000_000.0])
                _scatter_cnv_chromosome_points(
                    ax, cnv_df, color_by_state=plot_log2, show_trend=show_trend,
                    contig=contig,
                )
                if axis_span:
                    offscale = _mark_offscale_bins(
                        ax, cnv_df["position_mb"], cnv_df["ploidy"], y_min, y_max
                    )
                    if offscale:
                        logger.debug(
                            "%s: %d bin(s) outside the fixed axis", contig, offscale
                        )
                _apply_cnv_chromosome_axes(
                    ax,
                    x_max_mb,
                    y_max,
                    y_min=y_min,
                    xlabel="Position (Mb)",
                    ylabel=ylabel,
                    show_xlabel=True,
                    use_log=plot_log2,
                )
                if significant_panel_points:
                    _add_chromosome_panel_coverage_overlay(
                        ax,
                        _clamp_coverage_points_to_band(
                            significant_panel_points, band_lo, band_hi
                        ),
                        x_max_mb,
                        y_min=y_min,
                        y_max=y_max,
                        band_lo=band_lo,
                        band_hi=band_hi,
                        label_size=gene_label_size,
                        rotated=gene_labels_rotated,
                        panel_width_pt=float(total_width) * 72.0,
                        panel_height_pt=float(total_height)
                        * max(margins["top"] - margins["bottom"], 0.05)
                        * 72.0,
                        reference_levels=_drawn_reference_levels(
                            use_log=plot_log2,
                            y_min=y_min,
                            y_max=y_max,
                            chromosome=contig,
                            sex_estimate=sex_estimate,
                            cutoff_override=cutoff_override,
                        ),
                    )
                    if figure_legend:
                        _add_clinical_trial_legend(
                            fig, significant_panel_points, corner="top"
                        )
                return fig

            if full_range_axis:
                # The report's own page, on the true data range and with no
                # "full range" suffix — there is no other page to distinguish
                # it from here.
                yield contig, _build_page(
                    None, full_range=True, label_full_range=False
                )
            else:
                yield contig, _build_page(fixed_axis_log2)

            # A companion page, fitted to the data, whenever the fixed window
            # cannot show everything — so a deep event is never reduced to an
            # edge marker with no magnitude.
            offscale_count = _count_offscale_bins(
                plot_values, fixed_axis_log2, use_log=plot_log2
            )
            # A panel gene pinned to the edge is just as much a reason to offer
            # the full-range view as an off-scale bin is.
            if fixed_axis_log2:
                fixed_lo, fixed_hi = cnv_chromosome_axis_window(
                    use_log=bool(plot_log2), span_log2=float(fixed_axis_log2)
                )
                offscale_count += sum(
                    1
                    for point in significant_panel_points
                    if np.isfinite(point.get("y_norm", np.nan))
                    and not (fixed_lo <= float(point["y_norm"]) <= fixed_hi)
                )
            if full_range_pages and offscale_count:
                span_label = _format_axis_span(fixed_axis_log2, use_log=plot_log2)
                yield contig, _build_page(
                    None,
                    note=(
                        f"Full range: axis fitted to the data so the "
                        f"{offscale_count} bin(s) beyond {span_label} are shown "
                        f"at their true value"
                    ),
                    full_range=True,
                )

    except Exception as e:
        logger.error(f"Error in iter_CNV_chromosome_figures: {str(e)}")
        plt.close("all")


def create_CNV_plot_per_chromosome(*args, **kwargs):
    """Per-chromosome CNV plots as JPEG buffers for the PDF report.

    Returns:
        List[Tuple[str, io.BytesIO]]: Chromosome names and their plot buffers.
    """
    plots: List[Tuple[str, io.BytesIO]] = []
    for contig, fig in iter_CNV_chromosome_figures(*args, **kwargs):
        try:
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="jpg",
                dpi=300,
                bbox_inches="tight",
                pad_inches=CNV_CHROMOSOME_PAD_INCHES,
            )
            plt.close(fig)
            buf.seek(0)

            buf_data = buf.getvalue()
            if not buf_data:
                logger.warning(f"Empty buffer for chromosome {contig} CNV plot")
                continue
            if len(buf_data) < 2 or buf_data[:2] != b"\xff\xd8":
                logger.warning(f"Invalid JPEG data for chromosome {contig} CNV plot")
                continue

            buf.seek(0)
            plots.append((contig, buf))
        except Exception as e:
            logger.error(f"Error creating CNV plot for chromosome {contig}: {str(e)}")
            plt.close("all")
            continue
    return plots


def _write_cnv_pdf(
    figures: Iterator[Any],
    metadata: Optional[Dict[str, str]] = None,
    header_text: Optional[str] = None,
) -> bytes:
    """Write matplotlib figures into one multi-page PDF and return its bytes.

    ``header_text`` is stamped at the top-left of the **first** page only, which
    is where the CNV load summary goes: it is a property of the sample, not of
    any one chromosome, so repeating it on every page would be noise.
    """
    buf = io.BytesIO()
    pages = 0
    with PdfPages(buf, metadata=metadata or {}) as pdf:
        for fig in figures:
            if fig is None:
                continue
            if header_text and pages == 0:
                fig.text(
                    0.005,
                    0.995,
                    header_text,
                    ha="left",
                    va="top",
                    fontsize=CNV_FONT["annotation"],
                    color=CNV_TEXT["primary"],
                    fontproperties=_CNV_FONT_REGULAR,
                )
            try:
                pdf.savefig(fig, bbox_inches="tight", pad_inches=CNV_CHROMOSOME_PAD_INCHES)
                pages += 1
            finally:
                plt.close(fig)
    if pages == 0:
        return b""
    return buf.getvalue()


def create_CNV_genome_pdf(*args, **kwargs) -> bytes:
    """Genome-wide CNV summary as a single-page vector PDF.

    Same figure as the report, written as vector art so it stays sharp when the
    reporting scientists zoom into a region.
    """
    title = kwargs.pop("pdf_title", "ROBIN genome-wide CNV")
    header_text = kwargs.pop("header_text", None)
    try:
        fig = build_CNV_genome_figure(*args, **kwargs)
        return _write_cnv_pdf(
            iter([fig]), metadata={"Title": title}, header_text=header_text
        )
    except Exception as e:
        logger.error(f"Error creating genome-wide CNV PDF: {e}")
        plt.close("all")
        return b""


def create_CNV_chromosome_pdf(*args, **kwargs) -> bytes:
    """All per-chromosome CNV plots as one multi-page vector PDF (one page each)."""
    title = kwargs.pop("pdf_title", "ROBIN per-chromosome CNV")
    header_text = kwargs.pop("header_text", None)
    try:
        figures = (
            fig for _contig, fig in iter_CNV_chromosome_figures(*args, **kwargs)
        )
        return _write_cnv_pdf(
            figures, metadata={"Title": title}, header_text=header_text
        )
    except Exception as e:
        logger.error(f"Error creating per-chromosome CNV PDF: {e}")
        plt.close("all")
        return b""


def classification_plot(df, title, threshold):
    """
    Creates a classification plot.

    Args:
        df (pd.DataFrame): DataFrame containing the classification data.
        title (str): Title of the plot.
        threshold (float): Threshold value for filtering classifications.

    Returns:
        io.BytesIO: Buffer containing the plot image.
    """
    set_modern_style()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Reshape the data to long format
    df_melted = df.melt(id_vars=["timestamp"], var_name="Condition", value_name="Value")
    meta_conditions = {
        "number_probes",
        "covered_cpgs",
        "temperature",
        "diagnostic",
        "probes",
    }
    df_melted = df_melted[
        ~df_melted["Condition"].astype(str).str.strip().str.lower().isin(meta_conditions)
    ]

    # Filter conditions that cross the threshold
    top_conditions = df_melted.groupby("Condition")["Value"].max().nlargest(10).index
    df_filtered = df_melted[df_melted["Condition"].isin(top_conditions)]

    conditions_above_threshold = df_filtered[df_filtered["Value"] > threshold][
        "Condition"
    ].unique()
    df_filtered = df_filtered[df_filtered["Condition"].isin(conditions_above_threshold)]

    # Create figure with adjusted size and margins
    fig = plt.figure(figsize=(10, 6))

    # Only create legend if we have data to plot
    if not df_filtered.empty:
        sns.lineplot(
            data=df_filtered, x="timestamp", y="Value", hue="Condition", palette="Set2"
        )

        # Move the legend below the plot with adjusted position
        plt.legend(
            title="Condition", bbox_to_anchor=(0.5, -0.3), loc="upper center", ncol=3
        )
    else:
        # If no data, create an empty plot
        plt.plot([])
        plt.text(
            0.5,
            0.5,
            "No classification data above threshold",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.title(f"{title} Classifications over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.xticks(rotation=45)

    # Format the x-axis with custom date format
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M"))

    # Adjust layout with explicit margins
    plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, top=0.9)

    # Save the plot as a JPG file with reduced DPI
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    return buf


def coverage_plot(df):
    """
    Creates a coverage plot.

    Args:
        df (pd.DataFrame): DataFrame containing coverage data.

    Returns:
        io.BytesIO: Buffer containing the plot image.
    """
    set_modern_style()

    # df = df[df["#rname"] != "chrM"].copy()
    df = df[
        df["#rname"].isin(["chr" + str(i) for i in range(0, 23)] + ["chrX", "chrY"])
    ].copy()

    # Sort chromosomes naturally
    df["#rname"] = pd.Categorical(
        df["#rname"], categories=natsort.natsorted(df["#rname"].unique()), ordered=True
    )
    df = df.sort_values("#rname")

    # Create figure with adjusted size
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)

    # Font size settings
    title_fontsize = 8
    label_fontsize = 8
    tick_fontsize = 6

    # Plot number of reads per chromosome
    ax0 = plt.subplot(gs[0])
    sns.barplot(
        x="#rname", y="numreads", data=df, ax=ax0, color=MODERN_COLORS["accent"]
    )
    ax0.set_title("Number of Reads per Chromosome", fontsize=title_fontsize)
    ax0.set_xlabel("", fontsize=label_fontsize)
    ax0.set_ylabel("Number of Reads", fontsize=label_fontsize)
    ax0.tick_params(axis="x", rotation=90, labelsize=tick_fontsize)
    ax0.tick_params(axis="y", labelsize=tick_fontsize)

    # Plot number of bases per chromosome
    ax1 = plt.subplot(gs[1])
    sns.barplot(
        x="#rname", y="covbases", data=df, ax=ax1, color=MODERN_COLORS["accent"]
    )
    ax1.set_title("Number of Bases per Chromosome", fontsize=title_fontsize)
    ax1.set_xlabel("", fontsize=label_fontsize)
    ax1.set_ylabel("Number of Bases", fontsize=label_fontsize)
    ax1.tick_params(axis="x", rotation=90, labelsize=tick_fontsize)
    ax1.tick_params(axis="y", labelsize=tick_fontsize)

    # Plot mean depth per chromosome
    ax2 = plt.subplot(gs[2])
    sns.barplot(
        x="#rname", y="meandepth", data=df, ax=ax2, color=MODERN_COLORS["accent"]
    )
    ax2.set_title("Mean Depth per Chromosome", fontsize=title_fontsize)
    ax2.set_xlabel("Chromosome", fontsize=label_fontsize)
    ax2.set_ylabel("Mean Depth", fontsize=label_fontsize)
    ax2.tick_params(axis="x", rotation=90, labelsize=tick_fontsize)
    ax2.tick_params(axis="y", labelsize=tick_fontsize)

    # Adjust layout with explicit margins
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=0.5)

    # Save the plot as a JPG file with reduced DPI
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    return buf


def plot_classification_timeline(
    df: pd.DataFrame, classification_level: str = "class", title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot classification changes over time.

    Args:
        df: DataFrame containing classification data
        classification_level: Level of classification to plot ('superfamily', 'family', 'class', 'subclass')
        title: Optional title for the plot

    Returns:
        Tuple of (figure, axes) objects
    """
    if df.empty:
        logger.warning("No data available for plotting")
        return None, None

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot classification scores
    ax.plot(
        df["timestamp"],
        df[f"{classification_level}_score"],
        marker="o",
        linestyle="-",
        label="Score",
    )

    # Add labels for each point
    for x, y, label in zip(
        df["timestamp"],
        df[f"{classification_level}_score"],
        df[f"{classification_level}_label"],
    ):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

    # Customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Classification Score")
    ax.set_title(title or f"{classification_level.title()} Classification Over Time")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_mgmt_timeline(
    df: pd.DataFrame, title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot MGMT methylation changes over time.

    Args:
        df: DataFrame containing MGMT data
        title: Optional title for the plot

    Returns:
        Tuple of (figure, axes) objects
    """
    if df.empty:
        logger.warning("No data available for plotting")
        return None, None

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot methylation percentage
    ax.plot(
        df["timestamp"],
        df["mgmt_methylation"],
        marker="o",
        linestyle="-",
        label="Methylation %",
    )

    # Add status labels
    for x, y, status in zip(df["timestamp"], df["mgmt_methylation"], df["mgmt_status"]):
        ax.annotate(status, (x, y), xytext=(5, 5), textcoords="offset points")

    # Customize plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Methylation Percentage")
    ax.set_title(title or "MGMT Methylation Over Time")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def save_plot(fig: plt.Figure, output_path: str, dpi: int = 300):
    """Save a plot to a file.

    Args:
        fig: Figure object to save
        output_path: Path to save the plot
        dpi: DPI for the output image
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save plot
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")


def plot_to_bytes(fig: plt.Figure, format: str = "png", dpi: int = 300) -> bytes:
    """Convert a plot to bytes.

    Args:
        fig: Figure object to convert
        format: Output format ('png', 'jpg', etc.)
        dpi: DPI for the output image

    Returns:
        Bytes containing the plot image
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Error converting plot to bytes: {str(e)}")
        return None
