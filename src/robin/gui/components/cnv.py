from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import asyncio
import json
import natsort
import numpy as np
from functools import lru_cache
import logging
import pickle
import time
import importlib.resources as importlib_resources
import pandas as pd

try:
    from nicegui import ui
except ImportError:  # pragma: no cover
    ui = None

from robin.gui.theme import (
    styled_table,
    register_theme_sync_callback,
    get_user_dark_mode,
    client_timer,
    stop_timer,
)
from robin.analysis.cnv_classification import (
    CNVEvent,
    detect_cnv_events,
    format_cnv_events_card_lines,
    format_cnv_events_section_summary,
)
from robin.analysis.cnv_analysis import (
    compute_cnv_log2_from_ploidy,
    downsample_cnv_for_plot,
    prepare_cnv_calling_track,
    resolve_cnv_calling_track,
    resolve_cnv_plot_bin_width,
)
from robin.analysis.cnv_regional import (
    SIGNIFICANT_CNV_STATES,
    analyze_cytoband_cnv,
    build_regional_cnv_events,
    format_regional_event_table_row,
    is_reportable_chromosome,
    load_panel_gene_bed,
    unmappable_bin_mask,
)
from robin.analysis.itd_work import load_gene_target_coverage
from robin.classification_config import get_cnv_thresholds
from robin.cnv_plot_style import (
    CNV_LOG2_MAX_AXIS_SPAN,
    CNV_LOG2_MIN_AXIS_SPAN,
    assign_label_lanes,
    clamp_band_for_markers,
    cnv_axis_tick_spec,
    cnv_chromosome_axis_window,
    cnv_reference_levels,
    cnv_segment_points,
    gene_crosses_cutoff,
    horizontal_label_proximity_frac,
    snap_axis_window_to_ticks,
)

# Same chromosome set as reporting (plotting.py): chr0–chr22, chrX, chrY only
CNV_PLOT_CONTIGS = frozenset(
    ["chr" + str(i) for i in range(0, 23)] + ["chrX", "chrY"]
)

_CNV_PLOT_BIN_KEY_DEFAULT = "Data default"
_CNV_PLOT_BIN_OPTIONS = {
    _CNV_PLOT_BIN_KEY_DEFAULT: "Data default",
    "500 kb": "500 kb",
    "1 Mb": "1 Mb",
    "2 Mb": "2 Mb",
    "5 Mb": "5 Mb",
    "10 Mb": "10 Mb",
}
_CNV_PLOT_BIN_KEY_TO_BP = {
    _CNV_PLOT_BIN_KEY_DEFAULT: None,
    "500 kb": 500_000,
    "1 Mb": 1_000_000,
    "2 Mb": 2_000_000,
    "5 Mb": 5_000_000,
    "10 Mb": 10_000_000,
}
_CNV_PLOT_BIN_KEYS_ORDERED = list(_CNV_PLOT_BIN_KEY_TO_BP.keys())

# ECharts JS: y values to at most 1 decimal place, trailing ".0" dropped, so
# axis ticks and zoom-slider handles never show full float precision.
_CNV_Y_VALUE_FORMATTER_JS = (
    "(value) => { const n = Number(value); "
    "return Number.isFinite(n) ? String(Number(n.toFixed(1))) : value; }"
)
_CONFIGURED_GENES_SERIES_NAME = "configured_genes_highlight"
_CONFIGURED_GENES_LABELS_SERIES_NAME = "configured_genes_labels"
_CNV_TREND_SERIES_PREFIX = "cnv_trend"
_CNV_REFERENCE_SERIES_NAME = "cnv_reference_lines"
_CNV_CENTROMERE_SERIES_NAME = "cnv_centromere_lines"
_CNV_OVERLAY_SERIES_NAMES = frozenset(
    {
        "centromeres_highlight",
        "cytobands_highlight",
        _CONFIGURED_GENES_SERIES_NAME,
        _CONFIGURED_GENES_LABELS_SERIES_NAME,
        _CNV_REFERENCE_SERIES_NAME,
        _CNV_CENTROMERE_SERIES_NAME,
    }
)

# Y-axis window bounds. The log2 window is symmetric about zero so a gain and the
# equivalent loss sit the same distance from the baseline (WGM/conumee convention).
_CNV_LOG_Y_MIN_SPAN = CNV_LOG2_MIN_AXIS_SPAN
_CNV_LOG_Y_MAX_SPAN = CNV_LOG2_MAX_AXIS_SPAN
_CNV_LINEAR_Y_MAX_FACTOR = 4.0
_CNV_DIFF_Y_MIN_SPAN = 1.0
_CNV_DIFF_Y_MAX_SPAN = 6.0
_CNV_LARGE_SCATTER_THRESHOLD = 2000
_CNV_PROGRESSIVE_CHUNK = 4000
# ECharts draws this many minor divisions between labelled ticks.
_CNV_AXIS_MINOR_SPLIT = 2
# Room for the mirrored right-hand tick labels.
_CNV_GRID_RIGHT_WITH_MIRROR = "6%"

# Gene names are drawn rotated inside the panel (methylation-array convention),
# so nothing is reserved for them and the axis stays driven by the data.
_CNV_GENE_MARKER_SIZE = 9
# Rough width of a bold character as a fraction of font size, used to offset a
# rotated label clear of its own marker.
_CNV_LABEL_CHAR_WIDTH_RATIO = 0.62
# Fallback ECharts label size when no preference is set.
_CNV_LABEL_DEFAULT_PX = 11.0
# Nominal plotted width in px, used to judge how wide a horizontal label is
# relative to the panel when deciding which names need stacking.
_CNV_GUI_PANEL_WIDTH_PX = 1400.0
# Lane step for stacked horizontal labels, as a multiple of the font size.
_CNV_LABEL_LANE_SPACING = 1.35
# A rotated name runs along the y-axis, so its extent scales with its length.
# Used only to decide which side of the marker has room for it.
_CNV_LABEL_EXTENT_PER_CHAR = 0.03
_CNV_LABEL_EXTENT_MAX = 0.4

# Segment line over the bin cloud, matching the report figures.
_CNV_TREND_COLOR_DARK = "#f8fafc"
_CNV_TREND_COLOR_LIGHT = "#1E3A5F"
# Gain/loss calling cut-offs. Deliberately high-contrast: these are the lines the
# reporting scientists read gains and losses against.
_CNV_CUTOFF_COLOR_LIGHT = "#B45309"
_CNV_CUTOFF_COLOR_DARK = "#fbbf24"
# p/q arm divider: faint enough to sit behind the data.
_CNV_CENTROMERE_COLOR_LIGHT = "rgba(100, 116, 139, 0.55)"
_CNV_CENTROMERE_COLOR_DARK = "rgba(148, 163, 184, 0.5)"
_CNV_GENE_COVERAGE_FILTER_ALL = "all"
_CNV_GENE_COVERAGE_FILTER_OUTLIERS = "outliers"
_CNV_GENE_COVERAGE_FILTERS = (
    _CNV_GENE_COVERAGE_FILTER_ALL,
    _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
)
# Gains orange, losses green - the same convention the PDF reports use, so a
# gene reads the same way on screen and on paper. These previously ran the other
# way round from the report (gain red, loss blue).
_CNV_GENE_GAIN_COLOR = "#EA580C"
_CNV_GENE_LOSS_COLOR = "#15803D"
# Soft safety only — axis auto-scales to highlighted genes within this envelope.
_CNV_LOLLIPOP_LOG_Y_SOFT_CAP = 20.0
_CNV_LOLLIPOP_LINEAR_Y_SOFT_CAP_FACTOR = 20.0


@lru_cache(maxsize=16)
def _load_cnv_gene_locations(gene_names: tuple[str, ...]) -> tuple[Dict[str, Any], ...]:
    """Resolve configured gene symbols to GRCh38 intervals in the packaged gene BED."""
    requested = {name.casefold(): name for name in gene_names if name.strip()}
    if not requested:
        return ()

    intervals: Dict[tuple[str, str], Dict[str, Any]] = {}
    try:
        resources = importlib_resources.files("robin.resources")
        unresolved = set(requested)
        # all_genes3 supplies gene-body coordinates. unique_genes includes a few
        # panel aliases/non-coding genes absent from that reference.
        for resource_name in ("all_genes3.bed", "unique_genes.bed"):
            if not unresolved:
                break
            resource = resources / resource_name
            with resource.open("r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 4:
                        continue
                    chrom = fields[0].strip()
                    try:
                        start_pos, end_pos = int(fields[1]), int(fields[2])
                    except ValueError:
                        continue
                    for raw_symbol in fields[3].split(","):
                        symbol_key = raw_symbol.strip().casefold()
                        if symbol_key not in unresolved:
                            continue
                        requested_name = requested[symbol_key]
                        key = (symbol_key, chrom)
                        existing = intervals.get(key)
                        if existing is None:
                            intervals[key] = {
                                "gene": requested_name,
                                "chrom": chrom,
                                "start_pos": start_pos,
                                "end_pos": end_pos,
                            }
                        else:
                            existing["start_pos"] = min(
                                existing["start_pos"], start_pos
                            )
                            existing["end_pos"] = max(existing["end_pos"], end_pos)
            unresolved -= {key[0] for key in intervals}
    except Exception:
        logging.warning("Could not load CNV gene locations", exc_info=True)
        return ()

    order = {name.casefold(): index for index, name in enumerate(gene_names)}
    return tuple(
        sorted(
            intervals.values(),
            key=lambda row: (order.get(str(row["gene"]).casefold(), 10**9), row["chrom"]),
        )
    )


def _configured_genes_on_chrom(
    gene_locations: Sequence[Dict[str, Any]],
    selected: str,
) -> List[Dict[str, Any]]:
    """Return configured gene intervals for one chromosome (or all when selected is All)."""
    if selected == "All":
        return list(gene_locations)
    return [row for row in gene_locations if str(row["chrom"]) == selected]


def _find_configured_gene_interval(
    gene_locations: Sequence[Dict[str, Any]],
    *,
    selected: str,
    gene_name: str,
) -> Optional[Dict[str, Any]]:
    """Look up a configured gene interval on the current chromosome."""
    want = str(gene_name).casefold()
    for row in _configured_genes_on_chrom(gene_locations, selected):
        if str(row["gene"]).casefold() == want:
            return row
    return None


def _configured_gene_mark_series(
    gene_locations: Sequence[Dict[str, Any]],
    *,
    selected: str,
    chrom_offsets: Dict[str, float],
    dark: bool,
) -> Dict[str, Any]:
    """Build a dedicated ECharts overlay for configured CNV gene locations."""
    lines: List[Dict[str, Any]] = []
    areas: List[List[Dict[str, Any]]] = []
    # Invisible anchors so ECharts keeps the series (empty scatter markLines can vanish).
    anchors: List[List[float]] = []
    label_color = "#d8b4fe" if dark else "#6b21a8"
    line_color = "#a855f7" if dark else "#7e22ce"
    for row in gene_locations:
        chrom = str(row["chrom"])
        if selected != "All" and chrom != selected:
            continue
        if selected == "All" and chrom not in chrom_offsets:
            continue
        start_pos = float(row["start_pos"])
        end_pos = float(row["end_pos"])
        midpoint = (start_pos + end_pos) / 2.0
        offset = chrom_offsets.get(chrom, 0.0) if selected == "All" else 0.0
        x_pos = midpoint + offset
        gene = str(row["gene"])
        anchors.append([x_pos, 0.0])
        lines.append(
            {
                "name": gene,
                "xAxis": x_pos,
                "lineStyle": {
                    "color": line_color,
                    "width": 1 if selected == "All" else 1.5,
                    "opacity": 0.75,
                },
                "label": {
                    "show": True,
                    "formatter": gene,
                    "position": "insideEndTop",
                    "rotate": 90,
                    "fontSize": 9 if selected == "All" else 11,
                    "color": label_color,
                },
            }
        )
        # Chromosome view also gets a labelled span so names remain readable.
        if selected != "All":
            areas.append(
                [
                    {
                        "name": gene,
                        "xAxis": start_pos,
                        "itemStyle": {
                            "color": (
                                "rgba(168, 85, 247, 0.12)"
                                if dark
                                else "rgba(126, 34, 206, 0.10)"
                            )
                        },
                        "label": {
                            "show": True,
                            "position": "insideTop",
                            "formatter": gene,
                            "color": label_color,
                            "fontSize": 11,
                        },
                    },
                    {"xAxis": end_pos},
                ]
            )
    return {
        "type": "scatter",
        "name": _CONFIGURED_GENES_SERIES_NAME,
        "data": anchors,
        "symbolSize": 0,
        "silent": True,
        "zlevel": 2,
        "markLine": {
            "symbol": "none",
            "animation": False,
            "data": lines,
        },
        "markArea": {
            "silent": True,
            "data": areas,
        },
    }


def _upsert_configured_gene_series(
    chart: Any,
    gene_locations: Sequence[Dict[str, Any]],
    *,
    selected: str,
    chrom_offsets: Dict[str, float],
    dark: bool,
) -> None:
    """Ensure the configured-gene overlay is present and last among chart series."""
    series = chart.options.get("series")
    if not isinstance(series, list):
        return
    chart.options["series"] = [
        s
        for s in series
        if s.get("name")
        not in (
            _CONFIGURED_GENES_SERIES_NAME,
            _CONFIGURED_GENES_LABELS_SERIES_NAME,
        )
    ]
    if not gene_locations:
        return
    chart.options["series"].append(
        _configured_gene_mark_series(
            gene_locations,
            selected=selected,
            chrom_offsets=chrom_offsets,
            dark=dark,
        )
    )


def _resolve_cnv_labels_rotated(value: Any) -> bool:
    """True when gene names should be drawn rotated rather than horizontally."""
    try:
        from robin.gui.plotting_preferences import cnv_label_is_rotated

        return cnv_label_is_rotated(value)
    except Exception:
        logging.debug("Could not resolve label orientation %r", value, exc_info=True)
        return True


def _resolve_cnv_gene_label_points(value: Any) -> Optional[float]:
    """Resolve the stored gene label size to matplotlib points, for exports."""
    try:
        from robin.gui.plotting_preferences import resolve_cnv_gene_label_size

        return resolve_cnv_gene_label_size(value)
    except Exception:
        logging.debug("Could not resolve gene label points %r", value, exc_info=True)
        return None


def _resolve_cnv_gene_label_px(value: Any) -> Optional[float]:
    """Resolve the stored gene label size to an ECharts px size."""
    try:
        from robin.gui.plotting_preferences import cnv_gene_label_size_px

        return cnv_gene_label_size_px(value)
    except Exception:
        logging.debug("Could not resolve gene label size %r", value, exc_info=True)
        return None


def _resolve_cnv_chrom_axis(value: Any) -> Optional[float]:
    """Resolve the stored per-chromosome Y-range to a log2 half-span (None = auto)."""
    try:
        from robin.gui.plotting_preferences import resolve_cnv_chrom_axis

        return resolve_cnv_chrom_axis(value)
    except Exception:
        logging.debug("Could not resolve CNV chromosome axis %r", value, exc_info=True)
        return None


def _cutoff_suffix_text(cutoff_override: Optional[float] = None) -> str:
    """Cut-off note for a GUI heading whose rows move with the cut-off."""
    try:
        from robin.gui.plotting_preferences import cnv_cutoff_heading_suffix

        return cnv_cutoff_heading_suffix(cutoff_override)
    except Exception:
        return ""


def _set_cutoff_heading(label, base: str, cutoff_override: Optional[float]) -> None:
    """Rewrite a heading so it always names the cut-off its rows were built at."""
    try:
        label.set_text(base + _cutoff_suffix_text(cutoff_override))
    except Exception:
        logging.debug("Could not update the cut-off heading", exc_info=True)


def _resolve_cnv_cutoff(value: Any) -> Optional[float]:
    """Resolve the stored cut-off setting to a log2 magnitude (None = calling)."""
    try:
        from robin.gui.plotting_preferences import resolve_cnv_cutoff

        return resolve_cnv_cutoff(value)
    except Exception:
        logging.debug("Could not resolve CNV cut-off %r", value, exc_info=True)
        return None


def _cnv_cutoff_thresholds(
    chromosome: str,
    sex_estimate: str,
    cutoff_override: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Gain/loss cut-off in force, honouring the user's cut-off selection.

    An override moves everything the cut-off drives together: the drawn lines,
    the point colouring, the Outliers filter, the called regions, the events and
    regional tables, the gene states and the CNV load. One control, one number,
    every output.
    """
    if cutoff_override is not None:
        magnitude = abs(float(cutoff_override))
        return magnitude, -magnitude
    try:
        return get_cnv_thresholds(
            chromosome or "chr1", sex_estimate or "Unknown"
        )
    except Exception:
        logging.debug("Could not resolve CNV thresholds for %s", chromosome, exc_info=True)
        return None, None


def _is_configured_gene_cnv_outlier(
    cnv_val: float,
    *,
    chromosome: str,
    sex_estimate: str,
    use_log: bool,
    baseline: float,
    cutoff_override: Optional[float] = None,
) -> bool:
    """True when a gene crosses the gain/loss cut-off currently in force.

    Same test as the cut-off lines drawn on the plot, taken against the
    genome-wide baseline — so a gene on a gained chromosome is reported as gained.
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
        use_log=use_log,
        baseline=baseline,
    )


def _configured_gene_region_cnv(
    values: np.ndarray,
    *,
    start_pos: float,
    end_pos: float,
    bin_width: int,
    use_max_abs: bool,
) -> Optional[float]:
    """Peak (or max-abs) CNV across bins overlapping a gene interval."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or bin_width <= 0:
        return None
    start_bin = max(0, int(start_pos // bin_width))
    end_bin = min(arr.size - 1, int(end_pos // bin_width))
    if end_bin < start_bin:
        return None
    region = arr[start_bin : end_bin + 1]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return None
    if use_max_abs:
        return float(finite[np.nanargmax(np.abs(finite))])
    return float(np.nanmax(finite))


def _configured_gene_cnv_direction(
    cnv_val: float,
    *,
    use_log: bool,
    baseline: float,
) -> str:
    """Classify gene CNV as gain or loss against the genome-wide baseline."""
    if use_log:
        return "gain" if cnv_val >= 0.0 else "loss"
    return "gain" if cnv_val >= baseline else "loss"


def _build_configured_gene_coverage_points(
    gene_locations: Sequence[Dict[str, Any]],
    *,
    selected: str,
    chrom_offsets: Dict[str, float],
    abs_plot_map: Dict[str, np.ndarray],
    bin_width: int,
    coverage_by_gene: Dict[str, float],
    filter_mode: str,
    use_log: bool,
    scale_mean_cnv: float,
    sex_estimate: str = "Unknown",
    cutoff_override: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Mark configured genes at their own copy-number value on the CNV track.

    The marker sits on the profile, as in the methylation-array CNV plots the team
    reads alongside ROBIN. Target sequencing depth is carried in the tooltip rather
    than as the marker height: mapping depth onto the copy-number axis put markers
    several log2 units off the profile and forced the axis open to reach them.
    """
    if not gene_locations:
        return [], None

    cov_lookup = {str(k).casefold(): float(v) for k, v in (coverage_by_gene or {}).items()}
    cov_vals = [v for v in cov_lookup.values() if np.isfinite(v) and v > 0]
    mean_cov = float(np.mean(cov_vals)) if cov_vals else None

    baseline_y = 0.0 if use_log else float(scale_mean_cnv)
    points: List[Dict[str, Any]] = []
    for row in _configured_genes_on_chrom(gene_locations, selected):
        gene = str(row["gene"])
        chrom = str(row["chrom"])
        if selected == "All" and chrom not in chrom_offsets and chrom not in abs_plot_map:
            continue
        coverage = cov_lookup.get(gene.casefold())
        track = abs_plot_map.get(chrom)
        if track is None:
            continue
        cnv_val = _configured_gene_region_cnv(
            track,
            start_pos=float(row["start_pos"]),
            end_pos=float(row["end_pos"]),
            bin_width=int(bin_width),
            use_max_abs=use_log,
        )
        if cnv_val is None:
            continue
        if filter_mode == _CNV_GENE_COVERAGE_FILTER_OUTLIERS and not (
            _is_configured_gene_cnv_outlier(
                cnv_val,
                chromosome=chrom,
                sex_estimate=sex_estimate,
                use_log=use_log,
                baseline=float(scale_mean_cnv),
                cutoff_override=cutoff_override,
            )
        ):
            continue
        # Depth relative to the panel mean, for the tooltip only.
        coverage_ratio = None
        if (
            coverage is not None
            and np.isfinite(coverage)
            and mean_cov
            and np.isfinite(mean_cov)
            and mean_cov > 0
        ):
            coverage_ratio = float(coverage) / float(mean_cov)
        start_pos = float(row["start_pos"])
        end_pos = float(row["end_pos"])
        midpoint = (start_pos + end_pos) / 2.0
        offset = chrom_offsets.get(chrom, 0.0) if selected == "All" else 0.0
        points.append(
            {
                "gene": gene,
                "chrom": chrom,
                "x": midpoint + offset,
                "coverage": (
                    float(coverage)
                    if coverage is not None and np.isfinite(coverage)
                    else None
                ),
                "coverage_ratio": coverage_ratio,
                # The marker sits on the profile at the gene's own CNV value.
                "y": float(cnv_val),
                "baseline_y": float(baseline_y),
                "cnv_val": float(cnv_val),
                "direction": _configured_gene_cnv_direction(
                    float(cnv_val),
                    use_log=use_log,
                    baseline=float(scale_mean_cnv),
                ),
            }
        )
    return points, mean_cov


def _cnv_scatter_value_range(
    series_list: Sequence[Dict[str, Any]],
    *,
    percentile: float = 99.0,
) -> Optional[Tuple[float, float]]:
    """Robust value envelope of the plotted CNV bins.

    Uses percentiles rather than min/max so a single homozygous deletion cannot
    collapse the rest of the profile into a few pixels.
    """
    values: List[float] = []
    for series in series_list:
        if not isinstance(series, dict) or series.get("type") != "scatter":
            continue
        if series.get("name") in _CNV_OVERLAY_SERIES_NAMES:
            continue
        for point in series.get("data") or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            val = point[1]
            if val is None:
                continue
            fval = float(val)
            if np.isfinite(fval):
                values.append(fval)
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    lo = float(np.percentile(arr, 100.0 - float(percentile)))
    hi = float(np.percentile(arr, float(percentile)))
    if hi <= lo:
        pad = max(abs(hi) * 0.1, 0.1)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def _lollipop_baseline_y_range(
    *,
    use_log: bool,
    scale_mean_cnv: float,
    data_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """CNV data band before gene-label gutters are reserved.

    Log2 mode keeps the window symmetric about zero (conumee/WGM convention) so a
    gain and the matching loss are the same distance from the baseline.
    """
    if use_log:
        span = _CNV_LOG_Y_MIN_SPAN
        if data_range is not None:
            span = max(span, abs(float(data_range[0])), abs(float(data_range[1])))
        span = min(span, _CNV_LOG_Y_MAX_SPAN)
        return (-span, span)
    lo = 0.0
    hi = max(4.0, float(scale_mean_cnv) * 1.75)
    if data_range is not None:
        hi = max(hi, float(data_range[1]) + 0.25)
        hi = min(hi, max(4.0, float(scale_mean_cnv) * _CNV_LINEAR_Y_MAX_FACTOR))
    return (lo, hi)


def _lollipop_label_side(
    point: Dict[str, Any],
    *,
    y_lo: Optional[float] = None,
    y_hi: Optional[float] = None,
) -> str:
    """Which way a rotated gene name reads: gains up, losses down.

    Flipped when the preferred side has no room, so a name near the top or
    bottom of the panel is not left running off it.
    """
    baseline = float(point.get("baseline_y", 0.0))
    y_head = float(point.get("y_disp", point.get("y", 0.0)))
    if str(point.get("direction") or "") == "loss" or y_head < baseline:
        side = "below"
    else:
        side = "above"
    if y_lo is None or y_hi is None:
        return side
    span = max(float(y_hi) - float(y_lo), 1e-6)
    extent = span * min(
        len(str(point.get("gene") or "")) * _CNV_LABEL_EXTENT_PER_CHAR,
        _CNV_LABEL_EXTENT_MAX,
    )
    if side == "above" and y_head + extent > float(y_hi):
        return "below"
    if side == "below" and y_head - extent < float(y_lo):
        return "above"
    return side


def _soft_cap_lollipop_display_y(
    y_norm: float,
    *,
    use_log: bool,
    scale_mean_cnv: float,
) -> Tuple[float, bool]:
    """Only clip pathological extremes; normal gene values stay unscaled."""
    if use_log:
        lo, hi = -_CNV_LOLLIPOP_LOG_Y_SOFT_CAP, _CNV_LOLLIPOP_LOG_Y_SOFT_CAP
    else:
        lo = 0.0
        hi = max(12.0, float(scale_mean_cnv) * _CNV_LOLLIPOP_LINEAR_Y_SOFT_CAP_FACTOR)
    capped = bool(y_norm < lo or y_norm > hi)
    return float(np.clip(y_norm, lo, hi)), capped


def _plan_lollipop_layout(
    points: Sequence[Dict[str, Any]],
    *,
    use_log: bool,
    scale_mean_cnv: float,
    data_range: Optional[Tuple[float, float]] = None,
    fixed_axis_log2: Optional[float] = None,
) -> Dict[str, Any]:
    """Plan the CNV y-window and gene-label placement.

    Gene names are drawn rotated inside the panel, anchored to their marker, the
    way the methylation-array CNV plots do it. Nothing is reserved for them, so
    the axis is set by the copy-number data alone and keeps its fine tick
    intervals no matter how many genes are configured.
    """
    band_lo, band_hi = _lollipop_baseline_y_range(
        use_log=use_log,
        scale_mean_cnv=scale_mean_cnv,
        data_range=data_range,
    )

    marker_ys = [
        float(p["y_disp"])
        for p in points
        if p.get("y_disp") is not None and np.isfinite(float(p["y_disp"]))
    ]
    if marker_ys:
        band_lo, band_hi = clamp_band_for_markers(band_lo, band_hi, marker_ys)
    if not use_log:
        band_lo = max(0.0, band_lo)

    if fixed_axis_log2:
        # One window for every chromosome, so they can be compared directly.
        y_lo, y_hi = cnv_chromosome_axis_window(
            use_log=use_log,
            span_log2=float(fixed_axis_log2),
            baseline=float(scale_mean_cnv) if not use_log else 2.0,
        )
        ticks = cnv_axis_tick_spec(y_lo, y_hi, use_log=use_log)
    else:
        y_lo, y_hi = band_lo, band_hi
        ticks = cnv_axis_tick_spec(y_lo, y_hi, use_log=use_log)
        y_lo, y_hi = snap_axis_window_to_ticks(
            y_lo,
            y_hi,
            ticks.major,
            clamp_min=0.0 if not use_log else None,
        )
        ticks = cnv_axis_tick_spec(y_lo, y_hi, use_log=use_log)

    # A gene outside the window is pinned to the edge so it stays visible; the
    # tooltip flags it as capped.
    clamped: List[Dict[str, Any]] = []
    for point in points:
        y_disp = float(point["y_disp"])
        y_clamped = float(np.clip(y_disp, y_lo, y_hi))
        clamped.append(
            {
                **point,
                "y_disp": y_clamped,
                "capped": bool(point.get("capped")) or y_clamped != y_disp,
            }
        )
    points = clamped

    placements = [
        {
            "side": _lollipop_label_side(p, y_lo=y_lo, y_hi=y_hi),
            "y": float(p["y_disp"]),
        }
        for p in points
    ]

    return {
        "y_lo": float(y_lo),
        "y_hi": float(y_hi),
        "band_lo": float(band_lo),
        "band_hi": float(band_hi),
        "ticks": ticks,
        "placements": placements,
        "points": points,
    }


def _prepare_lollipop_points(
    points: Sequence[Dict[str, Any]],
    *,
    use_log: bool,
    scale_mean_cnv: float,
) -> List[Dict[str, Any]]:
    """Attach the soft-capped display y used for marker heads and layout."""
    prepared: List[Dict[str, Any]] = []
    for point in points:
        y_disp, capped = _soft_cap_lollipop_display_y(
            float(point["y"]),
            use_log=use_log,
            scale_mean_cnv=scale_mean_cnv,
        )
        prepared.append({**point, "y_disp": y_disp, "capped": capped})
    return prepared


def _rotated_label_distance(gene: str, font_size: float) -> float:
    """Pixel offset that clears a 90°-rotated gene name from its own marker.

    ECharts rotates a label about its own centre, so a label anchored at the
    marker would sit half on top of it. Offsetting by half the rendered text
    length plus the marker radius puts the text alongside instead.
    """
    text_px = max(len(str(gene)), 1) * float(font_size) * _CNV_LABEL_CHAR_WIDTH_RATIO
    return _CNV_GENE_MARKER_SIZE / 2.0 + 3.0 + text_px / 2.0


def _horizontal_label_distance(font_size: float, lane: int) -> float:
    """Pixel offset for a horizontal gene name, stepped by collision lane."""
    return (
        _CNV_GENE_MARKER_SIZE / 2.0
        + 3.0
        + int(lane) * float(font_size) * _CNV_LABEL_LANE_SPACING
    )


def _configured_gene_coverage_lollipop_series(
    prepared: Sequence[Dict[str, Any]],
    placements: Sequence[Dict[str, Any]],
    *,
    dark: bool,
    use_log: bool,
    label_size_px: Optional[float] = None,
    rotated: bool = True,
    x_span: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Gene markers on the profile, with rotated or horizontal names."""
    if not prepared:
        return []

    halo = "#0f172a" if dark else "#ffffff"
    base_font = float(label_size_px) if label_size_px else _CNV_LABEL_DEFAULT_PX
    # Crowded panels shed a little size, as before.
    label_font = base_font if len(prepared) <= 25 else base_font * 0.92

    # Horizontal names are as wide as the gene symbol, so overlapping ones step
    # into lanes; rotated names are narrow and sit on a single lane each.
    if rotated:
        lanes = [0] * len(prepared)
    else:
        lanes = assign_label_lanes(
            [float(p["x"]) for p in prepared],
            [str(pl.get("side") or "above") for pl in placements],
            x_span=x_span,
            x_proximity_frac=horizontal_label_proximity_frac(
                [str(p["gene"]) for p in prepared],
                font_size=label_font,
                panel_width_pt=_CNV_GUI_PANEL_WIDTH_PX,
            ),
        )

    marks: List[Dict[str, Any]] = []
    stems: List[Any] = []

    for point, place, lane in zip(prepared, placements, lanes):
        x_pos = float(point["x"])
        y_disp = float(point["y_disp"])
        baseline = float(point.get("baseline_y", 0.0 if use_log else 2.0))
        gene = str(point["gene"])
        capped = bool(point.get("capped"))
        color = (
            _CNV_GENE_GAIN_COLOR
            if point.get("direction") == "gain"
            else _CNV_GENE_LOSS_COLOR
        )
        # Read the name away from the baseline, so gains label upwards.
        above = place.get("side") != "below"
        marks.append(
            {
                "name": gene,
                "value": [x_pos, y_disp],
                "coverage": point.get("coverage"),
                "coverageRatio": point.get("coverage_ratio"),
                "capped": capped,
                "itemStyle": {
                    "color": color,
                    "borderColor": halo,
                    "borderWidth": 1,
                },
                "label": {
                    "show": True,
                    "formatter": gene,
                    "position": "top" if above else "bottom",
                    "distance": (
                        _rotated_label_distance(gene, label_font)
                        if rotated
                        else _horizontal_label_distance(label_font, lane)
                    ),
                    "rotate": 90 if rotated else 0,
                    "fontSize": label_font,
                    "fontWeight": "bold",
                    "color": color,
                    # A halo keeps the name legible over the bin cloud without a box.
                    "textBorderColor": halo,
                    "textBorderWidth": 2.5,
                    "align": "center",
                    "verticalAlign": "middle",
                },
            }
        )
        # Thin tie back to the baseline so the name traces to a genomic position.
        stems.append(
            [
                {
                    "coord": [x_pos, baseline],
                    "lineStyle": {
                        "color": color,
                        "type": "solid",
                        "width": 1,
                        "opacity": 0.45,
                    },
                },
                {"coord": [x_pos, y_disp]},
            ]
        )

    tooltip = {
        "trigger": "item",
        ":formatter": (
            "(params) => { const d = params.data || {}; "
            "const v = params.value; "
            "const y = Array.isArray(v) ? Number(v[1]) : Number(v); "
            "const cov = Number(d.coverage); "
            "const ratio = Number(d.coverageRatio); "
            "const parts = []; "
            "if (Number.isFinite(y)) parts.push(y.toFixed(2)); "
            "if (Number.isFinite(cov)) { "
            "  let c = cov.toFixed(1) + 'x'; "
            "  if (Number.isFinite(ratio)) c += ' (' + ratio.toFixed(2) + "
            "    '\\u00d7 panel mean)'; "
            "  parts.push(c); } "
            "if (d.capped) parts.push('off scale'); "
            "return params.name + ': ' + parts.join(' \\u00b7 '); }"
        ),
    }

    return [
        {
            "type": "scatter",
            "name": _CONFIGURED_GENES_SERIES_NAME,
            "yAxisIndex": 0,
            "symbolSize": _CNV_GENE_MARKER_SIZE,
            "zlevel": 4,
            "z": 12,
            "clip": False,
            "animation": False,
            "animationDuration": 0,
            "progressive": 0,
            "data": marks,
            "markLine": {
                "symbol": "none",
                "animation": False,
                "animationDuration": 0,
                "silent": True,
                "z": 11,
                "data": stems,
            },
            "tooltip": tooltip,
        }
    ]


def _apply_cnv_abs_y_window(
    chart: Any,
    y_lo: float,
    y_hi: float,
    *,
    ticks: Optional[Any] = None,
    use_log: bool = False,
) -> None:
    """Pin the CNV Y axis, its tick ladder, the mirrored axis and the slider."""
    if ticks is None:
        ticks = cnv_axis_tick_spec(y_lo, y_hi, use_log=use_log)
    try:
        dz_list = chart.options.get("dataZoom")
        if isinstance(dz_list, list) and len(dz_list) > 1 and isinstance(dz_list[1], dict):
            dz_list[1]["startValue"] = float(y_lo)
            dz_list[1]["endValue"] = float(y_hi)
            dz_list[1]["filterMode"] = "none"
            dz_list[1].pop("start", None)
            dz_list[1].pop("end", None)
    except Exception:
        pass
    try:
        y_axes = chart.options.get("yAxis")
        if isinstance(y_axes, list) and y_axes and isinstance(y_axes[0], dict):
            axis = y_axes[0]
            axis["min"] = float(y_lo)
            axis["max"] = float(y_hi)
            axis["scale"] = False
            axis["interval"] = float(ticks.major)
            axis["minorTick"] = {
                "show": True,
                "splitNumber": _CNV_AXIS_MINOR_SPLIT,
            }
            axis["minorSplitLine"] = {"show": True}
    except Exception:
        pass
    _apply_cnv_mirror_y_axis(chart, y_lo, y_hi, ticks=ticks)


def _apply_cnv_mirror_y_axis(
    chart: Any,
    y_lo: float,
    y_hi: float,
    *,
    ticks: Any,
) -> None:
    """Mirror the copy-number scale on the right-hand edge (WGM/conumee style).

    Reading a gain or loss off a genome-wide panel means tracking a point back to
    a tick that can be a metre of screen away; duplicating the labels on the right
    halves that distance.
    """
    try:
        y_axes = chart.options.get("yAxis")
        if not isinstance(y_axes, list) or len(y_axes) < 2:
            return
        mirror = y_axes[1]
        if not isinstance(mirror, dict):
            return
        mirror.update(
            {
                "type": "value",
                "name": "",
                "position": "right",
                "show": True,
                "min": float(y_lo),
                "max": float(y_hi),
                "scale": False,
                "interval": float(ticks.major),
                "axisTick": {"show": True},
                "minorTick": {"show": True, "splitNumber": _CNV_AXIS_MINOR_SPLIT},
                # Gridlines are drawn once, by the primary axis.
                "splitLine": {"show": False},
                "minorSplitLine": {"show": False},
                "axisLabel": {**(mirror.get("axisLabel") or {}), "show": True},
            }
        )
        grid = chart.options.get("grid")
        if isinstance(grid, dict):
            grid["right"] = _CNV_GRID_RIGHT_WITH_MIRROR
    except Exception:
        pass


def _upsert_configured_gene_coverage_lollipops(
    chart: Any,
    points: Sequence[Dict[str, Any]],
    *,
    use_log: bool,
    dark: bool,
    scale_mean_cnv: float,
    data_range: Optional[Tuple[float, float]] = None,
    fixed_axis_log2: Optional[float] = None,
    label_size_px: Optional[float] = None,
    labels_rotated: bool = True,
    x_span: Optional[float] = None,
) -> None:
    """Replace the abs-chart gene markers, or remove them when there are none."""
    series = chart.options.get("series")
    if not isinstance(series, list):
        return
    chart.options["series"] = [
        s
        for s in series
        if s.get("name")
        not in (
            _CONFIGURED_GENES_SERIES_NAME,
            _CONFIGURED_GENES_LABELS_SERIES_NAME,
        )
    ]
    prepared = _prepare_lollipop_points(
        points,
        use_log=use_log,
        scale_mean_cnv=scale_mean_cnv,
    )
    layout = _plan_lollipop_layout(
        prepared,
        use_log=use_log,
        scale_mean_cnv=scale_mean_cnv,
        data_range=data_range,
        fixed_axis_log2=fixed_axis_log2,
    )
    _apply_cnv_abs_y_window(
        chart,
        layout["y_lo"],
        layout["y_hi"],
        ticks=layout["ticks"],
        use_log=use_log,
    )
    if not prepared:
        return
    chart.options["series"].extend(
        _configured_gene_coverage_lollipop_series(
            layout["points"],
            layout["placements"],
            dark=dark,
            use_log=use_log,
            label_size_px=label_size_px,
            rotated=labels_rotated,
            x_span=x_span,
        )
    )


def cnv_export_settings(launcher: Any, sample_dir: Any) -> Tuple[bool, Dict[str, Any]]:
    """Resolve the render settings for a downloaded CNV PDF.

    Returns ``(use_log2, kwargs)`` for :func:`robin.reporting.cnv_export.build_cnv_pdf`.

    Everything resolves through the same overlay the PDF report uses: the live
    CNV panel wins where the reviewer has set something, and anything untouched
    falls back to the admin default. Reading the panel state alone meant a
    sample whose CNV section had not been opened this session exported on
    linear / outliers / calling-default / auto-fit no matter what the menus and
    the admin defaults said, so a download could disagree with the plot beside
    the button and with the report for the same sample.
    """
    from robin.gui.plotting_preferences import (
        cnv_label_is_rotated,
        resolve_cnv_chromosome_axis_log2,
        resolve_cnv_cutoff,
        resolve_cnv_gene_label_points,
        resolve_cnv_genome_axis_log2,
        resolve_cnv_gui_cutoff,
        resolve_cnv_gui_gene_coverage_filter,
        resolve_cnv_gui_label_orientation,
        resolve_cnv_gui_show_trend_line,
        resolve_cnv_gui_y_scale,
    )

    try:
        prefs = launcher._plotting_preferences_for_sample(sample_dir)
    except Exception:
        logging.debug(
            "Falling back to admin plotting preferences for the CNV export",
            exc_info=True,
        )
        prefs = getattr(launcher, "plotting_preferences", None)

    # y_scale is a live-only toggle with no preference field of its own, so it
    # falls back to the scale the admin default implies.
    try:
        state = launcher._cnv_state.get(str(sample_dir)) or {}
    except Exception:
        state = {}
    y_scale = state.get("y_scale")
    if y_scale not in ("linear", "log"):
        y_scale = resolve_cnv_gui_y_scale(prefs)

    gene_filter = resolve_cnv_gui_gene_coverage_filter(prefs)
    kwargs: Dict[str, Any] = dict(
        cutoff_override=resolve_cnv_cutoff(resolve_cnv_gui_cutoff(prefs)),
        outliers_only=gene_filter == _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
        show_trend=resolve_cnv_gui_show_trend_line(prefs),
        fixed_axis_log2=resolve_cnv_chromosome_axis_log2(prefs),
        genome_axis_log2=resolve_cnv_genome_axis_log2(prefs),
        gene_label_size=resolve_cnv_gene_label_points(prefs),
        gene_labels_rotated=cnv_label_is_rotated(
            resolve_cnv_gui_label_orientation(prefs)
        ),
    )
    return str(y_scale) == "log", kwargs


def _cnv_export_allowed(launcher: Any) -> bool:
    """Honour the launcher's export permission gate, as the other downloads do."""
    gate = getattr(launcher, "_require_export_or_notify", None)
    if gate is None:
        return True
    try:
        return bool(gate())
    except Exception:
        logging.debug("Export permission check failed", exc_info=True)
        return False


def _cnv_diff_y_window(
    data_range: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Symmetric window for the difference panel, sized to the data it holds."""
    span = _CNV_DIFF_Y_MIN_SPAN
    if data_range is not None:
        span = max(span, abs(float(data_range[0])), abs(float(data_range[1])))
    span = min(span, _CNV_DIFF_Y_MAX_SPAN)
    return (-span, span)


def _cnv_trend_color(dark: bool) -> str:
    return _CNV_TREND_COLOR_DARK if dark else _CNV_TREND_COLOR_LIGHT


def _cnv_trend_series(
    contig: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    dark: bool,
) -> Optional[Dict[str, Any]]:
    """Piecewise-constant segment bars over one chromosome's CNV bins.

    Flat within a segment and detached between them, as on the methylation-array
    CNV plots: a rolling average wanders with the noise, and a connected step line
    spikes wherever a short unmappable run forms its own segment.
    """
    data = cnv_segment_points(x_values, y_values)
    if not data:
        return None
    return {
        "type": "line",
        "name": f"{_CNV_TREND_SERIES_PREFIX}:{contig}",
        "yAxisIndex": 0,
        "data": data,
        "showSymbol": False,
        "symbol": "none",
        "smooth": False,
        # Each segment is its own horizontal bar; nulls keep the risers away.
        "connectNulls": False,
        "silent": True,
        "animation": False,
        "z": 9,
        "zlevel": 3,
        "lineStyle": {
            "color": _cnv_trend_color(dark),
            "width": 1.6,
            "opacity": 0.95,
        },
        "emphasis": {"disabled": True},
        "tooltip": {"show": False},
    }


def _cnv_reference_line_series(
    *,
    use_log: bool,
    dark: bool,
    chromosome: str,
    sex_estimate: str,
    y_lo: float,
    y_hi: float,
    cutoff_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Baseline and gain/loss cut-off guides for the CNV panel."""
    gain_threshold: Optional[float] = None
    loss_threshold: Optional[float] = None
    if use_log:
        gain_threshold, loss_threshold = _cnv_cutoff_thresholds(
            chromosome if chromosome and chromosome != "All" else "chr1",
            sex_estimate,
            cutoff_override,
        )

    levels = cnv_reference_levels(
        use_log=use_log,
        gain_threshold=gain_threshold,
        loss_threshold=loss_threshold,
        y_lo=y_lo,
        y_hi=y_hi,
    )
    if not levels:
        return None

    palette = _cnv_echart_palette(dark)
    cutoff_color = _CNV_CUTOFF_COLOR_DARK if dark else _CNV_CUTOFF_COLOR_LIGHT
    styles = {
        "baseline": {
            "color": palette["text"],
            "type": "solid",
            "width": 1.5,
            "opacity": 0.9,
        },
        # Calling cut-offs: dashed so they read as guides, but dark and thick
        # enough to be unmissable against the bin cloud.
        "threshold": {
            "color": cutoff_color,
            "type": [7, 5],
            "width": 1.8,
            "opacity": 0.95,
        },
        "ploidy": {
            "color": palette["muted"],
            "type": "dotted",
            "width": 1.0,
            "opacity": 0.5,
        },
    }
    mark_data = []
    for value, kind in levels:
        style = dict(styles.get(kind, styles["ploidy"]))
        label = {"show": False}
        if kind == "threshold":
            # Name the cut-off at the end of its own line.
            label = {
                "show": True,
                "position": "insideEndTop" if value > 0 else "insideEndBottom",
                "formatter": f"{value:+g}",
                "color": style["color"],
                "fontSize": 10,
                "fontWeight": "bold",
                "padding": [0, 4, 0, 0],
            }
        mark_data.append(
            {
                "yAxis": float(value),
                "lineStyle": style,
                "label": label,
            }
        )
    return {
        "type": "line",
        "name": _CNV_REFERENCE_SERIES_NAME,
        "yAxisIndex": 0,
        "data": [],
        "silent": True,
        "animation": False,
        "z": 2,
        "zlevel": 1,
        "tooltip": {"show": False},
        "markLine": {
            "symbol": "none",
            "animation": False,
            "silent": True,
            "data": mark_data,
        },
    }


@lru_cache(maxsize=1)
def _cnv_centromere_boundaries() -> Dict[str, int]:
    """p/q boundary per chromosome, empty when the resource is unavailable."""
    try:
        from robin.analysis.cnv_regional import load_centromere_boundaries

        return dict(load_centromere_boundaries())
    except Exception:
        logging.debug("Could not load centromere boundaries", exc_info=True)
        return {}


def _cnv_centromere_line_series(
    *,
    dark: bool,
    selected: str,
    chrom_offsets: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Faint dashed verticals separating the p and q arm of each chromosome."""
    boundaries = _cnv_centromere_boundaries()
    if not boundaries:
        return None

    color = _CNV_CENTROMERE_COLOR_DARK if dark else _CNV_CENTROMERE_COLOR_LIGHT
    style = {"color": color, "type": "dashed", "width": 1, "opacity": 1.0}
    mark_data = []
    if selected == "All":
        for contig, offset in chrom_offsets.items():
            boundary = boundaries.get(contig)
            if boundary is None:
                continue
            mark_data.append(
                {"xAxis": float(offset) + float(boundary), "lineStyle": style,
                 "label": {"show": False}}
            )
    else:
        boundary = boundaries.get(selected)
        if boundary is not None:
            mark_data.append(
                {"xAxis": float(boundary), "lineStyle": style, "label": {"show": False}}
            )
    if not mark_data:
        return None

    return {
        "type": "line",
        "name": _CNV_CENTROMERE_SERIES_NAME,
        "yAxisIndex": 0,
        "data": [],
        "silent": True,
        "animation": False,
        # Behind the bins, the segment line and the reference guides.
        "z": 1,
        "zlevel": 0,
        "tooltip": {"show": False},
        "markLine": {
            "symbol": "none",
            "animation": False,
            "silent": True,
            "data": mark_data,
        },
    }


def _upsert_cnv_centromere_lines(
    chart: Any,
    *,
    dark: bool,
    selected: str,
    chrom_offsets: Dict[str, float],
) -> None:
    """Insert (or refresh) the p/q arm dividers on a CNV chart."""
    series = chart.options.get("series")
    if not isinstance(series, list):
        return
    chart.options["series"] = [
        s for s in series if s.get("name") != _CNV_CENTROMERE_SERIES_NAME
    ]
    centromeres = _cnv_centromere_line_series(
        dark=dark, selected=selected, chrom_offsets=chrom_offsets
    )
    if centromeres is not None:
        chart.options["series"].insert(0, centromeres)


def _upsert_cnv_reference_lines(
    chart: Any,
    *,
    use_log: bool,
    dark: bool,
    chromosome: str,
    sex_estimate: str,
    y_lo: float,
    y_hi: float,
    cutoff_override: Optional[float] = None,
) -> None:
    """Insert (or refresh) the reference-line series on a CNV chart."""
    series = chart.options.get("series")
    if not isinstance(series, list):
        return
    chart.options["series"] = [
        s for s in series if s.get("name") != _CNV_REFERENCE_SERIES_NAME
    ]
    reference = _cnv_reference_line_series(
        use_log=use_log,
        dark=dark,
        chromosome=chromosome,
        sex_estimate=sex_estimate,
        y_lo=y_lo,
        y_hi=y_hi,
        cutoff_override=cutoff_override,
    )
    if reference is not None:
        chart.options["series"].insert(0, reference)


def _cnv_plot_bin_key_from_ui(value: Any) -> str:
    """Resolve NiceGUI select value (key, index, or event payload) to a plot-bin option key."""
    if value is None:
        return _CNV_PLOT_BIN_KEY_DEFAULT
    if isinstance(value, dict):
        inner = value.get("value", value.get("label"))
        if inner is not None and inner is not value:
            return _cnv_plot_bin_key_from_ui(inner)
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(_CNV_PLOT_BIN_KEYS_ORDERED):
            return _CNV_PLOT_BIN_KEYS_ORDERED[value]
    if isinstance(value, str):
        if value in _CNV_PLOT_BIN_KEY_TO_BP:
            return value
        for key, label in _CNV_PLOT_BIN_OPTIONS.items():
            if value == label:
                return key
    return _CNV_PLOT_BIN_KEY_DEFAULT


def _cnv_plot_bin_bp_from_ui(value: Any) -> Optional[int]:
    return _CNV_PLOT_BIN_KEY_TO_BP.get(_cnv_plot_bin_key_from_ui(value))


def _cnv_plot_bin_key_from_bp(bp: Optional[int]) -> str:
    if bp is None:
        return _CNV_PLOT_BIN_KEY_DEFAULT
    for key, width in _CNV_PLOT_BIN_KEY_TO_BP.items():
        if width == bp:
            return key
    return _CNV_PLOT_BIN_KEY_DEFAULT


def _cnv_gene_coverage_filter_from_ui(value: Any) -> str:
    """Normalize Coverage genes switch value to a filter mode key.

    Switch semantics: True / on = outliers only; False / off = all configured genes.
    """
    if isinstance(value, bool):
        return (
            _CNV_GENE_COVERAGE_FILTER_OUTLIERS
            if value
            else _CNV_GENE_COVERAGE_FILTER_ALL
        )
    if isinstance(value, dict):
        inner = value.get("value", value.get("label"))
        if inner is not None and inner is not value:
            return _cnv_gene_coverage_filter_from_ui(inner)
    vlow = str(value or "").strip().lower()
    if vlow in (
        _CNV_GENE_COVERAGE_FILTER_ALL,
        "all genes",
        "all",
        "false",
        "0",
        "off",
    ):
        return _CNV_GENE_COVERAGE_FILTER_ALL
    if vlow in (
        _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
        "outliers only",
        "outliers",
        "≠ average",
        "!= average",
        "vs average",
        "not average",
        "true",
        "1",
        "on",
    ):
        return _CNV_GENE_COVERAGE_FILTER_OUTLIERS
    return _CNV_GENE_COVERAGE_FILTER_OUTLIERS


def _cnv_contig_ok(contig: str) -> bool:
    """True if contig should be included in CNV plots (matches report behaviour)."""
    return contig in CNV_PLOT_CONTIGS


def _unwrap_cnv_track_map(raw: Any) -> Optional[Dict[str, np.ndarray]]:
    if not isinstance(raw, dict):
        return None
    if "cnv" in raw:
        inner = raw["cnv"]
        return inner if isinstance(inner, dict) else None
    return raw


def _cnv_sex_estimate_label(xy_val: Any) -> str:
    try:
        s = str(xy_val).strip().upper()
        if s in ("MALE", "XY"):
            return "Male"
        if s in ("FEMALE", "XX"):
            return "Female"
    except Exception:
        pass
    return "Unknown"


def _recompute_cnv_log2_state(state: Dict[str, Any]) -> None:
    """log2(ploidy / expected copy number) from the same track as the ploidy plot."""
    sample = _unwrap_cnv_track_map(state.get("cnv"))
    if not sample:
        state.pop("cnv_log2", None)
        return
    state["cnv_log2"] = resolve_cnv_calling_track(
        sample,
        _cnv_sex_estimate_label(state.get("xy")),
    )


def _cnv_genome_x_extent_bp(
    cnv_map: Dict[str, Any],
    binw_analysis: int,
    selected: str = "All",
) -> int:
    """Full genomic span in bp for the CNV scatter x-axis (independent of plot bin)."""
    if selected == "All":
        return sum(
            len(cnv_map[c]) * int(binw_analysis)
            for c in natsort.natsorted(cnv_map.keys())
            if _cnv_contig_ok(c)
        )
    chr_cnv = cnv_map.get(selected)
    return len(chr_cnv) * int(binw_analysis) if chr_cnv is not None else 0


def _cnv_set_genome_x_axis(chart: Any, x_axis_max: int) -> None:
    """Pin the x-axis to the full chromosome span (not downsampled data extent)."""
    xa = chart.options.setdefault("xAxis", {})
    if not isinstance(xa, dict):
        return
    xa.pop("max", None)  # drop dataMax so merges cannot keep a stale auto scale
    xa["type"] = "value"
    xa["min"] = 0
    xa["max"] = int(x_axis_max)
    xa["scale"] = True


def _cnv_reset_genome_x_data_zoom(chart: Any) -> None:
    """Reset the horizontal dataZoom to the full pinned x-axis span."""
    try:
        dz_list = chart.options.get("dataZoom")
        if not isinstance(dz_list, list) or not dz_list:
            return
        dz = dz_list[0]
        if not isinstance(dz, dict):
            return
        dz.pop("startValue", None)
        dz.pop("endValue", None)
        dz["start"] = 0
        dz["end"] = 100
    except Exception:
        pass


def _cnv_echarts_option_to_json(obj: Any) -> Any:
    """Convert ECharts option fragments to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _cnv_echarts_option_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cnv_echarts_option_to_json(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj) if isinstance(obj, np.floating) else int(obj)
        return None if (isinstance(obj, np.floating) and np.isnan(obj)) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _cnv_extract_js_options(obj: Any, js_values: Dict[str, str]) -> Any:
    """Swap NiceGUI ``:key`` JS options for tokens, recording the JS source.

    The raw ``setOption`` push below never runs NiceGUI's dynamic-property
    conversion, so these have to be re-inserted as unquoted JS afterwards.
    """
    if isinstance(obj, dict):
        converted: Dict[Any, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and key.startswith(":") and isinstance(value, str):
                token = f"__ROBIN_JS_{len(js_values)}__"
                js_values[token] = value
                converted[key[1:]] = token
            else:
                converted[key] = _cnv_extract_js_options(value, js_values)
        return converted
    if isinstance(obj, list):
        return [_cnv_extract_js_options(item, js_values) for item in obj]
    return obj


def _cnv_echart_push_update(chart: Any) -> None:
    """Replace the full ECharts option (NiceGUI merges by default and leaves stale scatter data)."""
    js_values: Dict[str, str] = {}
    options_clean = _cnv_echarts_option_to_json(
        _cnv_extract_js_options(chart.options, js_values)
    )
    opts_json = json.dumps(options_clean)
    # ``:setOption`` evaluates each argument as JS, so tokens become real functions.
    for token, js_source in js_values.items():
        opts_json = opts_json.replace(json.dumps(token), js_source)
    try:
        # Do not call chart.update() here: NiceGUI's update_chart uses setOption merge
        # unless the series count changes, which leaves stale per-chromosome scatter data
        # when only the plot bin width changes.
        chart.run_chart_method(":setOption", opts_json, '{"notMerge": true}')
    except Exception:
        logging.debug("CNV chart notMerge setOption failed", exc_info=True)


def _cnv_sample_relative_stats(
    cnv_map: Dict[str, np.ndarray],
    abs_plot_map: Optional[Dict[str, np.ndarray]],
    *,
    use_log: bool,
) -> Tuple[float, float]:
    """Sample-relative baseline from autosomes only, shared by all Up/Down coloring."""
    source_map = abs_plot_map if use_log and abs_plot_map else cnv_map
    default_mean = 0.0 if use_log else 2.0
    autosome_vals: List[float] = []
    for chrom, arr in source_map.items():
        if not chrom.startswith("chr") or not chrom[3:].isdigit():
            continue
        vals = np.asarray(arr, dtype=float)
        autosome_vals.extend(float(v) for v in vals if np.isfinite(v))
    if not autosome_vals:
        return default_mean, 1.0
    return float(np.mean(autosome_vals)), float(np.std(autosome_vals))


def _cnv_split_points_by_zscore(
    pts: List[List[float]],
    mean_val: float,
    std_val: float,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Partition points into high / low / normal by z-score."""
    high: List[List[float]] = []
    low: List[List[float]] = []
    norm: List[List[float]] = []
    for xi, vi in pts:
        z = (vi - mean_val) / std_val if std_val > 0 else 0.0
        (high if z > 0.5 else low if z < -0.5 else norm).append([xi, vi])
    return high, low, norm


def _build_cnv_track_scatter_series(
    track_map: Dict[str, np.ndarray],
    *,
    selected: str,
    binw_analysis: int,
    plot_bin_width: int,
    chrom_palette: List[str],
    filter_finite: bool = False,
    hide_unmappable_bands: bool = True,
) -> List[Dict[str, Any]]:
    """Build ECharts scatter series for a per-chromosome CNV track."""
    series: List[Dict[str, Any]] = []

    def _drop_unmappable(contig: str, x_bp: np.ndarray, vals: np.ndarray):
        """Hide bins with no uniquely mappable sequence.

        Bins the control profile cannot cover scatter far below the profile
        because their divisor is zero. Panel targets are never hidden. Display
        only - the track itself is unchanged.
        """
        if not hide_unmappable_bands:
            return x_bp, vals
        drop = unmappable_bin_mask(contig, x_bp, int(plot_bin_width))
        if not drop.any():
            return x_bp, vals
        return x_bp[~drop], vals[~drop]

    if selected == "All":
        offset_bp = 0
        dj = 0
        for contig, cnv in natsort.natsorted(track_map.items()):
            if not _cnv_contig_ok(contig):
                continue
            x_local, vals = downsample_cnv_for_plot(
                np.asarray(cnv), binw_analysis, int(plot_bin_width)
            )
            x_local, vals = _drop_unmappable(contig, x_local, vals)
            x_global = offset_bp + x_local
            if filter_finite:
                pts = [
                    [float(x), float(v)]
                    for x, v in zip(x_global.tolist(), vals.tolist())
                    if np.isfinite(v)
                ]
            else:
                pts = list(zip(x_global.tolist(), [float(v) for v in vals]))
            offset_bp += len(cnv) * binw_analysis
            series.append(
                {
                    "type": "scatter",
                    "name": contig,
                    "symbolSize": 3,
                    "itemStyle": {"color": chrom_palette[dj % len(chrom_palette)]},
                    "data": pts,
                }
            )
            dj += 1
    else:
        cnv = track_map.get(selected)
        if cnv is not None:
            x_local, vals = downsample_cnv_for_plot(
                np.asarray(cnv), binw_analysis, int(plot_bin_width)
            )
            x_local, vals = _drop_unmappable(selected, x_local, vals)
            if filter_finite:
                pts = [
                    [float(x), float(v)]
                    for x, v in zip(x_local.tolist(), vals.tolist())
                    if np.isfinite(v)
                ]
            else:
                pts = list(zip(x_local.tolist(), [float(v) for v in vals]))
            series.append(
                {
                    "type": "scatter",
                    "name": selected,
                    "symbolSize": 3,
                    "itemStyle": {"color": chrom_palette[0]},
                    "data": pts,
                }
            )
    return series


def _cnv_load_binary_payload(
    sample_dir: Path,
    *,
    cnv_dict_npy_changed: bool,
    cnv_npy_changed: bool,
    cnv3_npy_changed: bool,
    data_array_reload: bool,
    xy_pkl_changed: bool,
) -> Dict[str, Any]:
    """Load CNV numpy/pickle data from disk (no UI). Safe for ``asyncio.to_thread``."""
    out: Dict[str, Any] = {}
    cnv_dict_npy = sample_dir / "CNV_dict.npy"
    cnv_npy = sample_dir / "CNV.npy"
    cnv3_npy = sample_dir / "CNV3.npy"
    data_array_npy = sample_dir / "cnv_data_array.npy"
    xy_pkl = sample_dir / "XYestimate.pkl"

    if cnv_dict_npy_changed and cnv_dict_npy.exists():
        out["cnv_dict"] = np.load(cnv_dict_npy, allow_pickle=True).item()

    if xy_pkl_changed and xy_pkl.exists():
        try:
            with xy_pkl.open("rb") as f:
                out["xy"] = pickle.load(f)
        except Exception:
            pass

    if cnv_npy_changed and cnv_npy.exists():
        try:
            out["cnv"] = np.load(cnv_npy, allow_pickle=True).item()
        except Exception:
            out["cnv"] = None

    if cnv3_npy_changed and cnv3_npy.exists():
        try:
            out["cnv3"] = np.load(cnv3_npy, allow_pickle=True).item()
        except Exception:
            out["cnv3"] = None

    if data_array_reload and data_array_npy.exists():
        try:
            out["bp_array"] = np.load(data_array_npy, allow_pickle=True)
        except Exception:
            pass

    return out


def _is_dark_mode() -> bool:
    """Return normalized per-user dark mode."""
    return get_user_dark_mode(default=False)


def _cnv_chromosome_scatter_palette(dark: bool) -> List[str]:
    """Distinct scatter colours per chromosome.

    ECharts defaults include very dark greys that disappear on midnight backgrounds;
    dark mode uses lighter, saturated hues (design.md §5–§6).
    """
    if dark:
        return [
            "#34d399",
            "#38bdf8",
            "#fbbf24",
            "#fb7185",
            "#a78bfa",
            "#2dd4bf",
            "#f472b6",
            "#facc15",
            "#4ade80",
            "#60a5fa",
            "#f97316",
            "#e879f9",
            "#c084fc",
            "#22d3ee",
            "#fde047",
            "#93c5fd",
            "#f87171",
            "#bef264",
            "#5eead4",
            "#fcd34d",
            "#7dd3fc",
            "#fda4af",
            "#86efac",
            "#d8b4fe",
            "#eab308",
            "#67e8f9",
        ]
    return [
        "#059669",
        "#0284c7",
        "#b45309",
        "#dc2626",
        "#7c3aed",
        "#0d9488",
        "#db2777",
        "#ca8a04",
        "#16a34a",
        "#2563eb",
        "#ea580c",
        "#c026d3",
        "#9333ea",
        "#0891b2",
        "#ca8a04",
        "#3b82f6",
        "#ef4444",
        "#65a30d",
        "#14b8a6",
        "#eab308",
        "#0ea5e9",
        "#ec4899",
        "#4ade80",
        "#8b5cf6",
        "#ca8a04",
        "#06b6d4",
    ]


def _cnv_value_mode_colors(dark: bool) -> Tuple[str, str, str]:
    """High / low / in-range scatter colours (semantic green / rose / slate)."""
    if dark:
        return ("#34d399", "#fb7185", "#94a3b8")
    return ("#007AFF", "#FF3B30", "#8E8E93")


def _cnv_echart_palette(dark: bool) -> Dict[str, str]:
    """Axis, title, and tooltip colours for CNV scatter plots (design.md §5, §6)."""
    if dark:
        return {
            "text": "#e2e8f0",
            "muted": "#94a3b8",
            "axis_line": "#64748b",
            "split": "rgba(148, 163, 184, 0.52)",
            "tooltip_bg": "rgba(15, 23, 42, 0.96)",
            "tooltip_border": "#334155",
        }
    return {
        "text": "#0f172a",
        "muted": "#475569",
        "axis_line": "#94a3b8",
        "split": "rgba(71, 85, 105, 0.33)",
        "tooltip_bg": "rgba(255, 255, 255, 0.98)",
        "tooltip_border": "#e2e8f0",
    }


def _apply_cnv_scatter_performance(echart: Any) -> None:
    """Switch dense CNV scatter series onto ECharts' batched rendering path.

    Without this the higher point cap needed for a readable genome-wide profile
    would cost noticeable frame time on every live refresh.
    """
    try:
        series = echart.options.get("series")
        if not isinstance(series, list):
            return
        for s in series:
            if not isinstance(s, dict) or s.get("type") != "scatter":
                continue
            if s.get("name") in _CNV_OVERLAY_SERIES_NAMES:
                continue
            s["large"] = True
            s["largeThreshold"] = _CNV_LARGE_SCATTER_THRESHOLD
            s["progressive"] = _CNV_PROGRESSIVE_CHUNK
            s["progressiveThreshold"] = _CNV_LARGE_SCATTER_THRESHOLD
    except Exception:
        pass


def _apply_cnv_echart_chrome(echart: Any, dark: bool) -> None:
    """Apply light/dark readable chrome without touching series data."""
    p = _cnv_echart_palette(dark)
    try:
        o = echart.options
        if not isinstance(o, dict):
            return
        o["backgroundColor"] = "transparent"
        o["textStyle"] = {"color": p["text"]}
        title = o.get("title")
        if isinstance(title, dict):
            title["textStyle"] = {"color": p["text"], "fontSize": 14}
        o["tooltip"] = {
            **(o.get("tooltip") or {}),
            "backgroundColor": p["tooltip_bg"],
            "borderColor": p["tooltip_border"],
            "textStyle": {"color": p["text"]},
        }
        xa = o.get("xAxis")
        if isinstance(xa, dict):
            xa["axisLine"] = {"lineStyle": {"color": p["axis_line"]}}
            xa["axisLabel"] = {**(xa.get("axisLabel") or {}), "color": p["muted"]}
            xa["splitLine"] = {"lineStyle": {"color": p["split"]}}
        ya_list = o.get("yAxis")
        if isinstance(ya_list, list):
            for ya in ya_list:
                if not isinstance(ya, dict):
                    continue
                ya["axisLine"] = {"lineStyle": {"color": p["axis_line"]}}
                ya["axisLabel"] = {
                    **(ya.get("axisLabel") or {}),
                    "color": p["muted"],
                    ":formatter": _CNV_Y_VALUE_FORMATTER_JS,
                }
                ya["nameTextStyle"] = {"color": p["muted"]}
                ya["splitLine"] = {"lineStyle": {"color": p["split"]}}
        leg = o.get("legend")
        if isinstance(leg, dict):
            leg["textStyle"] = {"color": p["muted"]}
        try:
            dz_list = o.get("dataZoom")
            if isinstance(dz_list, list):
                for dz in dz_list:
                    if not isinstance(dz, dict):
                        continue
                    if dz.get("yAxisIndex") is not None:
                        # Handle labels otherwise show the raw data min/max.
                        dz[":labelFormatter"] = _CNV_Y_VALUE_FORMATTER_JS
                    dz["borderColor"] = p["axis_line"]
                    dz["fillerColor"] = (
                        "rgba(51, 65, 85, 0.35)"
                        if dark
                        else "rgba(148, 163, 184, 0.2)"
                    )
                    dz["handleStyle"] = {
                        "color": p["text"],
                        "borderColor": p["axis_line"],
                    }
                    dz["moveHandleStyle"] = {"color": p["muted"]}
                    dz["emphasis"] = {
                        "handleStyle": {"borderColor": p["text"]},
                    }
                    dbb = dz.get("dataBackground") or {}
                    if isinstance(dbb, dict):
                        dbb["lineStyle"] = {
                            **(dbb.get("lineStyle") or {}),
                            "color": p["muted"],
                        }
                        dbb["areaStyle"] = {
                            **(dbb.get("areaStyle") or {}),
                            "color": (
                                "rgba(148, 163, 184, 0.12)"
                                if dark
                                else "rgba(71, 85, 105, 0.08)"
                            ),
                        }
                        dz["dataBackground"] = dbb
        except Exception:
            pass
    except Exception:
        pass


def add_cnv_section(launcher: Any, sample_dir: Path) -> None:
    """Build the CNV UI section and attach refresh timers.

    Uses `launcher._cnv_state` for per-sample cache/state.
    Expects CNV.npy, CNV3.npy, CNV_dict.npy, XYestimate.pkl, cnv_data_array.npy in sample folder.
    The top chart can toggle between absolute ploidy and log2(ploidy / expected copy number).

    Controls now trigger immediate refresh instead of waiting for timer updates.
    """
    configured_gene_names: tuple[str, ...] = ()
    try:
        from robin.workflow_config import get_cnv_genes, load_workflow_toml

        workflow_config = None
        workflow_toml_path = getattr(launcher, "workflow_toml_path", None)
        if workflow_toml_path:
            workflow_config = load_workflow_toml(Path(workflow_toml_path))
        configured_gene_names = get_cnv_genes(workflow_config)
    except Exception:
        logging.warning("Could not resolve [cnv].genes for GUI plots", exc_info=True)
    configured_gene_locations = _load_cnv_gene_locations(configured_gene_names)
    if configured_gene_names:
        found = {str(row["gene"]).casefold() for row in configured_gene_locations}
        missing = [name for name in configured_gene_names if name.casefold() not in found]
        if missing:
            logging.warning(
                "No packaged GRCh38 location found for configured CNV genes: %s",
                ", ".join(missing),
            )

    with ui.element("div").classes("w-full min-w-0").props("id=analysis-detail-cnv"):
        with ui.element("div").classes("classification-insight-shell w-full min-w-0"):
            ui.label("Copy number (CNV)").classes(
                "classification-insight-heading text-headline-small"
            )
            with ui.element("div").classes(
                "classification-insight-card w-full min-w-0"
            ):
                with ui.column().classes("w-full min-w-0 gap-2 p-2 md:p-3"):
                    with ui.row().classes("items-center gap-2 min-w-0"):
                        ui.icon("person").classes("classification-insight-icon")
                        ui.label("Genome-wide profile").classes(
                            "classification-insight-model flex-1 min-w-0"
                        )
                    cnv_status = ui.label("Status: Awaiting Data").classes(
                        "classification-insight-result w-full"
                    )
                    cnv_xy = ui.label("Genetic sex: --").classes(
                        "classification-insight-meta w-full"
                    )
                    cnv_whole_chr_summary = ui.label("Whole chromosome: --").classes(
                        "classification-insight-meta w-full"
                    )
                    cnv_arm_summary = ui.label("Arm-level: --").classes(
                        "classification-insight-meta w-full"
                    )
                    with ui.row().classes(
                        "w-full justify-end gap-4 flex-wrap items-baseline"
                    ):
                        cnv_bin = ui.label("Bin width: --").classes(
                            "classification-insight-meta"
                        )
                        cnv_var = ui.label("Variance: --").classes(
                            "classification-insight-meta"
                        )
            with ui.card().classes(
                "classification-insight-card w-full min-w-0 mt-2"
            ):
                with ui.column().classes("w-full min-w-0 gap-2 p-2 md:p-3"):
                    with ui.row().classes("items-center gap-2 min-w-0"):
                        ui.icon("donut_large").classes("classification-insight-icon")
                        ui.label("CNV load").classes(
                            "classification-insight-model flex-1 min-w-0"
                        )
                    cnv_load_total = ui.label("Total: --").classes(
                        "classification-insight-result w-full"
                    )
                    with ui.row().classes(
                        "w-full gap-4 flex-wrap items-baseline"
                    ):
                        cnv_load_gain = ui.label("Gain: --").classes(
                            "classification-insight-meta"
                        )
                        cnv_load_loss = ui.label("Loss: --").classes(
                            "classification-insight-meta"
                        )
                        cnv_load_assessed = ui.label("Assessed: --").classes(
                            "classification-insight-meta"
                        )
                    ui.label(
                        "Proportion of the assessed genome past the calling "
                        "cut-off — the same threshold drawn on the plots."
                    ).classes("classification-insight-foot w-full")
            with ui.row().classes(
                "w-full gap-3 items-center mb-2 flex-wrap mt-2"
            ):
                ui.label("Chromosome").classes("classification-insight-meta")
                cnv_chrom_select = ui.select(options={"All": "All"}, value="All").style(
                    "width: 160px"
                )
                ui.label("Gene").classes("classification-insight-meta ml-2")
                cnv_gene_select = ui.select(options={"All": "All"}, value="All").style(
                    "width: 200px"
                )
                _cnv_ui_state = launcher._cnv_state.setdefault(str(sample_dir), {})
                from robin.gui.plotting_preferences import (
                    resolve_cnv_gui_color_mode,
                    resolve_cnv_gui_gene_coverage_filter,
                    resolve_cnv_gui_show_breakpoints,
                    resolve_cnv_gui_chrom_axis,
                    resolve_cnv_gui_genome_axis,
                    resolve_cnv_gui_cutoff,
                    resolve_cnv_gui_gene_label_size,
                    resolve_cnv_gui_label_orientation,
                    resolve_cnv_gui_show_trend_line,
                    resolve_cnv_gui_y_scale,
                )
                from robin.gui.plotting_preferences import (
                    cnv_chrom_axis_options,
                    cnv_genome_axis_options,
                    cnv_cutoff_options,
                    cnv_gene_label_size_options,
                )

                _plot_prefs = getattr(launcher, "plotting_preferences", None)
                if "gene_coverage_filter" not in _cnv_ui_state:
                    _cnv_ui_state["gene_coverage_filter"] = (
                        resolve_cnv_gui_gene_coverage_filter(_plot_prefs)
                    )
                if "color_mode" not in _cnv_ui_state:
                    _cnv_ui_state["color_mode"] = resolve_cnv_gui_color_mode(
                        _plot_prefs
                    )
                if "y_scale" not in _cnv_ui_state:
                    _cnv_ui_state["y_scale"] = resolve_cnv_gui_y_scale(_plot_prefs)
                if "show_bp" not in _cnv_ui_state:
                    _cnv_ui_state["show_bp"] = resolve_cnv_gui_show_breakpoints(
                        _plot_prefs
                    )
                if "show_trend" not in _cnv_ui_state:
                    _cnv_ui_state["show_trend"] = resolve_cnv_gui_show_trend_line(
                        _plot_prefs
                    )
                if "cutoff" not in _cnv_ui_state:
                    _cnv_ui_state["cutoff"] = resolve_cnv_gui_cutoff(_plot_prefs)
                if "chrom_axis" not in _cnv_ui_state:
                    _cnv_ui_state["chrom_axis"] = resolve_cnv_gui_chrom_axis(
                        _plot_prefs
                    )
                if "genome_axis" not in _cnv_ui_state:
                    _cnv_ui_state["genome_axis"] = resolve_cnv_gui_genome_axis(
                        _plot_prefs
                    )
                if "gene_label_size" not in _cnv_ui_state:
                    _cnv_ui_state["gene_label_size"] = (
                        resolve_cnv_gui_gene_label_size(_plot_prefs)
                    )
                if "label_orientation" not in _cnv_ui_state:
                    _cnv_ui_state["label_orientation"] = (
                        resolve_cnv_gui_label_orientation(_plot_prefs)
                    )

                ui.label("Coverage genes").classes("classification-insight-meta ml-2")
                _gene_cov_filter = _cnv_ui_state.get("gene_coverage_filter")
                if _gene_cov_filter not in _CNV_GENE_COVERAGE_FILTERS:
                    _gene_cov_filter = resolve_cnv_gui_gene_coverage_filter(_plot_prefs)
                _cnv_ui_state["gene_coverage_filter"] = _gene_cov_filter
                with ui.row().classes("items-center gap-1"):
                    ui.label("All").classes("classification-insight-meta")
                    cnv_gene_cov_filter = (
                        ui.switch(
                            value=_gene_cov_filter
                            == _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
                        )
                        .props("dense")
                        .tooltip(
                            "Left: all configured genes · Right: outliers only"
                        )
                    )
                    cnv_gene_cov_filter.value = (
                        _gene_cov_filter == _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                    )
                    ui.label("Outliers").classes("classification-insight-meta")
                ui.label("Color by").classes("classification-insight-meta ml-2")
                _color_mode = _cnv_ui_state.get("color_mode", "chromosome")
                if _color_mode not in ("chromosome", "value"):
                    _color_mode = resolve_cnv_gui_color_mode(_plot_prefs)
                _cnv_ui_state["color_mode"] = _color_mode
                with ui.row().classes("items-center gap-1"):
                    ui.label("Chromosome").classes("classification-insight-meta")
                    cnv_color = (
                        ui.switch(value=_color_mode == "value")
                        .props("dense")
                        .tooltip(
                            "Left: colour by chromosome · Right: gain/loss (up/down)"
                        )
                    )
                    cnv_color.value = _color_mode == "value"
                    ui.label("Up/Down").classes("classification-insight-meta")
                ui.label("Y-axis").classes("classification-insight-meta ml-2")
                default_y_scale = resolve_cnv_gui_y_scale(_plot_prefs)
                _y_scale = _cnv_ui_state.get("y_scale", default_y_scale)
                if _y_scale not in ("linear", "log"):
                    _y_scale = default_y_scale
                _cnv_ui_state["y_scale"] = _y_scale
                with ui.row().classes("items-center gap-1"):
                    ui.label("Linear").classes("classification-insight-meta")
                    cnv_scale = (
                        ui.switch(value=_y_scale == "log")
                        .props("dense")
                        .tooltip(
                            "Left: linear ploidy · Right: log2(ploidy / expected)"
                        )
                    )
                    cnv_scale.value = _y_scale == "log"
                    ui.label("Log2").classes("classification-insight-meta")
                ui.label("Cut-off").classes("classification-insight-meta ml-2")
                _cutoff_options = cnv_cutoff_options()
                _cutoff_value = str(_cnv_ui_state.get("cutoff") or "calling")
                if _cutoff_value not in _cutoff_options:
                    _cutoff_value = "calling"
                _cnv_ui_state["cutoff"] = _cutoff_value
                cnv_cutoff = (
                    ui.select(options=_cutoff_options, value=_cutoff_value)
                    .style("width: 150px")
                    .tooltip(
                        "Gain/loss cut-off used everywhere for this sample: the "
                        "plot lines and colouring, the Outliers filter, the "
                        "events, regional and NGTD tables, the gene states, CNV "
                        "load, and any report or plot exported from here."
                    )
                )
                ui.label("Gene label").classes("classification-insight-meta ml-2")
                _label_options = cnv_gene_label_size_options()
                _label_value = str(_cnv_ui_state.get("gene_label_size") or "4.5")
                if _label_value not in _label_options:
                    _label_value = "4.5"
                _cnv_ui_state["gene_label_size"] = _label_value
                cnv_gene_label = (
                    ui.select(options=_label_options, value=_label_value)
                    .style("width: 165px")
                    .tooltip(
                        "Size of the gene name labels on the plots and in "
                        "downloaded PDFs. Point sizes are as rendered in the report."
                    )
                )
                ui.label("Label style").classes("classification-insight-meta ml-2")
                _orientation = str(
                    _cnv_ui_state.get("label_orientation") or "rotated"
                )
                _cnv_ui_state["label_orientation"] = _orientation
                with ui.row().classes("items-center gap-1"):
                    ui.label("Horizontal").classes("classification-insight-meta")
                    cnv_label_orient = (
                        ui.switch(value=_orientation == "rotated")
                        .props("dense")
                        .tooltip(
                            "Left: gene names read horizontally · "
                            "Right: rotated alongside the marker (array style)"
                        )
                    )
                    cnv_label_orient.value = _orientation == "rotated"
                    ui.label("Portrait").classes("classification-insight-meta")
                ui.label("Chr Y-range").classes("classification-insight-meta ml-2")
                _axis_options = cnv_chrom_axis_options()
                _axis_value = str(_cnv_ui_state.get("chrom_axis") or "2")
                if _axis_value not in _axis_options:
                    _axis_value = "2"
                _cnv_ui_state["chrom_axis"] = _axis_value
                cnv_chrom_axis = (
                    ui.select(options=_axis_options, value=_axis_value)
                    .style("width: 150px")
                    .tooltip(
                        "Fixed Y range for single-chromosome views and the "
                        "per-chromosome plots, so chromosomes can be compared. "
                        "Bins outside it are flagged at the panel edge."
                    )
                )
                ui.label("Genome Y-range").classes(
                    "classification-insight-meta ml-2"
                )
                _genome_axis_options = cnv_genome_axis_options()
                _genome_axis_value = str(_cnv_ui_state.get("genome_axis") or "auto")
                if _genome_axis_value not in _genome_axis_options:
                    _genome_axis_value = "auto"
                _cnv_ui_state["genome_axis"] = _genome_axis_value
                cnv_genome_axis = (
                    ui.select(
                        options=_genome_axis_options, value=_genome_axis_value
                    )
                    .style("width: 150px")
                    .tooltip(
                        "Fixed Y range for the genome-wide view and its PDF. "
                        "Auto fits the axis to the data, which is the default: "
                        "unlike the per-chromosome plots there is nothing to "
                        "compare it against."
                    )
                )
                ui.label("Segment line").classes("classification-insight-meta ml-2")
                with ui.row().classes("items-center gap-1"):
                    ui.label("Hide").classes("classification-insight-meta")
                    cnv_trend = (
                        ui.switch(value=bool(_cnv_ui_state.get("show_trend", True)))
                        .props("dense")
                        .tooltip(
                            "Piecewise-constant segment line over the CNV bins"
                        )
                    )
                    cnv_trend.value = bool(_cnv_ui_state.get("show_trend", True))
                    ui.label("Show").classes("classification-insight-meta")
                ui.label("Plot bin").classes("classification-insight-meta ml-2")
                cnv_plot_bin = ui.select(
                    options=_CNV_PLOT_BIN_OPTIONS,
                    value=_CNV_PLOT_BIN_KEY_DEFAULT,
                ).style("width: 120px")
                cnv_bp_label = ui.label("Breakpoints").classes(
                    "classification-insight-meta ml-2"
                ).style("display: none")
                with ui.row().classes("items-center gap-1").style(
                    "display: none"
                ) as cnv_bp_row:
                    ui.label("Hide").classes("classification-insight-meta")
                    cnv_bp = (
                        ui.switch(value=bool(_cnv_ui_state.get("show_bp", True)))
                        .props("dense")
                        .tooltip(
                            "Show candidate breakpoint markers on single-chromosome view"
                        )
                    )
                    cnv_bp.value = bool(_cnv_ui_state.get("show_bp", True))
                    ui.label("Show").classes("classification-insight-meta")
                with ui.row().classes("items-center gap-2 ml-auto"):
                    cnv_pdf_genome_btn = (
                        ui.button("Genome PDF", icon="download")
                        .props("dense outline no-caps size=sm")
                        .tooltip(
                            "Download the genome-wide CNV plot as a vector PDF"
                        )
                    )
                    cnv_pdf_chrom_btn = (
                        ui.button("Chromosome PDFs", icon="download")
                        .props("dense outline no-caps size=sm")
                        .tooltip(
                            "Download every per-chromosome CNV plot as one "
                            "multi-page vector PDF"
                        )
                    )
            with ui.element("div").classes("w-full target-coverage-panel__plot-wrap"):
                cnv_abs = ui.echart(
                    {
                        "backgroundColor": "transparent",
                        "title": {"text": "CNV scatter plot", "left": "center", "top": 10},
                        "grid": {
                            "left": "5%",
                            "right": "5%",
                            "bottom": "10%",
                            "top": "20%",
                            "containLabel": True,
                        },
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "value", "min": 0},
                        "yAxis": [
                            {"type": "value", "name": "Ploidy"},
                            {
                                "type": "value",
                                "name": "Coverage (x)",
                                "position": "right",
                                "show": False,
                                "min": 0,
                            },
                        ],
                        "dataZoom": [
                            {"type": "slider", "xAxisIndex": [0]},
                            {
                                "type": "slider",
                                "yAxisIndex": [0],
                                "right": 20,
                                "startValue": 0,
                                "endValue": 6,
                            },
                        ],
                        "series": [
                            {"type": "scatter", "name": "CNV", "symbolSize": 3, "data": []},
                            {
                                "type": "scatter",
                                "name": "centromeres_highlight",
                                "data": [],
                                "symbolSize": 3,
                                "markArea": {
                                    "itemStyle": {"color": "rgba(135, 206, 250, 0.4)"},
                                    "data": [],
                                },
                            },
                            {
                                "type": "scatter",
                                "name": "cytobands_highlight",
                                "data": [],
                                "symbolSize": 3,
                                "markArea": {
                                    "itemStyle": {"color": "rgba(200, 200, 200, 0.4)"},
                                    "data": [],
                                },
                                "markLine": {"symbol": "none", "data": []},
                            },
                        ],
                    }
                ).classes("w-full h-[32rem] cnv-genome-abs-chart")
            with ui.element("div").classes("w-full target-coverage-panel__plot-wrap mt-2"):
                cnv_diff = ui.echart(
                    {
                        "backgroundColor": "transparent",
                        "title": {"text": "Difference plot", "left": "center", "top": 10},
                        "grid": {
                            "left": "5%",
                            "right": "5%",
                            "bottom": "10%",
                            "top": "20%",
                            "containLabel": True,
                        },
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "value", "min": 0},
                        "yAxis": [
                            {"type": "value", "name": "Relative"},
                            {
                                "type": "value",
                                "name": "Coverage (x)",
                                "position": "right",
                                "show": False,
                                "min": 0,
                            },
                        ],
                        "dataZoom": [
                            {"type": "slider", "xAxisIndex": [0]},
                            {
                                "type": "slider",
                                "yAxisIndex": [0],
                                "right": 20,
                                "startValue": -4,
                                "endValue": 4,
                            },
                        ],
                        "series": [
                            {
                                "type": "scatter",
                                "name": "CNV Δ",
                                "symbolSize": 3,
                                "data": [],
                            },
                            {
                                "type": "scatter",
                                "name": "centromeres_highlight",
                                "data": [],
                                "symbolSize": 3,
                                "markArea": {
                                    "itemStyle": {"color": "rgba(135, 206, 250, 0.4)"},
                                    "data": [],
                                },
                            },
                            {
                                "type": "scatter",
                                "name": "cytobands_highlight",
                                "data": [],
                                "symbolSize": 3,
                                "markArea": {
                                    "itemStyle": {"color": "rgba(200, 200, 200, 0.4)"},
                                    "data": [],
                                },
                                "markLine": {"symbol": "none", "data": []},
                            },
                        ],
                    }
                ).classes("w-full h-[26rem] cnv-genome-diff-chart")
            genome_charts = (cnv_abs, cnv_diff)

            ui.separator().classes("mgmt-detail-separator")
            regional_cnv_label = ui.label("Regional CNV events").classes(
                "target-coverage-panel__meta-label mt-2 mb-1"
            )
            regional_cnv_summary = ui.label("No regional CNV events detected").classes(
                "classification-insight-meta mb-2"
            )
            regional_cnv_columns = [
                {"name": "chrom", "label": "Chr", "field": "chrom", "sortable": True},
                {"name": "region", "label": "Region", "field": "region", "sortable": True},
                {
                    "name": "start_mb",
                    "label": "Start (Mb)",
                    "field": "start_mb",
                    "sortable": True,
                    "align": "right",
                },
                {
                    "name": "end_mb",
                    "label": "End (Mb)",
                    "field": "end_mb",
                    "sortable": True,
                    "align": "right",
                },
                {
                    "name": "length_mb",
                    "label": "Length (Mb)",
                    "field": "length_mb",
                    "sortable": True,
                    "align": "right",
                },
                {
                    "name": "mean_cnv",
                    "label": "Mean CNV",
                    "field": "mean_cnv",
                    "sortable": True,
                    "align": "right",
                },
                {
                    "name": "state",
                    "label": "State",
                    "field": "state",
                    "sortable": True,
                    "align": "center",
                },
                {"name": "panel_genes", "label": "Panel genes", "field": "panel_genes"},
            ]
            _, regional_cnv_table = styled_table(
                columns=regional_cnv_columns, rows=[], pagination=20, class_size="table-xs"
            )
            try:
                regional_cnv_table.props('multi-sort rows-per-page-options="[10,20,50,0]"')
            except Exception:
                pass

            ui.separator().classes("mgmt-detail-separator")
            cnv_events_label = ui.label("Arm / whole-chromosome CNV events").classes(
                "target-coverage-panel__meta-label mt-2 mb-1"
            )
            cnv_events_summary = ui.label("No CNV events detected").classes(
                "classification-insight-meta mb-2"
            )
        
        # CNV Events Table
        cnv_events_columns = [
            {"name": "chromosome", "label": "Chr", "field": "chromosome", "sortable": True},
            {"name": "event_type", "label": "Event Type", "field": "event_type", "sortable": True},
            {"name": "arm", "label": "Arm", "field": "arm", "sortable": True},
            {"name": "start_mb", "label": "Start (Mb)", "field": "start_mb", "sortable": True, "align": "right"},
            {"name": "end_mb", "label": "End (Mb)", "field": "end_mb", "sortable": True, "align": "right"},
            {"name": "length_mb", "label": "Length (Mb)", "field": "length_mb", "sortable": True, "align": "right"},
            {"name": "mean_cnv_str", "label": "Mean CNV", "field": "mean_cnv_str", "sortable": True, "align": "right"},
            {"name": "confidence", "label": "Confidence", "field": "confidence", "sortable": True, "align": "center"},
            {"name": "proportion_affected", "label": "% Affected", "field": "proportion_affected", "sortable": True, "align": "right"},
            {"name": "genes_str", "label": "Genes", "field": "genes_str"},
        ]
        _, cnv_events_table = styled_table(
            columns=cnv_events_columns, rows=[], pagination=20, class_size="table-xs"
        )
        try:
            cnv_events_table.props('multi-sort rows-per-page-options="[10,20,50,0]"')
        except Exception:
            pass

        ui.separator().classes("mgmt-detail-separator")
        ngtd_label = ui.label("NGTD and Step 2 targets CNVs").classes(
            "target-coverage-panel__meta-label mt-2 mb-1"
        )
        ngtd_summary = ui.label("Awaiting CNV data").classes(
            "classification-insight-meta mb-2"
        )
        ngtd_columns = [
            {"name": "gene", "label": "Gene", "field": "gene", "sortable": True},
            {"name": "chrom", "label": "Chr", "field": "chrom", "sortable": True},
            {
                "name": "value",
                "label": "Log2 ratio",
                "field": "value",
                "sortable": True,
                "align": "right",
            },
            {
                "name": "state",
                "label": "Result",
                "field": "state",
                "sortable": True,
                "align": "center",
            },
        ]
        _, ngtd_table = styled_table(
            columns=ngtd_columns, rows=[], pagination=20, class_size="table-xs"
        )
        try:
            ngtd_table.props('multi-sort rows-per-page-options="[10,20,50,0]"')
        except Exception:
            pass
        ui.label(
            "Every configured target is listed. \u201cNo CNVs Detected\u201d means the gene was "
            "assessed and nothing crossed the cut-off; \u201cNot on panel\u201d and "
            "\u201cNo data yet\u201d mean it was not assessed, which is not the same thing."
        ).classes("classification-insight-foot mb-2")

    # Adaptive thinning helpers. The cap is what limits how much of the CNV track
    # actually reaches the genome-wide panel, so it is set well above the number of
    # analysis bins in a human genome and ECharts' fast scatter path carries it.
    MAX_POINTS_PER_CHART = 40000

    def _get_visible_range(chart, series_list):
        try:
            # Compute overall data span
            min_x = None
            max_x = None
            for s in series_list:
                data = s.get("data") or []
                if not data:
                    continue
                sx = data[0][0] if isinstance(data[0], (list, tuple)) else None
                ex = data[-1][0] if isinstance(data[-1], (list, tuple)) else None
                if sx is None or ex is None:
                    # Fallback compute min/max
                    for p in data:
                        try:
                            x = float(p[0])
                        except Exception:
                            continue
                        min_x = x if min_x is None else min(min_x, x)
                        max_x = x if max_x is None else max(max_x, x)
                else:
                    vmin = min(float(sx), float(ex))
                    vmax = max(float(sx), float(ex))
                    min_x = vmin if min_x is None else min(min_x, vmin)
                    max_x = vmax if max_x is None else max(max_x, vmax)
            dz = None
            try:
                dzo = chart.options.get("dataZoom")
                if isinstance(dzo, list) and dzo:
                    dz = dzo[0]
            except Exception:
                dz = None
            if not dz:
                return (min_x, max_x)
            # Prefer explicit values
            sv = dz.get("startValue") if isinstance(dz, dict) else None
            ev = dz.get("endValue") if isinstance(dz, dict) else None
            if sv is not None or ev is not None:
                left = float(sv) if sv is not None else min_x
                right = float(ev) if ev is not None else max_x
                return (left, right)
            # Fallback to percentage range
            sp = dz.get("start") if isinstance(dz, dict) else None
            ep = dz.get("end") if isinstance(dz, dict) else None
            if (
                (sp is not None or ep is not None)
                and min_x is not None
                and max_x is not None
            ):
                width = (
                    max_x - min_x if max_x is not None and min_x is not None else None
                )
                if width and width > 0:
                    left = min_x + (float(sp or 0) / 100.0) * width
                    right = min_x + (float(ep or 100) / 100.0) * width
                    return (left, right)
            return (min_x, max_x)
        except Exception:
            return (None, None)

    def _evenly_sample(seq, k):
        try:
            n = len(seq)
            if k <= 0 or n <= k:
                return list(seq)
            if k == 1:
                return [seq[n // 2]]
            # Choose k indices evenly across [0, n-1]
            return [seq[int(round(i * (n - 1) / (k - 1)))] for i in range(k)]
        except Exception:
            return list(seq)[:k]

    def _thin_chart_series(chart, max_points: int = MAX_POINTS_PER_CHART) -> None:
        try:
            series = chart.options.get("series", [])
            # Identify data series to thin (exclude overlays)
            data_idx = []
            data_series = []
            for idx, s in enumerate(series):
                name = s.get("name", "")
                if (
                    s.get("type") == "scatter"
                    and name not in _CNV_OVERLAY_SERIES_NAMES
                ):
                    data = s.get("data") or []
                    if isinstance(data, list) and data:
                        data_idx.append(idx)
                        data_series.append(s)
            if not data_series:
                return
            x_range = _get_visible_range(chart, data_series)
            # Gather visible counts and data within range
            vis_data = []
            total = 0
            left, right = x_range
            for s in data_series:
                pts = s.get("data") or []
                if left is not None and right is not None:
                    sub = [
                        p
                        for p in pts
                        if isinstance(p, (list, tuple)) and left <= float(p[0]) <= right
                    ]
                else:
                    sub = list(pts)
                vis_data.append(sub)
                total += len(sub)
            if total <= max_points:
                return
            # Allocate budgets proportional to visible counts with a small floor
            budgets = []
            
            for sub in vis_data:
                share = int(max(1, round((len(sub) / total) * max_points)))
                budgets.append(share)
            # Normalize budgets to exactly max_points
            adj = sum(budgets) - max_points
            i = 0
            while adj != 0 and budgets:
                if adj > 0 and budgets[i] > 1:
                    budgets[i] -= 1
                    adj -= 1
                elif adj < 0:
                    budgets[i] += 1
                    adj += 1
                i = (i + 1) % len(budgets)
            # Apply sampling and replace data (preserve points outside range sparsely)
            for (idx, s), sub, k in zip(zip(data_idx, data_series), vis_data, budgets):
                original = s.get("data") or []
                # Keep outside-range points sparsely so context remains when zoomed out/in
                if left is not None and right is not None:
                    outside = [
                        p
                        for p in original
                        if isinstance(p, (list, tuple))
                        and not (left <= float(p[0]) <= right)
                    ]
                    outside_keep = _evenly_sample(
                        outside, max(0, k // 10)
                    )  # at most 10% of budget
                else:
                    outside_keep = []
                inside_keep = _evenly_sample(sub, max(1, k - len(outside_keep)))
                new_data = inside_keep + outside_keep
                chart.options["series"][idx]["data"] = new_data
        except Exception:
            pass

    @lru_cache(maxsize=1)
    def _load_centromere_regions() -> Dict[str, List[Tuple[int, int, str]]]:
        """Load centromere/satellite regions from packaged resources.
        Returns mapping: chrom -> list of (start_bp, end_bp, name).
        """
        regions: Dict[str, List[Tuple[int, int, str]]] = {}
        try:
            res_path = (
                importlib_resources.files("robin.resources") / "cenSatRegions.bed"
            )
            with res_path.open("r") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) < 4:
                        continue
                    chrom, start, end, name = (
                        parts[0],
                        int(parts[1]),
                        int(parts[2]),
                        parts[3],
                    )
                    regions.setdefault(chrom, []).append((start, end, name))
        except Exception:
            pass
        return regions

    @lru_cache(maxsize=1)
    def _load_cytobands_df() -> pd.DataFrame:
        try:
            res_path = importlib_resources.files("robin.resources") / "cytoBand.txt"
            df = pd.read_csv(
                res_path,
                sep="\t",
                header=None,
                names=["chrom", "start_pos", "end_pos", "name", "stain"],
            )
            return df
        except Exception:
            return pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "name", "stain"])

    @lru_cache(maxsize=1)
    def _load_centromere_bed_df() -> pd.DataFrame:
        try:
            res_path = importlib_resources.files("robin.resources") / "cenSatRegions.bed"
            return pd.read_csv(
                res_path,
                sep="\t",
                header=None,
                names=["chrom", "start_pos", "end_pos", "name"],
                usecols=[0, 1, 2, 3],
            )
        except Exception:
            return pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "name"])

    _EMPTY_GENE_BED = pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "gene"])

    def _analyze_cytoband_cnv(
        cnv_data: Dict[str, np.ndarray],
        chromosome: str,
        bin_width: int,
        sex_estimate: str,
        cutoff_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """Run shared regional cytoband analysis (same logic as PDF reports)."""
        try:
            return analyze_cytoband_cnv(
                cnv_data,
                chromosome,
                {"bin_width": int(bin_width)},
                _load_cytobands_df(),
                _load_centromere_bed_df(),
                _EMPTY_GENE_BED,
                sex_estimate,
                cutoff_override=cutoff_override,
            )
        except Exception:
            return pd.DataFrame()

    def _load_gene_bed(sample_dir: Path = None) -> pd.DataFrame:
        """Load gene BED file based on the analysis panel used for the sample"""
        try:
            # Determine which panel to use
            panel = ""  # No default fallback
            if sample_dir:
                try:
                    master_csv_path = sample_dir / "master.csv"
                    if master_csv_path.exists():
                        import pandas as pd
                        df = pd.read_csv(master_csv_path)
                        if not df.empty and "analysis_panel" in df.columns:
                            panel_val = df.iloc[0]["analysis_panel"]
                            if panel_val and str(panel_val).strip() != "":
                                panel = str(panel_val).strip()
                except Exception:
                    pass
            
            # Map panel to BED filename
            bed_filename = None
            if not panel:
                # No panel found - return empty DataFrame
                return pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "gene"])
            elif panel == "rCNS2":
                bed_filename = "rCNS2_panel_name_uniq.bed"
            elif panel == "AML":
                bed_filename = "AML_panel_name_uniq.bed"
            else:
                # Check for custom panel
                bed_filename = f"{panel}_panel_name_uniq.bed"
            
            # Try to load the panel-specific BED file
            try:
                res_path = importlib_resources.files("robin.resources") / bed_filename
                if res_path.exists():
                    return pd.read_csv(
                        res_path,
                        sep="\t",
                        header=None,
                        names=["chrom", "start_pos", "end_pos", "gene"],
                    )
            except Exception:
                pass
            
            # Fallback to unique_genes.bed if panel-specific file not found
            try:
                res_path = importlib_resources.files("robin.resources") / "unique_genes.bed"
                if res_path.exists():
                    return pd.read_csv(
                        res_path,
                        sep="\t",
                        header=None,
                        names=["chrom", "start_pos", "end_pos", "gene"],
                    )
            except Exception:
                pass
                
        except Exception:
            pass
            
        return pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "gene"])

    def _sex_label(xy_val: Any) -> str:
        try:
            s = str(xy_val).strip().upper()
            if s in ("MALE", "XY"):
                return "Male"
            if s in ("FEMALE", "XX"):
                return "Female"
        except Exception:
            pass
        return "Unknown"

    def _get_cytoband_cnv_summary(
        cnv_data: Dict[str, np.ndarray],
        chromosome: str,
        bin_width: int,
        sex_estimate: str,
        cutoff_override: Optional[float] = None,
    ) -> str:
        try:
            df = _analyze_cytoband_cnv(
                cnv_data, chromosome, bin_width, sex_estimate, cutoff_override
            )
            if df.empty:
                return "No regional CNV events detected"
            significant = df[df["cnv_state"].isin(SIGNIFICANT_CNV_STATES)]
            gains = significant[significant["cnv_state"].isin({"GAIN", "HIGH_GAIN"})]
            losses = significant[significant["cnv_state"].isin({"LOSS", "DEEP_LOSS"})]
            parts: List[str] = []
            if not gains.empty:
                parts.append(
                    "Gains: "
                    + ", ".join(
                        f"{r['name']} ({r['mean_cnv']:.2f})"
                        for _, r in gains.iterrows()
                    )
                )
            if not losses.empty:
                parts.append(
                    "Losses: "
                    + ", ".join(
                        f"{r['name']} ({r['mean_cnv']:.2f})"
                        for _, r in losses.iterrows()
                    )
                )
            return "\n".join(parts) if parts else "No regional CNV events detected"
        except Exception:
            return "No CNV data available"

    def _compute_all_cytoband_df(
        cnv_data: Dict[str, np.ndarray],
        bin_width: int,
        sex_estimate: str,
        cutoff_override: Optional[float] = None,
    ) -> pd.DataFrame:
        try:
            frames: List[pd.DataFrame] = []
            for chrom in natsort.natsorted(cnv_data.keys()):
                if not is_reportable_chromosome(chrom):
                    continue
                df = _analyze_cytoband_cnv(
                    cnv_data, chrom, bin_width, sex_estimate, cutoff_override
                )
                if not df.empty:
                    frames.append(df)
            if frames:
                out = pd.concat(frames, ignore_index=True)
                if not out.empty:
                    def _rank(label: Any) -> int:
                        try:
                            s = str(label)
                            if s.startswith("chr"):
                                s = s[3:]
                            mapping = {"X": 23, "Y": 24, "M": 25}
                            return int(s) if s.isdigit() else mapping.get(s, 1000)
                        except Exception:
                            return 1000

                    out["_chrom_rank"] = out["chrom"].map(_rank)
                    out = out.sort_values(["_chrom_rank", "start_pos"]).drop(
                        columns=["_chrom_rank"]
                    )
                return out
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _build_regional_rows(
        cytoband_df: pd.DataFrame, panel_genes_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        try:
            events = build_regional_cnv_events(cytoband_df, panel_genes_df)
            return [format_regional_event_table_row(event) for event in events]
        except Exception:
            return []

    def _update_cnv_events_analysis(state: Dict[str, Any]) -> None:
        """Update CNV events analysis using centralized classification rules."""
        try:
            cnv_map = state.get("cnv")
            if isinstance(cnv_map, dict) and "cnv" in cnv_map:
                cnv_map = cnv_map["cnv"]

            if not cnv_map:
                cnv_events_table.rows = []
                cnv_events_summary.set_text("No CNV data available")
                cnv_whole_chr_summary.set_text("Whole chromosome: --")
                cnv_arm_summary.set_text("Arm-level: --")
                return

            binw = state.get("cnv_dict", {}).get("bin_width", 1000000)
            sex_lbl = _sex_label(state.get("xy"))
            data, calling_binw = prepare_cnv_calling_track(
                cnv_map,
                int(binw),
                _cnv_sex_estimate_label(state.get("xy")),
            )

            if data and calling_binw:
                # Load cytobands and genes
                cyto_df = _load_cytobands_df()
                gene_df = _load_gene_bed(sample_dir)
                
                # Detect CNV events using centralized rules
                events = detect_cnv_events(
                    cnv_data=data,
                    bin_width=int(calling_binw),
                    sex_estimate=sex_lbl,
                    cytobands_df=cyto_df,
                    gene_df=gene_df,
                    cutoff_override=_resolve_cnv_cutoff(state.get("cutoff")),
                )
                _set_cutoff_heading(
                    cnv_events_label,
                    "Arm / whole-chromosome CNV events",
                    _resolve_cnv_cutoff(state.get("cutoff")),
                )

                # Update events table
                events_rows = []
                for event in events:
                    event_dict = event.to_dict()
                    # Format proportion as percentage
                    event_dict["proportion_affected"] = f"{event.proportion_affected:.1%}"
                    events_rows.append(event_dict)
                
                cnv_events_table.rows = events_rows
                try:
                    cnv_events_table.update()
                except Exception:
                    pass
                
                # Update summaries (insight card + events section)
                whole_text, arm_text = format_cnv_events_card_lines(events)
                cnv_whole_chr_summary.set_text(whole_text)
                cnv_arm_summary.set_text(arm_text)
                cnv_events_summary.set_text(format_cnv_events_section_summary(events))
            else:
                cnv_events_table.rows = []
                cnv_whole_chr_summary.set_text("Whole chromosome: --")
                cnv_arm_summary.set_text("Arm-level: --")
                cnv_events_summary.set_text("CNV data not available")
        except Exception as e:
            logging.error(f"Error updating CNV events analysis: {e}")
            cnv_events_table.rows = []
            cnv_whole_chr_summary.set_text("Whole chromosome: --")
            cnv_arm_summary.set_text("Arm-level: --")
            cnv_events_summary.set_text("Error analyzing CNV events")

    def _refresh_ngtd_table(state: Dict[str, Any]) -> None:
        """Fill the NGTD target table from the current log2 track."""
        try:
            from robin.analysis.cnv_regional import (
                compute_target_gene_cnv_states,
                load_gene_bed,
                load_panel_gene_bed,
            )
            from robin.workflow_config import get_cnv_ngtd_genes

            genes = tuple(get_cnv_ngtd_genes())
            track = state.get("cnv_log2") or {}
            bin_width = int((state.get("cnv_dict") or {}).get("bin_width") or 0)
            if not genes or not track or bin_width <= 0:
                ngtd_table.rows = []
                ngtd_summary.set_text("Awaiting CNV data")
                return
            _panel_name, panel_df = load_panel_gene_bed(str(sample_dir))
            frames = [f for f in (panel_df, load_gene_bed()) if f is not None and not f.empty]
            rows = compute_target_gene_cnv_states(
                track,
                bin_width,
                genes,
                _cnv_sex_estimate_label(state.get("xy")),
                gene_frames=frames or None,
                cutoff_override=_resolve_cnv_cutoff(state.get("cutoff")),
            )
        except Exception:
            logging.debug("Could not build the NGTD target table", exc_info=True)
            return
        ngtd_table.rows = [
            {
                "gene": row["gene"],
                "chrom": row["chrom"] or "--",
                "value": f"{row['value']:+.2f}" if row["value"] is not None else "--",
                "state": row["state"],
            }
            for row in rows
        ]
        called = sum(1 for row in rows if row["state"] in ("GAIN", "LOSS"))
        _set_cutoff_heading(
            ngtd_label,
            "NGTD and Step 2 targets CNVs",
            _resolve_cnv_cutoff(state.get("cutoff")),
        )
        ngtd_summary.set_text(
            f"{len(rows)} targets checked; {called} with a called gain or loss"
        )
        try:
            ngtd_table.update()
        except Exception:
            pass

    def _refresh_cnv_load_labels(state: Dict[str, Any]) -> None:
        """Update the CNV load card from the current log2 track."""
        try:
            from robin.analysis.cnv_regional import compute_cnv_load

            track = state.get("cnv_log2") or {}
            bin_width = int((state.get("cnv_dict") or {}).get("bin_width") or 0)
            load = (
                compute_cnv_load(
                    track,
                    bin_width,
                    _cnv_sex_estimate_label(state.get("xy")),
                    cutoff_override=_resolve_cnv_cutoff(state.get("cutoff")),
                )
                if track and bin_width > 0
                else None
            )
        except Exception:
            logging.debug(
                "Could not compute CNV load for the GUI", exc_info=True
            )
            load = None
        if not load or not load.get("assessed_mb"):
            cnv_load_total.set_text("Total: --")
            cnv_load_gain.set_text("Gain: --")
            cnv_load_loss.set_text("Loss: --")
            cnv_load_assessed.set_text("Assessed: --")
            return
        label = load.get("cutoff_label")
        cnv_load_total.set_text(
            f"Total: {load['total_percent']:.1f}% "
            f"({load['total_mb']:,.0f} Mb)"
            + (f"  \u00b7  cut-off {label}" if label else "")
        )
        cnv_load_gain.set_text(
            f"Gain: {load['gain_percent']:.1f}% ({load['gain_mb']:,.0f} Mb)"
        )
        cnv_load_loss.set_text(
            f"Loss: {load['loss_percent']:.1f}% ({load['loss_mb']:,.0f} Mb)"
        )
        cnv_load_assessed.set_text(
            f"Assessed: {load['assessed_mb']:,.0f} Mb"
        )

    def _render_cnv_from_state(state: Dict[str, Any]) -> None:
        try:
            _recompute_cnv_log2_state(state)
            _refresh_cnv_load_labels(state)
            _refresh_ngtd_table(state)
            cnv_map = state.get("cnv")
            cnv3_map = state.get("cnv3")
            cnv_log2_map = _unwrap_cnv_track_map(state.get("cnv_log2"))
            if isinstance(cnv_map, dict) and "cnv" in cnv_map:
                cnv_map = cnv_map["cnv"]
            if isinstance(cnv3_map, dict) and "cnv" in cnv3_map:
                cnv3_map = cnv3_map["cnv"]
            if not cnv_map:
                return
            dark_ui = _is_dark_mode()
            chrom_palette = _cnv_chromosome_scatter_palette(dark_ui)
            col_high, col_low, col_norm = _cnv_value_mode_colors(dark_ui)
            chrom_divider = _cnv_echart_palette(dark_ui)["muted"]
            binw_analysis = state.get("cnv_dict", {}).get("bin_width", 1_000_000)
            plot_bin_width = resolve_cnv_plot_bin_width(
                int(binw_analysis),
                state.get("plot_bin_width"),
            )
            # Keep plot bin width dropdown in sync with state
            try:
                pb = state.get("plot_bin_width")
                want_key = _cnv_plot_bin_key_from_bp(pb)
                if getattr(cnv_plot_bin, "value", None) != want_key:
                    cnv_plot_bin.value = want_key
                    cnv_plot_bin.update()
            except Exception:
                pass
            try:
                want_filter = str(
                    state.get(
                        "gene_coverage_filter",
                        _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
                    )
                )
                if want_filter not in _CNV_GENE_COVERAGE_FILTERS:
                    want_filter = _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                # Switch is boolean: True = outliers, False = all.
                want_outliers = want_filter == _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                current = getattr(cnv_gene_cov_filter, "value", None)
                if current is not want_outliers:
                    cnv_gene_cov_filter.value = want_outliers
                    cnv_gene_cov_filter.update()
            except Exception:
                pass
            selected = state.get("selected_chrom", "All")
            use_log = state.get("y_scale", "linear") == "log"
            abs_plot_map = (
                cnv_log2_map
                if use_log and cnv_log2_map
                else cnv_map
            )
            raw_color_mode = state.get("color_mode", "chromosome")
            # normalize color mode to expected keys
            lval = str(raw_color_mode).strip().lower()
            if lval in ("chromosome", "chromosomes"):
                color_mode = "chromosome"
            elif lval in (
                "value",
                "up/down",
                "updown",
                "up_down",
                "updown ",
                "up down",
            ):
                color_mode = "value"
            else:
                color_mode = "chromosome"

            cnv_abs.options["yAxis"][0]["type"] = "value"
            cnv_abs.options["yAxis"][0].pop("logBase", None)
            # Clear any previous pinned Y window; marker overlay re-applies auto-scale.
            cnv_abs.options["yAxis"][0].pop("min", None)
            cnv_abs.options["yAxis"][0].pop("max", None)
            if use_log:
                cnv_abs.options["yAxis"][0]["name"] = "Log2 ratio (ploidy / expected)"
                cnv_abs.options["title"]["text"] = "CNV scatter plot"
                cnv_abs.options["title"]["top"] = 4
                cnv_abs.options["title"]["subtext"] = (
                    "log2(ploidy / expected copy number); 0 = normal"
                )
                cnv_abs.options["grid"]["top"] = "26%"
            else:
                cnv_abs.options["yAxis"][0]["name"] = "Ploidy"
                cnv_abs.options["title"]["text"] = "CNV scatter plot"
                cnv_abs.options["title"]["top"] = 10
                cnv_abs.options["title"].pop("subtext", None)
                cnv_abs.options["grid"]["top"] = "20%"
            # Temporary baseline until gene-marker auto-scale runs at the end of render.
            try:
                _apply_cnv_abs_y_window(
                    cnv_abs,
                    -2.0 if use_log else 0.0,
                    2.0 if use_log else 6.0,
                    use_log=use_log,
                )
            except Exception:
                pass
            # X-axis is always in genomic base pairs; use actual genome/chromosome length
            # so the scale does not change when plot bin width changes (dataMax would shrink
            # with fewer downsampled points).
            x_axis_max = _cnv_genome_x_extent_bp(cnv_map, int(binw_analysis), selected)
            _cnv_set_genome_x_axis(cnv_abs, x_axis_max)
            _cnv_set_genome_x_axis(cnv_diff, x_axis_max)
            # Clear any previous zoom constraints when viewing All
            if selected == "All":
                _cnv_reset_genome_x_data_zoom(cnv_abs)
                _cnv_reset_genome_x_data_zoom(cnv_diff)
            logging.debug(
                f"CNV render: selected={selected}, y_scale={state.get('y_scale')}, color_mode={state.get('color_mode')}"
            )
            sample_rel_mean, sample_rel_std = _cnv_sample_relative_stats(
                cnv_map,
                abs_plot_map if isinstance(abs_plot_map, dict) else None,
                use_log=use_log,
            )
            try:
                coverage_by_gene = load_gene_target_coverage(sample_dir)
            except Exception:
                logging.debug(
                    "Could not load gene target coverage for CNV lollipops",
                    exc_info=True,
                )
                coverage_by_gene = {}

            # Robust value envelope of the plotted bins, filled in once the scatter
            # series exist; drives the y-window so labels get their own gutters.
            abs_data_range: Dict[str, Optional[Tuple[float, float]]] = {"range": None}
            sex_estimate_label = _cnv_sex_estimate_label(state.get("xy"))
            show_trend = bool(state.get("show_trend", True))
            cutoff_override = _resolve_cnv_cutoff(state.get("cutoff"))
            gene_label_px = _resolve_cnv_gene_label_px(state.get("gene_label_size"))
            labels_rotated = _resolve_cnv_labels_rotated(state.get("label_orientation"))
            # The two views have separate Y-range settings: a fixed window lets
            # chromosomes be compared with each other, while the genome-wide view
            # has nothing to compare against and so defaults to fitting the data.
            fixed_axis_log2 = (
                _resolve_cnv_chrom_axis(state.get("chrom_axis"))
                if selected != "All"
                else _resolve_cnv_chrom_axis(state.get("genome_axis"))
            )

            def _apply_abs_gene_coverage_overlay() -> None:
                """Coverage lollipops on the abs chart only (position markers if no coverage)."""
                filter_mode = str(
                    state.get(
                        "gene_coverage_filter",
                        _CNV_GENE_COVERAGE_FILTER_OUTLIERS,
                    )
                )
                if filter_mode not in _CNV_GENE_COVERAGE_FILTERS:
                    filter_mode = _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                plot_map = (
                    abs_plot_map if isinstance(abs_plot_map, dict) else cnv_map
                )
                points, _mean_cov = _build_configured_gene_coverage_points(
                    configured_gene_locations,
                    selected=selected,
                    chrom_offsets=chrom_offsets,
                    abs_plot_map=plot_map,
                    bin_width=int(binw_analysis),
                    coverage_by_gene=coverage_by_gene,
                    filter_mode=filter_mode,
                    use_log=use_log,
                    scale_mean_cnv=float(sample_rel_mean),
                    sex_estimate=sex_estimate_label,
                    cutoff_override=cutoff_override,
                )
                _upsert_configured_gene_coverage_lollipops(
                    cnv_abs,
                    points,
                    use_log=use_log,
                    dark=dark_ui,
                    scale_mean_cnv=float(sample_rel_mean),
                    data_range=abs_data_range["range"],
                    fixed_axis_log2=fixed_axis_log2,
                    label_size_px=gene_label_px,
                    labels_rotated=labels_rotated,
                    x_span=float(x_axis_max) if x_axis_max else None,
                )
                _upsert_cnv_centromere_lines(
                    cnv_abs,
                    dark=dark_ui,
                    selected=selected,
                    chrom_offsets=chrom_offsets,
                )
                y_axis = (cnv_abs.options.get("yAxis") or [{}])[0]
                _upsert_cnv_reference_lines(
                    cnv_abs,
                    use_log=use_log,
                    dark=dark_ui,
                    chromosome=selected,
                    sex_estimate=sex_estimate_label,
                    y_lo=float(y_axis.get("min", 0.0)),
                    y_hi=float(y_axis.get("max", 1.0)),
                    cutoff_override=cutoff_override,
                )
                if points:
                    return
                # Fall back to position markers when target coverage is unavailable.
                if configured_gene_locations and not coverage_by_gene:
                    _upsert_configured_gene_series(
                        cnv_abs,
                        configured_gene_locations,
                        selected=selected,
                        chrom_offsets=chrom_offsets,
                        dark=dark_ui,
                    )

            # Absolute plot
            series_abs = []
            trend_series_abs: List[Dict[str, Any]] = []
            # Prepare chromosome partitions for labels/areas when viewing All
            chrom_bounds = []  # list of (name, start_bp, end_bp)
            chrom_offsets: Dict[str, float] = {}
            if selected == "All":
                # X-axis is in genomic base pairs; chromosome bounds use actual lengths.
                offset_bp = 0
                for contig, cnv in natsort.natsorted(cnv_map.items()):
                    if not _cnv_contig_ok(contig):
                        continue
                    plot_cnv = abs_plot_map.get(contig, cnv)
                    x_local, vals = downsample_cnv_for_plot(
                        np.asarray(plot_cnv), binw_analysis, int(plot_bin_width)
                    )
                    x_global = offset_bp + x_local
                    if use_log:
                        pts = [
                            [float(x), float(v)]
                            for x, v in zip(x_global.tolist(), vals.tolist())
                            if np.isfinite(v)
                        ]
                    else:
                        pts = list(zip(x_global.tolist(), [float(v) for v in vals]))
                    if show_trend:
                        trend = _cnv_trend_series(
                            contig,
                            x_global.tolist(),
                            vals.tolist(),
                            dark=dark_ui,
                        )
                        if trend is not None:
                            trend_series_abs.append(trend)
                    start_bp = offset_bp
                    end_bp = offset_bp + len(cnv) * binw_analysis
                    chrom_offsets[contig] = start_bp
                    chrom_bounds.append((contig, start_bp, end_bp))
                    offset_bp = end_bp
                    if color_mode == "chromosome":
                        ci = len(
                            [s for s in series_abs if s.get("type") == "scatter"]
                        )
                        series_abs.append(
                            {
                                "type": "scatter",
                                "name": contig,
                                "symbolSize": 3,
                                "itemStyle": {
                                    "color": chrom_palette[
                                        ci % len(chrom_palette)
                                    ]
                                },
                                "data": pts,
                            }
                        )
                    else:
                        high, low, norm = _cnv_split_points_by_zscore(
                            pts,
                            sample_rel_mean,
                            sample_rel_std,
                        )
                        if high:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"High {contig}",
                                    "symbolSize": 4,
                                    "itemStyle": {"color": col_high},
                                    "data": high,
                                }
                            )
                        if low:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"Low {contig}",
                                    "symbolSize": 4,
                                    "itemStyle": {"color": col_low},
                                    "data": low,
                                }
                            )
                        if norm:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"Normal {contig}",
                                    "symbolSize": 2,
                                    "itemStyle": {"color": col_norm},
                                    "data": norm,
                                }
                            )
            else:
                plot_cnv = abs_plot_map.get(selected)
                if plot_cnv is None:
                    plot_cnv = cnv_map.get(selected)
                if plot_cnv is not None:
                    x_local, vals = downsample_cnv_for_plot(
                        np.asarray(plot_cnv), binw_analysis, int(plot_bin_width)
                    )
                    if use_log:
                        pts = [
                            [float(x), float(v)]
                            for x, v in zip(x_local.tolist(), vals.tolist())
                            if np.isfinite(v)
                        ]
                    else:
                        pts = list(zip(x_local.tolist(), [float(v) for v in vals]))
                    if show_trend:
                        trend = _cnv_trend_series(
                            selected,
                            x_local.tolist(),
                            vals.tolist(),
                            dark=dark_ui,
                        )
                        if trend is not None:
                            trend_series_abs.append(trend)
                    if color_mode == "chromosome":
                        series_abs.append(
                            {
                                "type": "scatter",
                                "name": selected,
                                "symbolSize": 3,
                                "itemStyle": {"color": chrom_palette[0]},
                                "data": pts,
                            }
                        )
                    else:
                        high, low, norm = _cnv_split_points_by_zscore(
                            pts,
                            sample_rel_mean,
                            sample_rel_std,
                        )
                        if high:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"High {selected}",
                                    "symbolSize": 4,
                                    "itemStyle": {"color": col_high},
                                    "data": high,
                                }
                            )
                        if low:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"Low {selected}",
                                    "symbolSize": 4,
                                    "itemStyle": {"color": col_low},
                                    "data": low,
                                }
                            )
                        if norm:
                            series_abs.append(
                                {
                                    "type": "scatter",
                                    "name": f"Normal {selected}",
                                    "symbolSize": 2,
                                    "itemStyle": {"color": col_norm},
                                    "data": norm,
                                }
                            )
            # Robust envelope from the un-thinned bins, before overlays are added.
            abs_data_range["range"] = _cnv_scatter_value_range(series_abs)
            # Preserve highlight series (centromeres, cytobands) and replace data series only
            keep = [
                s
                for s in cnv_abs.options["series"]
                if s.get("name") in ("centromeres_highlight", "cytobands_highlight")
            ]
            cnv_abs.options["series"] = series_abs + trend_series_abs + keep
            # Baseline / threshold guides are added by the overlay pass below so they
            # survive the later series rebuilds.
            _apply_abs_gene_coverage_overlay()
            # Build background chromosome areas and vertical labels when showing All
            try:
                if selected == "All" and chrom_bounds:
                    # Alternating shaded bands per chromosome for readability
                    areas_data = []
                    lines_data = []
                    for contig, start_bp, end_bp in chrom_bounds:
                        areas_data.append(
                            [{"xAxis": float(start_bp)}, {"xAxis": float(end_bp)}]
                        )
                        # Label at the center of chromosome region
                        center_bp = (start_bp + end_bp) / 2
                        lines_data.append(
                            {
                                "xAxis": float(center_bp),
                                "lineStyle": {
                                    "type": "dashed",
                                    "color": chrom_divider,
                                },
                                "label": {
                                    "show": True,
                                    "formatter": contig,
                                    "color": chrom_divider,
                                },
                            }
                        )

                    # Helper to set overlays by series name regardless of index
                    def _apply_overlays(
                        chart, band_areas, band_lines, centro_areas_list
                    ):
                        try:
                            idx_cyto = next(
                                (
                                    i
                                    for i, s in enumerate(chart.options["series"])
                                    if s.get("name") == "cytobands_highlight"
                                ),
                                None,
                            )
                            idx_centro = next(
                                (
                                    i
                                    for i, s in enumerate(chart.options["series"])
                                    if s.get("name") == "centromeres_highlight"
                                ),
                                None,
                            )
                            if idx_cyto is not None:
                                # Remove background shading per request; keep dashed vertical lines only
                                chart.options["series"][idx_cyto]["markArea"][
                                    "data"
                                ] = []
                                chart.options["series"][idx_cyto].setdefault(
                                    "markLine", {}
                                )
                                chart.options["series"][idx_cyto]["markLine"][
                                    "data"
                                ] = band_lines
                                # Disable animation for marker lines so they appear instantly
                                chart.options["series"][idx_cyto]["markLine"][
                                    "animation"
                                ] = False
                            # Do not show centromeres in All view
                            if idx_centro is not None:
                                chart.options["series"][idx_centro]["markArea"][
                                    "data"
                                ] = []
                        except Exception:
                            pass

                    _apply_overlays(cnv_abs, areas_data, lines_data, [])
                    _apply_overlays(cnv_diff, areas_data, lines_data, [])
                else:
                    # Clear overlays when focusing on a single chromosome
                    def _clear_overlays(chart):
                        try:
                            for s in chart.options["series"]:
                                if s.get("name") in (
                                    "cytobands_highlight",
                                    "centromeres_highlight",
                                ):
                                    if "markArea" in s and "data" in s["markArea"]:
                                        s["markArea"]["data"] = []
                                    if "markLine" in s and "data" in s["markLine"]:
                                        s["markLine"]["data"] = []
                        except Exception:
                            pass

                    _clear_overlays(cnv_abs)
                    _clear_overlays(cnv_diff)
                    # In single-chromosome view, draw centromeres, cytobands (by state), genes, and breakpoint candidates
                    if selected != "All":
                        try:
                            centro = _load_centromere_regions()
                            idx_centro_abs = next(
                                (
                                    i
                                    for i, s in enumerate(cnv_abs.options["series"])
                                    if s.get("name") == "centromeres_highlight"
                                ),
                                None,
                            )
                            idx_centro_diff = next(
                                (
                                    i
                                    for i, s in enumerate(cnv_diff.options["series"])
                                    if s.get("name") == "centromeres_highlight"
                                ),
                                None,
                            )
                            areas = []
                            for s, e, _n in centro.get(selected, []):
                                areas.append([{"xAxis": float(s)}, {"xAxis": float(e)}])
                            if idx_centro_abs is not None:
                                cnv_abs.options["series"][idx_centro_abs]["markArea"][
                                    "data"
                                ] = areas
                            if idx_centro_diff is not None:
                                cnv_diff.options["series"][idx_centro_diff]["markArea"][
                                    "data"
                                ] = areas
                        except Exception:
                            pass
                        # Cytobands colored by CNV state (from CNV3 values)
                        try:
                            idx_cyto_abs = next(
                                (
                                    i
                                    for i, s in enumerate(cnv_abs.options["series"])
                                    if s.get("name") == "cytobands_highlight"
                                ),
                                None,
                            )
                            if (
                                idx_cyto_abs is not None
                                and cnv3_map
                                and selected in cnv3_map
                            ):
                                cyto_df = _load_cytobands_df()
                                bands = cyto_df[cyto_df["chrom"] == selected]
                                vals = (
                                    np.array(cnv3_map[selected])
                                    if isinstance(
                                        cnv3_map[selected], (list, np.ndarray)
                                    )
                                    else np.array([])
                                )
                                band_areas = []
                                
                                # Get CNV events for this chromosome to highlight significant events
                                events = []
                                try:
                                    sex_lbl = _sex_label(state.get("xy"))
                                    gene_df = _load_gene_bed(sample_dir)
                                    calling_map, calling_binw = prepare_cnv_calling_track(
                                        cnv_map,
                                        int(binw_analysis),
                                        _cnv_sex_estimate_label(state.get("xy")),
                                    )
                                    call_vals = calling_map.get(selected)
                                    if call_vals is not None:
                                        events = detect_cnv_events(
                                            cnv_data={selected: np.asarray(call_vals)},
                                            bin_width=int(calling_binw),
                                            sex_estimate=sex_lbl,
                                            cytobands_df=cyto_df,
                                            gene_df=gene_df,
                                            cutoff_override=_resolve_cnv_cutoff(
                                                state.get("cutoff")
                                            ),
                                        )
                                except Exception:
                                    pass
                                
                                # Create event lookup for highlighting
                                event_regions = {}
                                for event in events:
                                    key = f"{event.start_pos}-{event.end_pos}"
                                    event_regions[key] = event
                                
                                for _, row in bands.iterrows():
                                    s_bp, e_bp = int(row["start_pos"]), int(row["end_pos"])
                                    s_bin = max(0, s_bp // binw_analysis)
                                    e_bin = min(len(vals) - 1, max(0, e_bp // binw_analysis))
                                    if len(vals) > 0 and e_bin >= s_bin:
                                        mean_val = float(
                                            np.mean(vals[s_bin : e_bin + 1])
                                        )
                                    else:
                                        mean_val = 0.0
                                    
                                    # Check if this region has a significant CNV event
                                    region_key = f"{s_bp}-{e_bp}"
                                    event = event_regions.get(region_key)
                                    
                                    fill_neutral = (
                                        "rgba(255, 255, 255, 0.07)"
                                        if dark_ui
                                        else "rgba(0, 0, 0, 0.03)"
                                    )
                                    if event:
                                        # Highlight significant events with stronger colors
                                        if event.event_type in ("GAIN", "WHOLE_CHR_GAIN"):
                                            color = "rgba(52, 199, 89, 0.3)"  # gains
                                        elif event.event_type in ("LOSS", "WHOLE_CHR_LOSS"):
                                            color = "rgba(255, 45, 85, 0.3)"  # losses
                                        else:
                                            color = fill_neutral
                                    else:
                                        # Standard cytoband coloring
                                        if mean_val > 0.5:
                                            color = "rgba(52, 199, 89, 0.12)"
                                        elif mean_val < -0.5:
                                            color = "rgba(255, 45, 85, 0.12)"
                                        else:
                                            color = fill_neutral

                                    band_areas.append(
                                        [
                                            {
                                                "name": str(row["name"]),
                                                "xAxis": float(s_bp),
                                                "itemStyle": {"color": color},
                                                "label": {
                                                    "show": True,
                                                    "position": "insideTop",
                                                    "color": (
                                                        "#cbd5e1"
                                                        if dark_ui
                                                        else "#555"
                                                    ),
                                                    "fontSize": 11,
                                                },
                                            },
                                            {"xAxis": float(e_bp)},
                                        ]
                                    )
                                cnv_abs.options["series"][idx_cyto_abs]["markArea"][
                                    "data"
                                ] = band_areas
                        except Exception:
                            pass
                        # Configured genes of interest (TOML [cnv].genes) + gene selector
                        try:
                            gene_opts = {"All": "All"}
                            chrom_genes = _configured_genes_on_chrom(
                                configured_gene_locations, selected
                            )
                            for row in chrom_genes:
                                gene_opts[str(row["gene"])] = str(row["gene"])

                            # Fall back to panel gene BED when no TOML genes are configured.
                            if len(gene_opts) == 1:
                                gene_df = _load_gene_bed(sample_dir)
                                gchr = gene_df[gene_df["chrom"] == selected]
                                for _, gr in gchr.iterrows():
                                    gene_opts[str(gr["gene"])] = str(gr["gene"])
                                if series_abs:
                                    main = series_abs[0]
                                    mark = []
                                    for _, gr in gchr.iterrows():
                                        mark.append(
                                            [
                                                {
                                                    "name": str(gr["gene"]),
                                                    "xAxis": float(gr["start_pos"]),
                                                    "label": {
                                                        "position": "insideTop",
                                                        "color": (
                                                            "#e2e8f0"
                                                            if dark_ui
                                                            else "#000"
                                                        ),
                                                        "fontSize": 11,
                                                    },
                                                },
                                                {"xAxis": float(gr["end_pos"])},
                                            ]
                                        )
                                    main.setdefault("markArea", {"data": []})
                                    main["markArea"]["data"] = (
                                        main["markArea"]["data"] or []
                                    ) + mark
                                    series_abs[0] = main

                            try:
                                cnv_gene_select.set_options(gene_opts)
                                current_gene = launcher._cnv_state.setdefault(
                                    str(sample_dir), {}
                                ).get("selected_gene", "All")
                                if current_gene not in gene_opts:
                                    launcher._cnv_state[str(sample_dir)][
                                        "selected_gene"
                                    ] = "All"
                                    cnv_gene_select.value = "All"
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Keep gene overlays on top after cytoband / breakpoint overlays.
                        _apply_abs_gene_coverage_overlay()
                        _upsert_configured_gene_series(
                            cnv_diff,
                            configured_gene_locations,
                            selected=selected,
                            chrom_offsets=chrom_offsets,
                            dark=dark_ui,
                        )
                        
                        # Breakpoint candidates as dashed vertical lines
                        try:
                            idx_cyto_abs = next(
                                (
                                    i
                                    for i, s in enumerate(cnv_abs.options["series"])
                                    if s.get("name") == "cytobands_highlight"
                                ),
                                None,
                            )
                            if (
                                idx_cyto_abs is not None
                                and state.get("bp_array") is not None
                            ):
                                arr = state["bp_array"]
                                pos = [
                                    int(r["end_pos"]) for r in arr if r["name"] == selected
                                ]
                                lines = [
                                    {
                                        "xAxis": float(p),
                                        "lineStyle": {
                                            "type": "dashed",
                                            "color": "#E0162B",
                                        },
                                    }
                                    for p in pos
                                ]
                                cnv_abs.options["series"][idx_cyto_abs].setdefault(
                                    "markLine", {"symbol": "none", "data": []}
                                )
                                cnv_abs.options["series"][idx_cyto_abs]["markLine"][
                                    "data"
                                ] = lines
                        except Exception:
                            pass
                        # Re-assert abs coverage lollipops after breakpoint markLines mutate series.
                        _apply_abs_gene_coverage_overlay()
                # debug label removed
            except Exception:
                pass
            # Adaptive thinning based on current zoom and cap total points
            _thin_chart_series(cnv_abs, MAX_POINTS_PER_CHART)

            # Apply gene zoom before updating chart
            try:
                sel_gene = launcher._cnv_state.setdefault(
                    str(sample_dir), {}
                ).get("selected_gene", "All")

                gene_interval = None
                if sel_gene and sel_gene != "All" and selected != "All":
                    gene_interval = _find_configured_gene_interval(
                        configured_gene_locations,
                        selected=selected,
                        gene_name=sel_gene,
                    )
                    if gene_interval is None:
                        gene_df = _load_gene_bed(sample_dir)
                        gchr = gene_df[gene_df["chrom"] == selected]
                        row = gchr[gchr["gene"] == sel_gene]
                        if not row.empty:
                            gene_interval = {
                                "start_pos": int(row.iloc[0]["start_pos"]),
                                "end_pos": int(row.iloc[0]["end_pos"]),
                            }

                if gene_interval is not None:
                    s_bp = int(gene_interval["start_pos"])
                    e_bp = int(gene_interval["end_pos"])
                    pad = 10 * binw_analysis
                    zoom_start = max(0, s_bp - pad)
                    zoom_end = e_bp + pad
                    try:
                        cnv_abs.options["dataZoom"][0].update(
                            {
                                "startValue": zoom_start,
                                "endValue": zoom_end,
                                "start": None,
                                "end": None,
                            }
                        )
                    except Exception:
                        pass
                else:
                    # Reset zoom when "All" is selected
                    try:
                        if isinstance(cnv_abs.options.get("dataZoom"), list) and cnv_abs.options["dataZoom"]:
                            dz = cnv_abs.options["dataZoom"][0]
                            dz.pop("startValue", None)
                            dz.pop("endValue", None)
                            dz.update({"start": 0, "end": 100})
                    except Exception:
                        pass
            except Exception:
                pass

            # Ensure abs coverage lollipops remain on top after thinning / overlay mutations.
            _apply_abs_gene_coverage_overlay()

            _apply_cnv_scatter_performance(cnv_abs)
            _apply_cnv_echart_chrome(cnv_abs, _is_dark_mode())
            _cnv_echart_push_update(cnv_abs)
            # Difference plot (linear CNV3)
            if cnv3_map:
                series_diff = _build_cnv_track_scatter_series(
                    cnv3_map,
                    selected=selected,
                    binw_analysis=int(binw_analysis),
                    plot_bin_width=int(plot_bin_width),
                    chrom_palette=chrom_palette,
                )
                try:
                    base_series = cnv_diff.options["series"]
                    keep = [
                        s
                        for s in base_series
                        if s.get("name")
                        in ("centromeres_highlight", "cytobands_highlight")
                    ]
                except Exception:
                    keep = []
                trend_series_diff: List[Dict[str, Any]] = []
                if show_trend:
                    for series in series_diff:
                        data = series.get("data") or []
                        if not data:
                            continue
                        trend = _cnv_trend_series(
                            str(series.get("name") or "diff"),
                            [float(p[0]) for p in data],
                            [float(p[1]) for p in data],
                            dark=dark_ui,
                        )
                        if trend is not None:
                            trend_series_diff.append(trend)
                cnv_diff.options["series"] = series_diff + trend_series_diff + keep
                # The difference track is centred on zero, so give it the same
                # symmetric, mirrored, finely ticked axis as the log2 CNV panel.
                diff_lo, diff_hi = _cnv_diff_y_window(
                    _cnv_scatter_value_range(series_diff)
                )
                diff_ticks = cnv_axis_tick_spec(diff_lo, diff_hi, use_log=True)
                diff_lo, diff_hi = snap_axis_window_to_ticks(
                    diff_lo, diff_hi, diff_ticks.major
                )
                diff_ticks = cnv_axis_tick_spec(diff_lo, diff_hi, use_log=True)
                _apply_cnv_abs_y_window(
                    cnv_diff,
                    diff_lo,
                    diff_hi,
                    ticks=diff_ticks,
                    use_log=True,
                )
                _upsert_cnv_centromere_lines(
                    cnv_diff,
                    dark=dark_ui,
                    selected=selected,
                    chrom_offsets=chrom_offsets,
                )
                _upsert_cnv_reference_lines(
                    cnv_diff,
                    use_log=True,
                    dark=dark_ui,
                    chromosome=selected,
                    sex_estimate=sex_estimate_label,
                    y_lo=diff_lo,
                    y_hi=diff_hi,
                    cutoff_override=cutoff_override,
                )
                _upsert_configured_gene_series(
                    cnv_diff,
                    configured_gene_locations,
                    selected=selected,
                    chrom_offsets=chrom_offsets,
                    dark=dark_ui,
                )
                _thin_chart_series(cnv_diff, MAX_POINTS_PER_CHART)
                _upsert_configured_gene_series(
                    cnv_diff,
                    configured_gene_locations,
                    selected=selected,
                    chrom_offsets=chrom_offsets,
                    dark=dark_ui,
                )
                _apply_cnv_scatter_performance(cnv_diff)
                _apply_cnv_echart_chrome(cnv_diff, _is_dark_mode())
                _cnv_echart_push_update(cnv_diff)
            else:
                _apply_cnv_scatter_performance(cnv_diff)
                _apply_cnv_echart_chrome(cnv_diff, _is_dark_mode())
                _cnv_echart_push_update(cnv_diff)

            # Gene zoom on difference chart (single-chromosome view)
            try:
                sel_gene = launcher._cnv_state.setdefault(
                    str(sample_dir), {}
                ).get("selected_gene", "All")
                for rel_chart in (cnv_diff,):
                    gene_interval = None
                    if sel_gene and sel_gene != "All" and selected != "All":
                        gene_interval = _find_configured_gene_interval(
                            configured_gene_locations,
                            selected=selected,
                            gene_name=sel_gene,
                        )
                        if gene_interval is None:
                            gene_df = _load_gene_bed(sample_dir)
                            gchr = gene_df[gene_df["chrom"] == selected]
                            row = gchr[gchr["gene"] == sel_gene]
                            if not row.empty:
                                gene_interval = {
                                    "start_pos": int(row.iloc[0]["start_pos"]),
                                    "end_pos": int(row.iloc[0]["end_pos"]),
                                }
                    if gene_interval is not None:
                        s_bp = int(gene_interval["start_pos"])
                        e_bp = int(gene_interval["end_pos"])
                        pad = 10 * binw_analysis
                        zoom_start = max(0, s_bp - pad)
                        zoom_end = e_bp + pad
                        try:
                            rel_chart.options["dataZoom"][0].update(
                                {
                                    "startValue": zoom_start,
                                    "endValue": zoom_end,
                                    "start": None,
                                    "end": None,
                                }
                            )
                            _cnv_echart_push_update(rel_chart)
                        except Exception:
                            pass
                    else:
                        try:
                            if (
                                isinstance(rel_chart.options.get("dataZoom"), list)
                                and rel_chart.options["dataZoom"]
                            ):
                                dz = rel_chart.options["dataZoom"][0]
                                dz.pop("startValue", None)
                                dz.pop("endValue", None)
                                dz.update({"start": 0, "end": 100})
                        except Exception:
                            pass
            except Exception:
                pass

            # Regional CNV events table (same logic as PDF reports)
            try:
                selected = state.get("selected_chrom", "All")
                binw = state.get("cnv_dict", {}).get("bin_width", 1_000_000)
                sex_lbl = _sex_label(state.get("xy"))
                # log2(ploidy / expected) — the same track every other CNV
                # output uses. The sample-minus-reference track this replaces
                # has the signal largely subtracted out, because the reference
                # pass re-processes the same BAM on top of the reference
                # baseline and so inherits the sample's own aberration.
                if isinstance(cnv_log2_map, dict) and cnv_log2_map:
                    data = cnv_log2_map
                    source = "cnv_log2"
                else:
                    data = cnv_map if isinstance(cnv_map, dict) else None
                    source = "cnv"
                if data and binw:
                    cyto_cutoff = _resolve_cnv_cutoff(state.get("cutoff"))
                    cache_key = (
                        f"{source}:{state.get(source+'_m')}:{int(binw)}:{sex_lbl}"
                        f":{cyto_cutoff}"
                    )
                    if state.get("cyto_cache_key") != cache_key:
                        panel_name, panel_genes_df = load_panel_gene_bed(str(sample_dir))
                        df_all = _compute_all_cytoband_df(
                            data, int(binw), sex_lbl, cyto_cutoff
                        )
                        state["cyto_df_all"] = df_all
                        state["panel_name"] = panel_name
                        state["panel_genes_df"] = panel_genes_df
                        state["cyto_cache_key"] = cache_key
                    df_all = state.get("cyto_df_all")
                    panel_genes_df = state.get("panel_genes_df")
                    if not isinstance(panel_genes_df, pd.DataFrame):
                        _, panel_genes_df = load_panel_gene_bed(str(sample_dir))
                    panel_name = state.get("panel_name")
                    regional_title = "Regional CNV events"
                    if panel_name:
                        regional_title += f" ({panel_name} panel genes)"
                    regional_cnv_label.set_text(
                        regional_title + _cutoff_suffix_text(cyto_cutoff)
                    )

                    if isinstance(df_all, pd.DataFrame) and not df_all.empty:
                        if selected and selected != "All":
                            df_show = df_all[df_all["chrom"] == selected]
                        else:
                            df_show = df_all
                        regional_rows = _build_regional_rows(df_show, panel_genes_df)
                        regional_cnv_table.rows = regional_rows
                        try:
                            regional_cnv_table.update()
                        except Exception:
                            pass
                        panel_gene_count = sum(
                            1 for row in regional_rows if row.get("panel_genes") != "—"
                        )
                        if selected and selected != "All":
                            regional_cnv_summary.set_text(
                                _get_cytoband_cnv_summary(
                                    data, selected, int(binw), sex_lbl, cyto_cutoff
                                )
                            )
                        elif regional_rows:
                            summary = (
                                f"Detected {len(regional_rows)} regional CNV events"
                            )
                            if panel_gene_count:
                                summary += f"; {panel_gene_count} with panel genes"
                            regional_cnv_summary.set_text(summary)
                        else:
                            regional_cnv_summary.set_text(
                                "No regional CNV events detected"
                            )
                    else:
                        regional_cnv_table.rows = []
                        try:
                            regional_cnv_table.update()
                        except Exception:
                            pass
                        regional_cnv_summary.set_text(
                            "No regional CNV events detected"
                        )
                else:
                    regional_cnv_summary.set_text("CNV data not available")
            except Exception:
                pass
        except Exception:
            pass

    def _prepare_cnv_refresh() -> Optional[Dict[str, Any]]:
        """Main-thread debounce, UI sync, mtime checks. Returns None if no work needed."""
        key = str(sample_dir)
        state = launcher._cnv_state.get(key, {})

        current_time = time.time()
        last_refresh = state.get("_last_refresh", 0)
        force_color_refresh = state.get("_force_color_refresh", False)
        force_gene_refresh = state.get("_force_gene_refresh", False)
        force_chrom_refresh = state.get("_force_chrom_refresh", False)
        force_gene_cov_filter_refresh = state.get(
            "_force_gene_cov_filter_refresh", False
        )
        force_ui_refresh = state.get("_force_ui_refresh", False)
        force_refresh = (
            force_color_refresh
            or force_gene_refresh
            or force_chrom_refresh
            or force_gene_cov_filter_refresh
            or force_ui_refresh
        )
        # Never debounce away an explicit UI control click.
        if current_time - last_refresh < 0.1 and not force_refresh:
            return None
        state["_last_refresh"] = current_time

        ui_changed = False
        try:
            ui_sel = getattr(cnv_chrom_select, "value", None)
            if ui_sel and ui_sel != state.get("selected_chrom"):
                state["selected_chrom"] = ui_sel
                ui_changed = True
            ui_scale = getattr(cnv_scale, "value", None)
            if ui_scale is not None:
                # Switch: True = log2 ratio, False = linear ploidy.
                if isinstance(ui_scale, bool):
                    want_scale = "log" if ui_scale else "linear"
                else:
                    scale_key = str(ui_scale).strip().lower()
                    if scale_key in (
                        "log",
                        "log2",
                        "log2 ratio",
                        "log2 ratio (ploidy / expected)",
                        "true",
                        "1",
                    ):
                        want_scale = "log"
                    elif scale_key in ("linear", "ploidy", "false", "0"):
                        want_scale = "linear"
                    else:
                        want_scale = (
                            "log" if "log" in scale_key else state.get("y_scale", "linear")
                        )
                if want_scale != state.get("y_scale"):
                    state["y_scale"] = want_scale
                    ui_changed = True
            ui_bp = getattr(cnv_bp, "value", None)
            if ui_bp is not None:
                # Switch: True = show breakpoints, False = hide.
                if isinstance(ui_bp, bool):
                    desired = ui_bp
                else:
                    desired = str(ui_bp).strip().lower() in (
                        "show",
                        "true",
                        "1",
                        "on",
                    )
                if desired != state.get("show_bp", True):
                    state["show_bp"] = desired
                    ui_changed = True
            ui_color = getattr(cnv_color, "value", None)
            current_color_mode = state.get("color_mode", "chromosome")
            if ui_color is not None:
                # Switch: True = up/down, False = chromosome.
                if isinstance(ui_color, bool):
                    want_color = "value" if ui_color else "chromosome"
                else:
                    vlow = str(ui_color).strip().lower()
                    if vlow in ("value", "up/down", "updown", "true", "1"):
                        want_color = "value"
                    else:
                        want_color = "chromosome"
                if want_color != current_color_mode:
                    state["color_mode"] = want_color
                    ui_changed = True
            ui_plot_bin = getattr(cnv_plot_bin, "value", None)
            if ui_plot_bin is not None:
                want_bin = _cnv_plot_bin_bp_from_ui(ui_plot_bin)
                if want_bin != state.get("plot_bin_width"):
                    state["plot_bin_width"] = want_bin
                    ui_changed = True
            ui_gene_cov = getattr(cnv_gene_cov_filter, "value", None)
            if ui_gene_cov is not None:
                # Coerce legacy string values that may still be on the widget.
                if isinstance(ui_gene_cov, bool):
                    want_filter = (
                        _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                        if ui_gene_cov
                        else _CNV_GENE_COVERAGE_FILTER_ALL
                    )
                else:
                    want_filter = _cnv_gene_coverage_filter_from_ui(ui_gene_cov)
                if want_filter != state.get(
                    "gene_coverage_filter", _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                ):
                    state["gene_coverage_filter"] = want_filter
                    ui_changed = True
                # Keep the widget strictly boolean after any legacy string value.
                want_outliers = want_filter == _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                if ui_gene_cov is not want_outliers:
                    try:
                        cnv_gene_cov_filter.value = want_outliers
                    except Exception:
                        pass
        except Exception:
            pass

        is_fresh_visit = "last_visit_time" not in state
        if is_fresh_visit:
            state["last_visit_time"] = time.time()

        cnv_npy = sample_dir / "CNV.npy"
        cnv3_npy = sample_dir / "CNV3.npy"
        cnv_dict_npy = sample_dir / "CNV_dict.npy"
        data_array_npy = sample_dir / "cnv_data_array.npy"
        xy_pkl = sample_dir / "XYestimate.pkl"

        cnv_npy_mtime = cnv_npy.stat().st_mtime if cnv_npy.exists() else 0
        cnv3_npy_mtime = cnv3_npy.stat().st_mtime if cnv3_npy.exists() else 0
        cnv_dict_npy_mtime = cnv_dict_npy.stat().st_mtime if cnv_dict_npy.exists() else 0
        data_array_npy_mtime = data_array_npy.stat().st_mtime if data_array_npy.exists() else 0
        xy_pkl_mtime = xy_pkl.stat().st_mtime if xy_pkl.exists() else 0

        prev_cnv_npy_mtime = state.get("cnv_m", 0)
        prev_cnv3_npy_mtime = state.get("cnv3_m", 0)
        prev_cnv_dict_npy_mtime = state.get("dict_m", 0)
        prev_data_array_npy_mtime = state.get("bp_array_mtime", 0)
        prev_xy_pkl_mtime = state.get("xy_m", 0)

        cnv_npy_changed = prev_cnv_npy_mtime != cnv_npy_mtime
        cnv3_npy_changed = prev_cnv3_npy_mtime != cnv3_npy_mtime
        cnv_dict_npy_changed = prev_cnv_dict_npy_mtime != cnv_dict_npy_mtime
        data_array_npy_changed = prev_data_array_npy_mtime != data_array_npy_mtime
        xy_pkl_changed = prev_xy_pkl_mtime != xy_pkl_mtime

        files_changed = (
            cnv_npy_changed
            or cnv3_npy_changed
            or cnv_dict_npy_changed
            or data_array_npy_changed
            or xy_pkl_changed
        )

        needs_update = (
            is_fresh_visit
            or files_changed
            or ui_changed
            or force_refresh
            or not state.get("_rendered_once")
        )

        if not needs_update:
            logging.debug("[CNV] ⏭ Skipping CNV update - no changes detected")
            launcher._cnv_state[key] = state
            return None

        reasons = []
        if is_fresh_visit:
            reasons.append("fresh_visit")
        if cnv_npy_changed:
            reasons.append("CNV.npy")
        if cnv3_npy_changed:
            reasons.append("CNV3.npy")
        if cnv_dict_npy_changed:
            reasons.append("CNV_dict.npy")
        if data_array_npy_changed:
            reasons.append("cnv_data_array.npy")
        if xy_pkl_changed:
            reasons.append("XYestimate.pkl")
        if ui_changed:
            reasons.append("ui_changed")
        if force_color_refresh:
            reasons.append("force_color_refresh")
        if force_gene_refresh:
            reasons.append("force_gene_refresh")
        if force_chrom_refresh:
            reasons.append("force_chrom_refresh")
        if force_gene_cov_filter_refresh:
            reasons.append("force_gene_cov_filter_refresh")
        if force_ui_refresh:
            reasons.append("force_ui_refresh")
        if not state.get("_rendered_once"):
            reasons.append("first_render")
        logging.debug(f"[CNV] Update needed. Reasons: {', '.join(reasons)}")

        data_array_reload = data_array_npy_changed or is_fresh_visit
        need_load = (
            cnv_dict_npy_changed
            or cnv_npy_changed
            or cnv3_npy_changed
            or data_array_reload
            or xy_pkl_changed
        )

        return {
            "state": state,
            "key": key,
            "ui_changed": ui_changed,
            "is_fresh_visit": is_fresh_visit,
            "cnv_npy": cnv_npy,
            "cnv3_npy": cnv3_npy,
            "cnv_dict_npy": cnv_dict_npy,
            "data_array_npy": data_array_npy,
            "xy_pkl": xy_pkl,
            "cnv_npy_mtime": cnv_npy_mtime,
            "cnv3_npy_mtime": cnv3_npy_mtime,
            "cnv_dict_npy_mtime": cnv_dict_npy_mtime,
            "data_array_npy_mtime": data_array_npy_mtime,
            "xy_pkl_mtime": xy_pkl_mtime,
            "cnv_dict_npy_changed": cnv_dict_npy_changed,
            "cnv_npy_changed": cnv_npy_changed,
            "cnv3_npy_changed": cnv3_npy_changed,
            "data_array_npy_changed": data_array_npy_changed,
            "xy_pkl_changed": xy_pkl_changed,
            "data_array_reload": data_array_reload,
            "need_load": need_load,
        }

    def _apply_breakpoint_marklines(
        chart,
        selected: str,
        state: Dict[str, Any],
        breakpoint_lines: List[int],
        *,
        reference_at_zero: bool = False,
    ) -> None:
        """Attach breakpoint x-lines (and optional y=0 reference) to the main data series."""
        try:
            current_series = [
                s
                for s in chart.options["series"]
                if not s.get("name", "").startswith("Breakpoint")
            ]
            if selected != "All" and state.get("show_bp", True) and breakpoint_lines:
                if current_series:
                    mark_line_data: List[Dict[str, Any]] = []
                    if reference_at_zero:
                        mark_line_data.append(
                            {
                                "yAxis": 0,
                                "lineStyle": {
                                    "type": "dashed",
                                    "color": "#888888",
                                    "width": 1,
                                },
                            }
                        )
                    for bp_pos in breakpoint_lines:
                        mark_line_data.append(
                            {
                                "xAxis": bp_pos,
                                "lineStyle": {
                                    "type": "dashed",
                                    "color": "#ff6b6b",
                                    "width": 3,
                                },
                            }
                        )
                    current_series[0]["markLine"] = {
                        "data": mark_line_data,
                        "symbol": "none",
                        "lineStyle": {"type": "dashed", "color": "#ff6b6b", "width": 3},
                    }
            elif reference_at_zero and current_series:
                current_series[0]["markLine"] = {
                    "symbol": "none",
                    "data": [
                        {
                            "yAxis": 0,
                            "lineStyle": {
                                "type": "dashed",
                                "color": "#888888",
                                "width": 1,
                            },
                        }
                    ],
                }
            else:
                if current_series:
                    current_series[0].pop("markLine", None)
            chart.options["series"] = current_series
            _upsert_configured_gene_series(
                chart,
                configured_gene_locations,
                selected=selected,
                chrom_offsets={},
                dark=_is_dark_mode(),
            )
            _apply_cnv_echart_chrome(chart, _is_dark_mode())
            _cnv_echart_push_update(chart)
        except Exception:
            pass

    def _apply_cnv_refresh_after_load(
        plan: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        """Merge disk payload into state and update charts (main thread only)."""
        p = plan
        state = p["state"]
        key = p["key"]
        ui_changed = p["ui_changed"]
        is_fresh_visit = p["is_fresh_visit"]
        cnv_dict_npy = p["cnv_dict_npy"]
        cnv_npy = p["cnv_npy"]
        cnv3_npy = p["cnv3_npy"]
        data_array_npy = p["data_array_npy"]
        xy_pkl = p["xy_pkl"]
        cnv_npy_mtime = p["cnv_npy_mtime"]
        cnv3_npy_mtime = p["cnv3_npy_mtime"]
        cnv_dict_npy_mtime = p["cnv_dict_npy_mtime"]
        data_array_npy_mtime = p["data_array_npy_mtime"]
        xy_pkl_mtime = p["xy_pkl_mtime"]
        cnv_dict_npy_changed = p["cnv_dict_npy_changed"]
        cnv_npy_changed = p["cnv_npy_changed"]
        cnv3_npy_changed = p["cnv3_npy_changed"]
        data_array_npy_changed = p["data_array_npy_changed"]
        data_array_reload = p["data_array_reload"]
        xy_pkl_changed = p["xy_pkl_changed"]

        changed = ("cnv" in payload or "cnv3" in payload or "xy" in payload)

        if cnv_dict_npy.exists():
            m = cnv_dict_npy_mtime
            if cnv_dict_npy_changed and "cnv_dict" in payload:
                state["cnv_dict"] = payload["cnv_dict"]
                cnv_bin.set_text(
                    f"Bin width: {state['cnv_dict'].get('bin_width', '--'):,}"
                )
                cnv_var.set_text(
                    f"Variance: {state['cnv_dict'].get('variance','--'):.3f}"
                    if isinstance(state["cnv_dict"].get("variance"), (int, float))
                    else "Variance: --"
                )
                state["dict_m"] = m
        if xy_pkl.exists():
            m = xy_pkl_mtime
            if xy_pkl_changed:
                try:
                    if "xy" in payload:
                        xy = payload["xy"]
                        cnv_xy.set_text(f"Genetic sex: {xy}")
                        state["xy"] = xy
                except Exception:
                    pass
                state["xy_m"] = m

        if "cnv" in payload:
            state["cnv"] = payload["cnv"]
            state["cnv_m"] = cnv_npy_mtime
        if "cnv3" in payload:
            state["cnv3"] = payload["cnv3"]
            state["cnv3_m"] = cnv3_npy_mtime
        _recompute_cnv_log2_state(state)
        _refresh_cnv_load_labels(state)
        _refresh_ngtd_table(state)

        if state.get("cnv"):
            if changed:
                cnv_status.set_text("Status: CNV data loaded")
            if changed or not state.get("_chrom_opts_set"):
                try:
                    cnv_map = state["cnv"].get("cnv", state["cnv"])
                    chrom_opts = {"All": "All"}
                    for contig in natsort.natsorted(cnv_map.keys()):
                        if _cnv_contig_ok(contig):
                            chrom_opts[contig] = contig
                    cnv_chrom_select.set_options(chrom_opts)
                    chrom_opts = {"All": "All"}
                    for contig in natsort.natsorted(cnv_map.keys()):
                        if _cnv_contig_ok(contig):
                            chrom_opts[contig] = contig
                    cnv_chrom_select.set_options(chrom_opts)
                    sel = launcher._cnv_state.setdefault(str(sample_dir), {}).get(
                        "selected_chrom", "All"
                    )
                    if sel not in chrom_opts:
                        sel = "All"
                    cnv_chrom_select.value = sel
                    try:
                        cnv_chrom_select.update()
                    except Exception:
                        pass
                    state["_chrom_opts_set"] = True
                except Exception:
                    pass
            force_color_refresh = state.get("_force_color_refresh", False)
            force_gene_refresh = state.get("_force_gene_refresh", False)
            force_chrom_refresh = state.get("_force_chrom_refresh", False)
            force_gene_cov_filter_refresh = state.get(
                "_force_gene_cov_filter_refresh", False
            )
            force_ui_refresh = state.get("_force_ui_refresh", False)
            if (
                changed
                or ui_changed
                or not state.get("_rendered_once")
                or force_color_refresh
                or force_gene_refresh
                or force_chrom_refresh
                or force_gene_cov_filter_refresh
                or force_ui_refresh
            ):
                _render_cnv_from_state(state)
                state["_rendered_once"] = True
                if force_color_refresh:
                    state["_force_color_refresh"] = False
                if force_gene_refresh:
                    state["_force_gene_refresh"] = False
                if force_chrom_refresh:
                    state["_force_chrom_refresh"] = False
                if force_gene_cov_filter_refresh:
                    state["_force_gene_cov_filter_refresh"] = False
                if force_ui_refresh:
                    state["_force_ui_refresh"] = False

            _update_cnv_events_analysis(state)

        if data_array_npy.exists() and (data_array_npy_changed or is_fresh_visit):
            try:
                arr = payload.get("bp_array")
                if arr is None:
                    arr = np.load(data_array_npy, allow_pickle=True)
                if hasattr(arr, "dtype") and "name" in arr.dtype.names:
                    state["bp_array"] = arr
                    selected = launcher._cnv_state.setdefault(
                        str(sample_dir), {}
                    ).get("selected_chrom", "All")
                    breakpoint_lines = []
                    for r in arr:
                        if selected == "All" or r["name"] == selected:
                            start_pos = int(r["start"])
                            end_pos = int(r["end"])
                            breakpoint_lines.append((start_pos + end_pos) // 2)
                    _apply_breakpoint_marklines(
                        cnv_diff, selected, state, breakpoint_lines
                    )
                state["bp_array_mtime"] = data_array_npy_mtime
            except Exception:
                pass
        elif data_array_npy.exists() and not data_array_npy_changed:
            if state.get("bp_array") is not None:
                try:
                    selected = launcher._cnv_state.setdefault(
                        str(sample_dir), {}
                    ).get("selected_chrom", "All")
                    arr = state["bp_array"]
                    breakpoint_lines = []
                    for r in arr:
                        if selected == "All" or r["name"] == selected:
                            start_pos = int(r["start"])
                            end_pos = int(r["end"])
                            breakpoint_lines.append((start_pos + end_pos) // 2)
                    _apply_breakpoint_marklines(
                        cnv_diff, selected, state, breakpoint_lines
                    )
                except Exception:
                    pass

        state["cnv_m"] = cnv_npy_mtime
        state["cnv3_m"] = cnv3_npy_mtime
        state["dict_m"] = cnv_dict_npy_mtime
        state["xy_m"] = xy_pkl_mtime
        state["last_visit_time"] = state.get("last_visit_time", time.time())
        state["cnv_plot_theme_dark"] = _is_dark_mode()

        launcher._cnv_state[key] = state
        _update_breakpoints_visibility()

    async def _refresh_cnv_async() -> None:
        try:
            plan = _prepare_cnv_refresh()
            if plan is None:
                return
            if plan["need_load"]:
                payload = await asyncio.to_thread(
                    _cnv_load_binary_payload,
                    sample_dir,
                    cnv_dict_npy_changed=plan["cnv_dict_npy_changed"],
                    cnv_npy_changed=plan["cnv_npy_changed"],
                    cnv3_npy_changed=plan["cnv3_npy_changed"],
                    data_array_reload=plan["data_array_reload"],
                    xy_pkl_changed=plan["xy_pkl_changed"],
                )
            else:
                payload = {}
            _apply_cnv_refresh_after_load(plan, payload)
        except Exception:
            pass

    def _refresh_cnv_sync(sample_dir: Path, launcher: Any) -> None:
        """Synchronous CNV refresh (no event loop)."""
        try:
            plan = _prepare_cnv_refresh()
            if plan is None:
                return
            if plan["need_load"]:
                payload = _cnv_load_binary_payload(
                    sample_dir,
                    cnv_dict_npy_changed=plan["cnv_dict_npy_changed"],
                    cnv_npy_changed=plan["cnv_npy_changed"],
                    cnv3_npy_changed=plan["cnv3_npy_changed"],
                    data_array_reload=plan["data_array_reload"],
                    xy_pkl_changed=plan["xy_pkl_changed"],
                )
            else:
                payload = {}
            _apply_cnv_refresh_after_load(plan, payload)
        except Exception:
            pass

    def _refresh_cnv() -> None:
        """Refresh CNV data; offload binary loads when a loop is running."""
        try:
            if not sample_dir or not sample_dir.exists():
                logging.warning(f"[CNV] Sample directory not found: {sample_dir}")
                return
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                _refresh_cnv_sync(sample_dir, launcher)
                return
            asyncio.create_task(_refresh_cnv_async())
        except Exception as e:
            logging.exception(f"[CNV] Refresh failed: {e}")

    def _update_breakpoints_visibility() -> None:
        """Show/hide breakpoints controls based on chromosome selection."""
        try:
            key = str(sample_dir)
            state = launcher._cnv_state.get(key, {})
            selected = state.get("selected_chrom", "All")
            
            # Show breakpoints controls only when viewing individual chromosomes
            should_show = selected != "All"
            
            try:
                display_value = "block" if should_show else "none"
                cnv_bp_label.style(f"display: {display_value}")
                cnv_bp_row.style(f"display: {display_value}")
            except Exception:
                pass
        except Exception:
            pass

    # Bind control events
    try:

        def _val(ev, default=None):
            # Prefer boolean switch values before legacy toggle label parsing.
            if hasattr(ev, "value") and isinstance(ev.value, bool):
                return ev.value
            args = getattr(ev, "args", None)
            if isinstance(args, bool):
                return args
            if isinstance(args, (list, tuple)) and args and isinstance(args[0], bool):
                return args[0]
            # Handle select/toggle events like [index, {'value': X, 'label': 'Y'}]
            if isinstance(args, list) and len(args) >= 2 and isinstance(args[1], dict):
                return args[1].get("label", default)
            # Handle direct value objects like {'value': 2, 'label': 'GNB1'}
            if hasattr(ev, "value") and isinstance(ev.value, dict):
                return ev.value.get("label", default)
            # Handle args that are directly a dictionary with label
            if isinstance(args, dict) and "label" in args:
                return args.get("label", default)
            # Fallback to standard value extraction
            if hasattr(ev, "value"):
                return ev.value
            return args if args is not None else default

        def _switch_bool(ev, *, default: bool = False) -> bool:
            """Read a ui.switch boolean, falling back to the widget value if needed."""
            raw = _val(ev, None)
            if isinstance(raw, bool):
                return raw
            if raw is None:
                return default
            return str(raw).strip().lower() in ("true", "1", "on", "show")

        def _force_redraw_with_marker_autoscale() -> None:
            """Every CNV control click should re-render and re-fit Y to markers."""
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["_force_ui_refresh"] = True
            ui.timer(0.05, _refresh_cnv, once=True)

        def _on_chrom(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["selected_chrom"] = _val(ev, "All") or "All"
            st["_force_chrom_refresh"] = True  # Force refresh for chromosome selection
            logging.debug(f"CNV select changed -> {st['selected_chrom']}")
            # Update breakpoints visibility
            _update_breakpoints_visibility()
            # reset x zoom when switching scope
            try:
                for chart in genome_charts:
                    if (
                        isinstance(chart.options.get("dataZoom"), list)
                        and chart.options["dataZoom"]
                    ):
                        chart.options["dataZoom"][0].pop("startValue", None)
                        chart.options["dataZoom"][0].pop("endValue", None)
            except Exception:
                pass
            _force_redraw_with_marker_autoscale()

        def _on_scale(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["y_scale"] = "log" if _switch_bool(ev, default=False) else "linear"
            # Prefer live widget value if event parsing failed.
            try:
                if isinstance(getattr(cnv_scale, "value", None), bool):
                    st["y_scale"] = "log" if cnv_scale.value else "linear"
            except Exception:
                pass
            _force_redraw_with_marker_autoscale()

        def _on_bp(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["show_bp"] = _switch_bool(ev, default=True)
            try:
                if isinstance(getattr(cnv_bp, "value", None), bool):
                    st["show_bp"] = bool(cnv_bp.value)
            except Exception:
                pass
            _force_redraw_with_marker_autoscale()

        def _on_cutoff(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            value = _val(ev, "calling")
            if isinstance(value, dict):
                value = value.get("value", value.get("label"))
            st["cutoff"] = str(value or "calling")
            try:
                if getattr(cnv_cutoff, "value", None):
                    st["cutoff"] = str(cnv_cutoff.value)
            except Exception:
                pass
            st["_force_chrom_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_label_orient(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["label_orientation"] = (
                "rotated" if _switch_bool(ev, default=True) else "horizontal"
            )
            try:
                if isinstance(getattr(cnv_label_orient, "value", None), bool):
                    st["label_orientation"] = (
                        "rotated" if cnv_label_orient.value else "horizontal"
                    )
            except Exception:
                pass
            st["_force_chrom_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_gene_label(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            value = _val(ev, "4.5")
            if isinstance(value, dict):
                value = value.get("value", value.get("label"))
            st["gene_label_size"] = str(value or "4.5")
            try:
                if getattr(cnv_gene_label, "value", None):
                    st["gene_label_size"] = str(cnv_gene_label.value)
            except Exception:
                pass
            st["_force_chrom_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_chrom_axis(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            value = _val(ev, "2")
            if isinstance(value, dict):
                value = value.get("value", value.get("label"))
            st["chrom_axis"] = str(value or "2")
            try:
                if getattr(cnv_chrom_axis, "value", None):
                    st["chrom_axis"] = str(cnv_chrom_axis.value)
            except Exception:
                pass
            st["_force_chrom_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_genome_axis(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            value = _val(ev, "auto")
            if isinstance(value, dict):
                value = value.get("value", value.get("label"))
            st["genome_axis"] = str(value or "auto")
            try:
                if getattr(cnv_genome_axis, "value", None):
                    st["genome_axis"] = str(cnv_genome_axis.value)
            except Exception:
                pass
            _force_redraw_with_marker_autoscale()

        def _on_trend(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["show_trend"] = _switch_bool(ev, default=True)
            try:
                if isinstance(getattr(cnv_trend, "value", None), bool):
                    st["show_trend"] = bool(cnv_trend.value)
            except Exception:
                pass
            st["_force_chrom_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_color(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            st["color_mode"] = "value" if _switch_bool(ev, default=False) else "chromosome"
            try:
                if isinstance(getattr(cnv_color, "value", None), bool):
                    st["color_mode"] = "value" if cnv_color.value else "chromosome"
            except Exception:
                pass
            # Force a refresh by setting a flag that bypasses the state sync logic
            st["_force_color_refresh"] = True
            _force_redraw_with_marker_autoscale()

        def _on_plot_bin(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            v = getattr(ev, "args", None) if hasattr(ev, "args") else getattr(ev, "value", None)
            if v is None and hasattr(ev, "value"):
                v = ev.value
            st["plot_bin_width"] = _cnv_plot_bin_bp_from_ui(v)
            st["_force_chrom_refresh"] = True  # force re-render with new bin width
            _force_redraw_with_marker_autoscale()

        # Bind both native change and model-value updates for robustness
        cnv_chrom_select.on("change", _on_chrom)
        cnv_chrom_select.on("update:model-value", _on_chrom)

        # Gene selection zoom
        def _on_gene(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            selected_gene = _val(ev, "All") or "All"
            st["selected_gene"] = selected_gene
            st["_force_gene_refresh"] = True  # Force refresh for gene selection
            _force_redraw_with_marker_autoscale()

        cnv_gene_select.on("change", _on_gene)
        cnv_gene_select.on("update:model-value", _on_gene)

        def _on_gene_cov_filter(ev):
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            # Prefer the live widget boolean; event args can be ambiguous.
            enabled = True
            try:
                raw = getattr(cnv_gene_cov_filter, "value", None)
                if isinstance(raw, bool):
                    enabled = raw
                else:
                    enabled = _switch_bool(ev, default=True)
            except Exception:
                enabled = _switch_bool(ev, default=True)
            st["gene_coverage_filter"] = (
                _CNV_GENE_COVERAGE_FILTER_OUTLIERS
                if enabled
                else _CNV_GENE_COVERAGE_FILTER_ALL
            )
            try:
                if cnv_gene_cov_filter.value is not enabled:
                    cnv_gene_cov_filter.value = enabled
            except Exception:
                pass
            st["_force_gene_cov_filter_refresh"] = True
            _force_redraw_with_marker_autoscale()

        cnv_gene_cov_filter.on("change", _on_gene_cov_filter)
        cnv_gene_cov_filter.on("update:model-value", _on_gene_cov_filter)
        cnv_scale.on("change", _on_scale)
        cnv_scale.on("update:model-value", _on_scale)
        cnv_plot_bin.on("change", _on_plot_bin)
        cnv_plot_bin.on("update:model-value", _on_plot_bin)
        cnv_bp.on("change", _on_bp)
        cnv_bp.on("update:model-value", _on_bp)
        cnv_color.on("change", _on_color)
        cnv_color.on("update:model-value", _on_color)
        cnv_trend.on("change", _on_trend)
        cnv_trend.on("update:model-value", _on_trend)
        cnv_cutoff.on("change", _on_cutoff)
        cnv_cutoff.on("update:model-value", _on_cutoff)
        cnv_chrom_axis.on("change", _on_chrom_axis)
        cnv_chrom_axis.on("update:model-value", _on_chrom_axis)
        cnv_genome_axis.on("change", _on_genome_axis)
        cnv_genome_axis.on("update:model-value", _on_genome_axis)
        cnv_gene_label.on("change", _on_gene_label)
        cnv_gene_label.on("update:model-value", _on_gene_label)
        cnv_label_orient.on("change", _on_label_orient)
        cnv_label_orient.on("update:model-value", _on_label_orient)

        async def _download_cnv_pdf(kind: str, button: Any) -> None:
            """Render the report CNV figures for this sample and hand back a PDF."""
            if not _cnv_export_allowed(launcher):
                return
            st = launcher._cnv_state.setdefault(str(sample_dir), {})
            use_log2, export_kwargs = cnv_export_settings(launcher, sample_dir)
            scope = None
            try:
                from robin.gui.plotting_preferences import (
                    resolve_plotting_reference_contig_scope,
                )

                scope = resolve_plotting_reference_contig_scope(
                    getattr(launcher, "plotting_preferences", None)
                )
            except Exception:
                logging.debug("Could not resolve contig scope for CNV PDF", exc_info=True)

            button.disable()
            ui.notify("Building CNV PDF…", type="ongoing")
            try:
                from robin.reporting.cnv_export import (
                    CnvExportUnavailable,
                    build_cnv_pdf,
                )

                # Matplotlib rendering is slow enough to stall the event loop.
                payload, filename = await asyncio.to_thread(
                    build_cnv_pdf,
                    sample_dir,
                    kind=kind,
                    use_log2=use_log2,
                    configured_genes=configured_gene_names,
                    reference_contig_scope=scope,
                    plot_bin_width=st.get("plot_bin_width"),
                    **export_kwargs,
                )
            except CnvExportUnavailable as exc:
                ui.notify(str(exc), type="warning")
                return
            except Exception as exc:
                logging.exception("CNV PDF export failed")
                ui.notify(f"CNV PDF export failed: {exc}", type="negative")
                return
            finally:
                button.enable()

            if not payload:
                ui.notify("No CNV plots could be rendered for this sample", type="warning")
                return
            ui.download(payload, filename=filename, media_type="application/pdf")

        cnv_pdf_genome_btn.on_click(
            lambda _e: _download_cnv_pdf("genome", cnv_pdf_genome_btn)
        )
        cnv_pdf_chrom_btn.on_click(
            lambda _e: _download_cnv_pdf("chromosomes", cnv_pdf_chrom_btn)
        )
    except Exception:
        pass

    def _sync_cnv_echarts_theme_if_needed() -> None:
        """Re-apply axis/tooltip/title colours when the user toggles light/dark mode."""
        try:
            dark = _is_dark_mode()
        except Exception:
            dark = False
        key = str(sample_dir)
        st = launcher._cnv_state.get(key, {})
        if st.get("cnv_plot_theme_dark") == dark:
            return
        _apply_cnv_echart_chrome(cnv_abs, dark)
        _apply_cnv_echart_chrome(cnv_diff, dark)
        try:
            _cnv_echart_push_update(cnv_abs)
            _cnv_echart_push_update(cnv_diff)
        except Exception:
            pass
        launcher._cnv_state.setdefault(key, {})["cnv_plot_theme_dark"] = dark

    # Start the refresh timer (every 30 seconds)
    refresh_timer = client_timer(30.0, _refresh_cnv, active=True, immediate=False)
    client_timer(0.5, _refresh_cnv, once=True)
    unregister_cnv_theme_sync = register_theme_sync_callback(
        _sync_cnv_echarts_theme_if_needed,
        element=cnv_abs,
        interval_s=0.5,
        immediate=True,
    )
    try:
        ui.context.client.on_disconnect(
            lambda: (stop_timer(refresh_timer), unregister_cnv_theme_sync())
        )
    except Exception:
        pass
