"""Global GUI plotting preferences (admin-controlled, persisted)."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Optional

from robin.reference_contigs import (
    DEFAULT_REFERENCE_CONTIG_SCOPE,
    REFERENCE_CONTIG_SCOPE_LABELS,
    REFERENCE_CONTIG_SCOPES,
    resolve_reference_contig_scope,
)

PLOTTING_PREFERENCES_KEY = "plotting_preferences"
PLOTTING_PREFERENCES_SCHEMA_VERSION = 9

CNV_REPORT_SCALE_PLOIDY = "ploidy"
CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE = "normalized_difference"
CNV_REPORT_SCALES = (
    CNV_REPORT_SCALE_PLOIDY,
    CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
)
CNV_REPORT_SCALE_LABELS = {
    CNV_REPORT_SCALE_PLOIDY: "Estimated ploidy (absolute copy number)",
    CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE: (
        "Log2 ratio (ploidy / expected copy number)"
    ),
}
DEFAULT_CNV_REPORT_SCALE = CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
_LEGACY_CNV_SCALE_ALIASES = {
    "log2_ratio": CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
}

# Live CNV GUI control defaults (Administration → Plotting).
CNV_GUI_GENE_COVERAGE_FILTER_ALL = "all"
CNV_GUI_GENE_COVERAGE_FILTER_OUTLIERS = "outliers"
CNV_GUI_GENE_COVERAGE_FILTERS = (
    CNV_GUI_GENE_COVERAGE_FILTER_ALL,
    CNV_GUI_GENE_COVERAGE_FILTER_OUTLIERS,
)
DEFAULT_CNV_GUI_GENE_COVERAGE_FILTER = CNV_GUI_GENE_COVERAGE_FILTER_ALL

CNV_GUI_COLOR_MODE_CHROMOSOME = "chromosome"
CNV_GUI_COLOR_MODE_VALUE = "value"
CNV_GUI_COLOR_MODES = (
    CNV_GUI_COLOR_MODE_CHROMOSOME,
    CNV_GUI_COLOR_MODE_VALUE,
)
DEFAULT_CNV_GUI_COLOR_MODE = CNV_GUI_COLOR_MODE_VALUE

DEFAULT_CNV_GUI_SHOW_BREAKPOINTS = True

# Gain/loss cut-off used for the CNV plots and everything judged against them.
# "calling" follows the configured calling thresholds; a number overrides them
# with a symmetric log2 cut-off. The override moves the drawn lines, the gene
# Outliers filter and the point colouring, and also the called regions, the
# events, regional and NGTD tables, the gene states, CNV load and any exported
# report — so a load or event count is only comparable between samples read at
# the same cut-off, which is why every consumer labels the value it used.
CNV_CUTOFF_CALLING = "calling"
CNV_CUTOFF_PRESETS: tuple[float, ...] = (
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
)
DEFAULT_CNV_CUTOFF = "0.35"


# Fixed y-window for the per-chromosome plots, as a symmetric log2 half-span.
# "auto" lets each chromosome size its own axis to the data.
CNV_CHROM_AXIS_AUTO = "auto"
CNV_CHROM_AXIS_PRESETS: tuple[float, ...] = (0.6, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_CNV_CHROM_AXIS = "1.5"

# Separate y-window for the genome-wide summary, kept independent of the
# per-chromosome setting so the two can be scaled differently. It defaults to a
# fixed +/-2 window rather than "auto" so that the genome-wide panel reads on a
# consistent scale from sample to sample; "auto" remains available and fits the
# axis to whatever is in front of you.
DEFAULT_CNV_GENOME_AXIS = "2"


def cnv_genome_axis_options() -> Dict[str, str]:
    """Dropdown options for the genome-wide Y-range selector (value -> label)."""
    return cnv_chrom_axis_options()


def cnv_chrom_axis_options() -> Dict[str, str]:
    """Dropdown options for the per-chromosome Y-range selector (value -> label)."""
    options = {CNV_CHROM_AXIS_AUTO: "Auto (fit data)"}
    for value in CNV_CHROM_AXIS_PRESETS:
        options[f"{value:g}"] = f"\u00b1{value:g}"
    return options


def resolve_cnv_chrom_axis(value: Any) -> Optional[float]:
    """Resolve a stored Y-range setting to a log2 half-span, or None for auto."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in (CNV_CHROM_AXIS_AUTO, "auto", "fit", "data"):
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
    parsed = abs(parsed)
    if not parsed or parsed <= 0:
        return None
    return float(parsed)


def cnv_chrom_axis_setting(value: Any) -> str:
    """Normalise a per-chromosome Y-range setting for storage."""
    resolved = resolve_cnv_chrom_axis(value)
    if resolved is None:
        return CNV_CHROM_AXIS_AUTO
    return f"{resolved:g}"


def cnv_genome_axis_setting(value: Any) -> str:
    """Normalise the genome-wide Y-range setting for storage."""
    return cnv_chrom_axis_setting(value)


# Gene label size on the CNV figures, in matplotlib points as used by the PDF
# report. The live GUI scales its own label from the same setting.
CNV_GENE_LABEL_SIZE_PRESETS: tuple[float, ...] = (3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)
DEFAULT_CNV_GENE_LABEL_SIZE = "4.5"
# Report point size that the GUI's own default label size corresponds to, used to
# scale the ECharts label so one setting reads the same in both renderers.
CNV_GENE_LABEL_GUI_RATIO = 11.0 / 6.0


# Gene label orientation. "horizontal" reads normally, left to right, and is the
# default because a name that has to be read is easier to read that way; it needs
# room along the x-axis, so names that would collide stack into lanes. "rotated"
# reads bottom-to-top alongside the marker (the methylation-array convention) and
# is narrow, so it stays legible where a panel is too crowded for lanes.
CNV_LABEL_ORIENTATION_ROTATED = "rotated"
CNV_LABEL_ORIENTATION_HORIZONTAL = "horizontal"
CNV_LABEL_ORIENTATIONS = (
    CNV_LABEL_ORIENTATION_ROTATED,
    CNV_LABEL_ORIENTATION_HORIZONTAL,
)
DEFAULT_CNV_LABEL_ORIENTATION = CNV_LABEL_ORIENTATION_HORIZONTAL


def resolve_cnv_label_orientation(value: Any) -> str:
    """Normalise a gene label orientation setting."""
    text = str(value or "").strip().lower()
    if text in (CNV_LABEL_ORIENTATION_HORIZONTAL, "flat", "0", "false"):
        return CNV_LABEL_ORIENTATION_HORIZONTAL
    if text in (
        CNV_LABEL_ORIENTATION_ROTATED,
        "portrait",
        "vertical",
        "90",
        "true",
    ):
        return CNV_LABEL_ORIENTATION_ROTATED
    return DEFAULT_CNV_LABEL_ORIENTATION


def cnv_label_is_rotated(value: Any) -> bool:
    """True when gene names should be drawn rotated rather than horizontally."""
    return resolve_cnv_label_orientation(value) == CNV_LABEL_ORIENTATION_ROTATED


def cnv_gene_label_size_options() -> Dict[str, str]:
    """Dropdown options for the CNV gene label size (value -> label)."""
    names = {
        3.5: "Tiny",
        4.0: "Very small",
        4.5: "Small",
        5.0: "Medium",
        6.0: "Large",
        7.0: "Very large",
        8.0: "Largest",
    }
    return {
        f"{value:g}": f"{names.get(value, '')} ({value:g} pt)".strip()
        for value in CNV_GENE_LABEL_SIZE_PRESETS
    }


def resolve_cnv_gene_label_size(value: Any) -> float:
    """Resolve a stored gene label size to a point size, falling back to default."""
    default = float(DEFAULT_CNV_GENE_LABEL_SIZE)
    if value is None:
        return default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not isfinite(parsed) or parsed <= 0:
        return default
    # Keep within the offered range so a stray value cannot make labels unreadable
    # or large enough to swamp the panel.
    return float(
        min(max(parsed, min(CNV_GENE_LABEL_SIZE_PRESETS)), max(CNV_GENE_LABEL_SIZE_PRESETS))
    )


def cnv_gene_label_size_setting(value: Any) -> str:
    """Normalise a gene label size for storage."""
    return f"{resolve_cnv_gene_label_size(value):g}"


def cnv_gene_label_size_px(value: Any) -> float:
    """Equivalent ECharts label size in px for the live GUI."""
    return float(resolve_cnv_gene_label_size(value) * CNV_GENE_LABEL_GUI_RATIO)


def cnv_cutoff_options() -> Dict[str, str]:
    """Dropdown options for the CNV cut-off selector (value -> label)."""
    options = {CNV_CUTOFF_CALLING: "Calling default"}
    for value in CNV_CUTOFF_PRESETS:
        options[f"{value:g}"] = f"\u00b1{value:g}"
    return options


def resolve_cnv_cutoff(value: Any) -> Optional[float]:
    """Resolve a stored cut-off setting to a log2 magnitude, or None for calling.

    Returns the positive magnitude; callers apply it as +value / -value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text in (CNV_CUTOFF_CALLING, "default", "auto"):
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
    parsed = abs(parsed)
    if not parsed or parsed <= 0:
        return None
    return float(parsed)


def cnv_cutoff_setting(value: Any) -> str:
    """Normalise a cut-off setting for storage."""
    resolved = resolve_cnv_cutoff(value)
    if resolved is None:
        return CNV_CUTOFF_CALLING
    return f"{resolved:g}"


# Piecewise-constant segment line over the CNV bin cloud.
DEFAULT_CNV_GUI_SHOW_TREND_LINE = True


def _resolve_gene_coverage_filter(value: Any) -> str:
    vlow = str(value or "").strip().lower()
    if vlow in (
        CNV_GUI_GENE_COVERAGE_FILTER_ALL,
        "all genes",
    ):
        return CNV_GUI_GENE_COVERAGE_FILTER_ALL
    if vlow in (
        CNV_GUI_GENE_COVERAGE_FILTER_OUTLIERS,
        "outliers only",
        "outliers",
        "≠ average",
        "!= average",
    ):
        return CNV_GUI_GENE_COVERAGE_FILTER_OUTLIERS
    return DEFAULT_CNV_GUI_GENE_COVERAGE_FILTER


def _resolve_color_mode(value: Any) -> str:
    vlow = str(value or "").strip().lower()
    if vlow in (
        CNV_GUI_COLOR_MODE_VALUE,
        "up/down",
        "updown",
        "up_down",
        "up down",
    ):
        return CNV_GUI_COLOR_MODE_VALUE
    if vlow in (CNV_GUI_COLOR_MODE_CHROMOSOME, "chromosomes"):
        return CNV_GUI_COLOR_MODE_CHROMOSOME
    return DEFAULT_CNV_GUI_COLOR_MODE


_FALSEY_SETTINGS = ("0", "false", "no", "off", "hide")


def _resolve_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSEY_SETTINGS


@dataclass
class PlottingPreferencesConfig:
    """Admin-controlled defaults for plots in the GUI and PDF reports."""

    schema_version: int = PLOTTING_PREFERENCES_SCHEMA_VERSION
    cnv_report_scale: str = DEFAULT_CNV_REPORT_SCALE
    reference_contig_scope: str = DEFAULT_REFERENCE_CONTIG_SCOPE
    cnv_gui_gene_coverage_filter: str = DEFAULT_CNV_GUI_GENE_COVERAGE_FILTER
    cnv_gui_color_mode: str = DEFAULT_CNV_GUI_COLOR_MODE
    cnv_gui_show_breakpoints: bool = DEFAULT_CNV_GUI_SHOW_BREAKPOINTS
    cnv_gui_show_trend_line: bool = DEFAULT_CNV_GUI_SHOW_TREND_LINE
    cnv_gui_cutoff: str = DEFAULT_CNV_CUTOFF
    cnv_gui_chrom_axis: str = DEFAULT_CNV_CHROM_AXIS
    cnv_gui_genome_axis: str = DEFAULT_CNV_GENOME_AXIS
    cnv_gui_gene_label_size: str = DEFAULT_CNV_GENE_LABEL_SIZE
    cnv_gui_label_orientation: str = DEFAULT_CNV_LABEL_ORIENTATION
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "cnv_report_scale": self.cnv_report_scale,
            "reference_contig_scope": self.reference_contig_scope,
            "cnv_gui_gene_coverage_filter": self.cnv_gui_gene_coverage_filter,
            "cnv_gui_color_mode": self.cnv_gui_color_mode,
            "cnv_gui_show_breakpoints": bool(self.cnv_gui_show_breakpoints),
            "cnv_gui_show_trend_line": bool(self.cnv_gui_show_trend_line),
            "cnv_gui_cutoff": cnv_cutoff_setting(self.cnv_gui_cutoff),
            "cnv_gui_chrom_axis": cnv_chrom_axis_setting(self.cnv_gui_chrom_axis),
            "cnv_gui_genome_axis": cnv_genome_axis_setting(self.cnv_gui_genome_axis),
            "cnv_gui_gene_label_size": cnv_gene_label_size_setting(
                self.cnv_gui_gene_label_size
            ),
            "cnv_gui_label_orientation": resolve_cnv_label_orientation(
                self.cnv_gui_label_orientation
            ),
        }
        if self.updated_at:
            out["updated_at"] = self.updated_at
        if self.updated_by:
            out["updated_by"] = self.updated_by
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlottingPreferencesConfig":
        if not data:
            return cls()
        scale = str(data.get("cnv_report_scale") or DEFAULT_CNV_REPORT_SCALE)
        scale = _LEGACY_CNV_SCALE_ALIASES.get(scale, scale)
        if scale not in CNV_REPORT_SCALES:
            scale = DEFAULT_CNV_REPORT_SCALE
        contig_scope = resolve_reference_contig_scope(
            data.get("reference_contig_scope")
        )
        gene_filter = _resolve_gene_coverage_filter(
            data.get(
                "cnv_gui_gene_coverage_filter",
                DEFAULT_CNV_GUI_GENE_COVERAGE_FILTER,
            )
        )
        color_mode = _resolve_color_mode(
            data.get("cnv_gui_color_mode", DEFAULT_CNV_GUI_COLOR_MODE)
        )
        show_bp = _resolve_bool(
            data.get("cnv_gui_show_breakpoints", DEFAULT_CNV_GUI_SHOW_BREAKPOINTS),
            DEFAULT_CNV_GUI_SHOW_BREAKPOINTS,
        )
        show_trend = _resolve_bool(
            data.get("cnv_gui_show_trend_line", DEFAULT_CNV_GUI_SHOW_TREND_LINE),
            DEFAULT_CNV_GUI_SHOW_TREND_LINE,
        )
        cutoff = cnv_cutoff_setting(data.get("cnv_gui_cutoff", DEFAULT_CNV_CUTOFF))
        chrom_axis = cnv_chrom_axis_setting(
            data.get("cnv_gui_chrom_axis", DEFAULT_CNV_CHROM_AXIS)
        )
        genome_axis = cnv_genome_axis_setting(
            data.get("cnv_gui_genome_axis", DEFAULT_CNV_GENOME_AXIS)
        )
        gene_label_size = cnv_gene_label_size_setting(
            data.get("cnv_gui_gene_label_size", DEFAULT_CNV_GENE_LABEL_SIZE)
        )
        label_orientation = resolve_cnv_label_orientation(
            data.get("cnv_gui_label_orientation", DEFAULT_CNV_LABEL_ORIENTATION)
        )
        schema_version = int(data.get("schema_version") or 1)
        if schema_version < PLOTTING_PREFERENCES_SCHEMA_VERSION:
            schema_version = PLOTTING_PREFERENCES_SCHEMA_VERSION
        return cls(
            schema_version=schema_version,
            cnv_report_scale=scale,
            reference_contig_scope=contig_scope,
            cnv_gui_gene_coverage_filter=gene_filter,
            cnv_gui_color_mode=color_mode,
            cnv_gui_show_breakpoints=show_bp,
            cnv_gui_show_trend_line=show_trend,
            cnv_gui_cutoff=cutoff,
            cnv_gui_chrom_axis=chrom_axis,
            cnv_gui_genome_axis=genome_axis,
            cnv_gui_gene_label_size=gene_label_size,
            cnv_gui_label_orientation=label_orientation,
            updated_at=data.get("updated_at"),
            updated_by=data.get("updated_by"),
        )

    def with_updates(
        self,
        *,
        cnv_report_scale: Optional[str] = None,
        reference_contig_scope: Optional[str] = None,
        cnv_gui_gene_coverage_filter: Optional[str] = None,
        cnv_gui_color_mode: Optional[str] = None,
        cnv_gui_show_breakpoints: Optional[bool] = None,
        cnv_gui_show_trend_line: Optional[bool] = None,
        cnv_gui_cutoff: Optional[str] = None,
        cnv_gui_chrom_axis: Optional[str] = None,
        cnv_gui_genome_axis: Optional[str] = None,
        cnv_gui_gene_label_size: Optional[str] = None,
        cnv_gui_label_orientation: Optional[str] = None,
        updated_at: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> "PlottingPreferencesConfig":
        scale = cnv_report_scale or self.cnv_report_scale
        scale = _LEGACY_CNV_SCALE_ALIASES.get(scale, scale)
        if scale not in CNV_REPORT_SCALES:
            scale = DEFAULT_CNV_REPORT_SCALE
        contig_scope = resolve_reference_contig_scope(
            reference_contig_scope or self.reference_contig_scope
        )
        gene_filter = _resolve_gene_coverage_filter(
            cnv_gui_gene_coverage_filter
            if cnv_gui_gene_coverage_filter is not None
            else self.cnv_gui_gene_coverage_filter
        )
        color_mode = _resolve_color_mode(
            cnv_gui_color_mode
            if cnv_gui_color_mode is not None
            else self.cnv_gui_color_mode
        )
        show_bp = (
            self.cnv_gui_show_breakpoints
            if cnv_gui_show_breakpoints is None
            else bool(cnv_gui_show_breakpoints)
        )
        show_trend = (
            self.cnv_gui_show_trend_line
            if cnv_gui_show_trend_line is None
            else bool(cnv_gui_show_trend_line)
        )
        cutoff = cnv_cutoff_setting(
            cnv_gui_cutoff if cnv_gui_cutoff is not None else self.cnv_gui_cutoff
        )
        chrom_axis = cnv_chrom_axis_setting(
            cnv_gui_chrom_axis
            if cnv_gui_chrom_axis is not None
            else self.cnv_gui_chrom_axis
        )
        genome_axis = cnv_genome_axis_setting(
            cnv_gui_genome_axis
            if cnv_gui_genome_axis is not None
            else self.cnv_gui_genome_axis
        )
        gene_label_size = cnv_gene_label_size_setting(
            cnv_gui_gene_label_size
            if cnv_gui_gene_label_size is not None
            else self.cnv_gui_gene_label_size
        )
        label_orientation = resolve_cnv_label_orientation(
            cnv_gui_label_orientation
            if cnv_gui_label_orientation is not None
            else self.cnv_gui_label_orientation
        )
        return PlottingPreferencesConfig(
            schema_version=PLOTTING_PREFERENCES_SCHEMA_VERSION,
            cnv_report_scale=scale,
            reference_contig_scope=contig_scope,
            cnv_gui_gene_coverage_filter=gene_filter,
            cnv_gui_color_mode=color_mode,
            cnv_gui_show_breakpoints=show_bp,
            cnv_gui_show_trend_line=show_trend,
            cnv_gui_cutoff=cutoff,
            cnv_gui_chrom_axis=chrom_axis,
            cnv_gui_genome_axis=genome_axis,
            cnv_gui_gene_label_size=gene_label_size,
            cnv_gui_label_orientation=label_orientation,
            updated_at=updated_at or self.updated_at,
            updated_by=updated_by or self.updated_by,
        )


def resolve_cnv_report_scale(scale: Optional[str]) -> str:
    if not scale:
        return DEFAULT_CNV_REPORT_SCALE
    scale = _LEGACY_CNV_SCALE_ALIASES.get(scale, scale)
    if scale in CNV_REPORT_SCALES:
        return scale
    return DEFAULT_CNV_REPORT_SCALE


def cnv_summary_normalized_from_scale(scale: Optional[str]) -> bool:
    """Return True when the configured CNV report scale is log2 ratio mode."""
    return (
        resolve_cnv_report_scale(scale) == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
    )


def load_plotting_preferences(store=None) -> PlottingPreferencesConfig:
    """Load persisted admin plotting preferences from the security store."""
    if store is None:
        from robin.security import SecurityStore

        store = SecurityStore()
    raw = store.get_gui_setting(PLOTTING_PREFERENCES_KEY)
    return PlottingPreferencesConfig.from_dict(raw)


def resolve_cnv_summary_normalized(
    explicit: Optional[bool] = None,
    *,
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> bool:
    """Resolve whether CNV summary/per-chromosome plots use log2 ratio mode.

    An explicit True/False (e.g. CLI ``--cnv-normalized-difference``) overrides
    admin plotting preferences. When ``explicit`` is None, the admin default is used.
    """
    if explicit is not None:
        return bool(explicit)
    prefs = plotting_preferences or load_plotting_preferences()
    return cnv_summary_normalized_from_scale(prefs.cnv_report_scale)


def resolve_plotting_reference_contig_scope(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Resolve the contig scope used for coverage and CNV plots."""
    prefs = plotting_preferences or load_plotting_preferences()
    return resolve_reference_contig_scope(prefs.reference_contig_scope)


def resolve_cnv_gui_gene_coverage_filter(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default Coverage genes filter for the live CNV GUI."""
    prefs = plotting_preferences or load_plotting_preferences()
    return _resolve_gene_coverage_filter(prefs.cnv_gui_gene_coverage_filter)


def resolve_cnv_gui_color_mode(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default Color-by mode for the live CNV GUI."""
    prefs = plotting_preferences or load_plotting_preferences()
    return _resolve_color_mode(prefs.cnv_gui_color_mode)


def resolve_cnv_gui_show_trend_line(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> bool:
    """Default trend-line visibility for the live CNV GUI."""
    prefs = plotting_preferences or load_plotting_preferences()
    return bool(prefs.cnv_gui_show_trend_line)


def resolve_cnv_gui_cutoff(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default CNV cut-off setting for the live GUI and exported figures."""
    prefs = plotting_preferences or load_plotting_preferences()
    return cnv_cutoff_setting(prefs.cnv_gui_cutoff)


def resolve_cnv_gui_genome_axis(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Stored genome-wide Y-range setting for the GUI selector."""
    prefs = plotting_preferences or load_plotting_preferences()
    return cnv_genome_axis_setting(prefs.cnv_gui_genome_axis)


def resolve_cnv_gui_chrom_axis(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default per-chromosome Y-range setting for the GUI and exported figures."""
    prefs = plotting_preferences or load_plotting_preferences()
    return cnv_chrom_axis_setting(prefs.cnv_gui_chrom_axis)


def resolve_cnv_chromosome_axis_log2(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> Optional[float]:
    """Fixed per-chromosome log2 half-span, or None when set to auto."""
    prefs = plotting_preferences or load_plotting_preferences()
    return resolve_cnv_chrom_axis(prefs.cnv_gui_chrom_axis)


def resolve_cnv_genome_axis_log2(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> Optional[float]:
    """Fixed genome-wide log2 half-span, or None when set to auto."""
    prefs = plotting_preferences or load_plotting_preferences()
    return resolve_cnv_chrom_axis(prefs.cnv_gui_genome_axis)


def resolve_cnv_gui_gene_label_size(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default CNV gene label size setting for the GUI and exported figures."""
    prefs = plotting_preferences or load_plotting_preferences()
    return cnv_gene_label_size_setting(prefs.cnv_gui_gene_label_size)


def resolve_cnv_gene_label_points(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> float:
    """CNV gene label size in points, for the matplotlib report figures."""
    prefs = plotting_preferences or load_plotting_preferences()
    return resolve_cnv_gene_label_size(prefs.cnv_gui_gene_label_size)


def resolve_cnv_gui_label_orientation(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default gene label orientation for the GUI and exported figures."""
    prefs = plotting_preferences or load_plotting_preferences()
    return resolve_cnv_label_orientation(prefs.cnv_gui_label_orientation)


def resolve_cnv_gui_show_breakpoints(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> bool:
    """Default Breakpoints visibility for single-chromosome CNV GUI view."""
    prefs = plotting_preferences or load_plotting_preferences()
    return bool(prefs.cnv_gui_show_breakpoints)


def resolve_cnv_gui_y_scale(
    plotting_preferences: Optional[PlottingPreferencesConfig] = None,
) -> str:
    """Default Y-axis mode for the live CNV GUI (``linear`` or ``log``)."""
    if resolve_cnv_summary_normalized(None, plotting_preferences=plotting_preferences):
        return "log"
    return "linear"


def cnv_report_ylabel(scale: str) -> str:
    if resolve_cnv_report_scale(scale) == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE:
        return "Log2 ratio (ploidy / expected)"
    return "Estimated copy number / ploidy"


def cnv_report_genome_ylabel(scale: str) -> str:
    if resolve_cnv_report_scale(scale) == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE:
        return "Log2 ratio (ploidy / expected)"
    return "Estimated ploidy"


def cnv_report_genome_ylabel_mathtext(scale: str) -> str:
    """Matplotlib axis label using the same sans-serif family as other report text."""
    return cnv_report_genome_ylabel(scale)


CNV_CLINICAL_TRIAL_LEGEND = (
    "Genes labelled in purple represent current clinical trial targets."
)


def cnv_calling_cutoff_magnitude() -> float:
    """Autosomal calling cut-off magnitude, read from the configured thresholds.

    Taken from ``classification_config`` rather than written into the caption, so
    the number a report prints cannot drift away from the number it called on.
    """
    from robin.classification_config import get_cnv_thresholds

    try:
        gain, _loss = get_cnv_thresholds("chr1", "Unknown")
        return abs(float(gain))
    except Exception:  # pragma: no cover - configuration would have to be malformed
        return 0.3


def cnv_cutoff_label(cutoff_override: Optional[float] = None) -> str:
    """Short label naming the cut-off in force, e.g. "±0.35, non-default".

    Every table and card whose contents move with the cut-off carries this, so a
    number can never be read without the threshold that produced it. A CNV load
    of 23% and one of 19% can be the same sample at two cut-offs; without the
    label there is nothing on the page to tell them apart.

    Non-default is called out explicitly rather than left to be inferred from the
    number, because a reader scanning a report will not know the site's
    configured value by heart.
    """
    calling = cnv_calling_cutoff_magnitude()
    if cutoff_override:
        magnitude = abs(float(cutoff_override))
        if magnitude and magnitude != calling:
            return f"±{magnitude:g}, non-default"
    return f"±{calling:g}, calling default"


def cnv_cutoff_heading_suffix(cutoff_override: Optional[float] = None) -> str:
    """Parenthesised cut-off for a table heading, e.g. " (cut-off ±0.35, non-default)"."""
    return f" (cut-off {cnv_cutoff_label(cutoff_override)})"


def cnv_report_plot_caption(
    scale: str,
    *,
    has_clinical_trial_genes: bool = False,
    cutoff_override: Optional[float] = None,
    baseline_shift: float = 0.0,
) -> str:
    """Caption printed under the genome-wide CNV figure in the PDF report.

    ``cutoff_override`` is the review cut-off everything in the report was called
    at, when one is set. It is stated on the figure because a report called at a
    non-default cut-off is not comparable with one called at the default, and the
    reader has no other way to tell which they are holding.
    """
    calling = cnv_calling_cutoff_magnitude()
    drawn = abs(float(cutoff_override)) if cutoff_override else calling
    overridden = bool(cutoff_override) and drawn != calling
    if resolve_cnv_report_scale(scale) == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE:
        if overridden:
            cutoff_text = (
                f"dashed amber lines = the ±{drawn:g} cut-off this report was "
                f"called at, set in place of the ±{calling:g} default."
            )
        else:
            cutoff_text = "dashed amber lines = gain / loss calling cut-offs."
        caption = (
            "Copy number variation across chromosomes "
            "(log2 ratio of observed ploidy to expected copy number; 0 = normal; "
            f"blue = gain, red = loss, grey = within ±{drawn:g}). "
            f"Dark line = segment level; {cutoff_text} Panel targets are marked on "
            "the profile at their own copy number and labelled by gene name."
        )
    else:
        caption = (
            "Copy number variation across chromosomes "
            "(blue = gain, red = loss). "
            "Dark line = segment level. Panel targets are marked on the profile at "
            "their own copy number and labelled by gene name."
        )
    if baseline_shift:
        # Ploidy is scaled to the genome mean, which is not the diploid level on
        # an aneuploid genome. The track is centred on its modal level; saying so
        # keeps a correction that moves every value from being invisible.
        caption += (
            f" Baseline centred on the modal copy number "
            f"({baseline_shift:+.2f} log2 applied)."
        )
    if has_clinical_trial_genes:
        caption += " " + CNV_CLINICAL_TRIAL_LEGEND
    return caption
