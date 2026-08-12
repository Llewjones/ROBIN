from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robin.reporting.plotting import create_CNV_plot


class _FakeResult:
    def __init__(self, cnv: dict) -> None:
        self.cnv = cnv


def _sample_cnv() -> dict:
    return {
        "chr1": np.array([2.0, 2.1, 1.9, 2.0]),
        "chr2": np.array([2.0, 2.2, 2.1, 2.0]),
    }


def test_create_cnv_plot_ploidy_mode_returns_jpeg() -> None:
    result = _FakeResult(_sample_cnv())
    buf = create_CNV_plot(result, {"bin_width": 1_000_000})
    data = buf.getvalue()
    assert data[:2] == b"\xff\xd8"


def test_log2_linear_axis_limits_uses_percentile_with_minimum_span() -> None:
    from robin.cnv_plot_style import CNV_LOG2_MIN_AXIS_SPAN
    from robin.reporting.plotting import _log2_linear_axis_limits

    # A tight cluster gets a tight symmetric window, so small changes stay legible.
    tight = np.array([0.0, 0.05, -0.05, 0.1, -0.1])
    y_min, y_max = _log2_linear_axis_limits(tight)
    assert y_min == pytest.approx(-CNV_LOG2_MIN_AXIS_SPAN)
    assert y_max == pytest.approx(CNV_LOG2_MIN_AXIS_SPAN)
    assert y_min < 0 < y_max

    # Outliers can expand up to the upper cap.
    wide = np.array([0.0] * 95 + [2.5, -2.5, 3.0, -3.0, 0.1])
    y_min, y_max = _log2_linear_axis_limits(wide)
    assert y_max <= 4.0 + 0.01
    assert y_min >= -4.0 - 0.01
    assert y_min < 0 < y_max


def test_report_log2_from_ploidy_matches_gui() -> None:
    """Report CNV log2 mode uses the same transform as the GUI."""
    from robin.analysis.cnv_analysis import compute_cnv_log2_from_ploidy

    cnv = {"chr5": np.array([3.5, 3.5, 3.5])}
    log2 = compute_cnv_log2_from_ploidy(cnv, "Male")
    np.testing.assert_allclose(log2["chr5"], np.log2(3.5 / 2.0), rtol=1e-6)


def test_create_cnv_plot_log2_ratio_mode_returns_jpeg() -> None:
    result = _FakeResult(_sample_cnv())
    log2_ratios = {
        "chr1": np.array([0.1, -0.2, 0.0, 0.3]),
        "chr2": np.array([-0.1, 0.4, -0.3, 0.0]),
    }
    buf = create_CNV_plot(
        result,
        {"bin_width": 1_000_000},
        normalized_cnv=log2_ratios,
        use_normalized_difference=True,
        plot_bin_width=1_000_000,
        sex_estimate="Female",
    )
    data = buf.getvalue()
    assert data[:2] == b"\xff\xd8"


def test_cnv_plot_point_state_uses_calling_thresholds() -> None:
    from robin.reporting.plotting import _cnv_plot_point_state

    assert _cnv_plot_point_state(0.5, "chr7", "Female") == "gain"
    assert _cnv_plot_point_state(-0.5, "chr7", "Female") == "loss"
    assert _cnv_plot_point_state(0.1, "chr7", "Female") == "neutral"


def test_collect_genome_significant_panel_points_offsets_by_chromosome() -> None:
    from robin.reporting.plotting import _collect_genome_significant_panel_points

    panel_genes = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start_pos": [1_000_000, 1_000_000],
            "end_pos": [2_000_000, 2_000_000],
            "gene": ["GENE1", "GENE2"],
        }
    )
    chr1_vals = np.zeros(100, dtype=float)
    chr1_vals[1] = 1.0
    chr2_vals = np.zeros(80, dtype=float)
    chr2_vals[1] = -1.0
    cnv_source = {
        "chr1": chr1_vals,
        "chr2": chr2_vals,
    }
    chrom_offsets = {"chr1": 0.0, "chr2": 100_000_000.0}
    target_coverage = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "startpos": [1_000_000, 1_000_000],
            "endpos": [2_000_000, 2_000_000],
            "name": ["GENE1", "GENE2"],
            "length": [1_000_000, 1_000_000],
            "coverage": [30.0, 25.0],
            "bases": [30_000_000, 25_000_000],
        }
    )
    significant_regions = {}

    points = _collect_genome_significant_panel_points(
        panel_genes,
        cnv_source,
        ["chr1", "chr2"],
        chrom_offsets,
        1_000_000,
        significant_regions,
        target_coverage,
        use_log2=True,
        scale_mean_cnv=0.0,
    )

    assert len(points) == 2
    by_label = {point["label"]: point for point in points}
    assert by_label["GENE1"]["direction"] == "gain"
    assert by_label["GENE2"]["direction"] == "loss"
    assert by_label["GENE1"]["position_bp"] == 1_500_000
    assert by_label["GENE2"]["position_bp"] == 101_500_000


def test_layout_panel_coverage_point_labels_anchors_on_the_marker() -> None:
    """Rotated names sit just clear of their own marker, not in a reserved band."""
    from robin.reporting.plotting import _layout_panel_coverage_point_labels

    points = [
        {"label": "GAIN", "position_bp": 10_000_000.0, "y_norm": 0.4, "direction": "gain"},
        {"label": "LOSS", "position_bp": 200_000_000.0, "y_norm": -0.4, "direction": "loss"},
    ]
    layouts = _layout_panel_coverage_point_labels(
        points,
        -2.0,
        2.0,
        250_000_000.0,
        x_key="position_bp",
        min_x_spacing=1_800_000.0,
    )
    gain_y = layouts[("GAIN", 10_000_000.0)]
    loss_y = layouts[("LOSS", 200_000_000.0)]
    # Gains read upwards from their marker, losses downwards.
    assert 0.4 < gain_y < 0.6
    assert -0.6 < loss_y < -0.4


def test_layout_panel_coverage_point_labels_flips_side_near_the_panel_edge() -> None:
    """A long name has nowhere to go above a marker near the top, so it flips."""
    from robin.reporting.plotting import _layout_panel_coverage_point_labels

    points = [
        {"label": "KIAA1549", "position_bp": 10_000_000.0, "y_norm": 0.95,
         "direction": "gain"},
    ]
    layouts = _layout_panel_coverage_point_labels(
        points, -1.0, 1.0, 250_000_000.0,
        x_key="position_bp", min_x_spacing=1_800_000.0,
    )
    assert layouts[("KIAA1549", 10_000_000.0)] < 0.95


def test_should_label_panel_gene_ignores_genes_inside_the_cutoff() -> None:
    from robin.reporting.plotting import _should_label_panel_gene

    point = {"cnv_val": 0.1, "mid_mb": 1.5}
    assert _should_label_panel_gene(point, chromosome="chr1", use_log2=True) is False


def test_is_gene_cnv_outlier_uses_the_calling_cutoffs() -> None:
    """The label rule is the same +/-0.3 cut-off the plot and events table use."""
    from robin.reporting.plotting import _is_gene_cnv_outlier

    assert _is_gene_cnv_outlier(0.25, chromosome="chr1", use_log2=True) is False
    assert _is_gene_cnv_outlier(0.35, chromosome="chr1", use_log2=True) is True
    assert _is_gene_cnv_outlier(-0.35, chromosome="chr1", use_log2=True) is True


def test_gene_on_a_gained_chromosome_is_reported() -> None:
    """A per-chromosome SD rule used to cancel out whole-chromosome events."""
    from robin.reporting.plotting import _is_gene_cnv_outlier

    # EGFR at log2 0.5 on a chromosome that is gained end to end.
    assert _is_gene_cnv_outlier(0.5, chromosome="chr7", use_log2=True) is True


def test_is_gene_cnv_outlier_handles_ploidy_scale() -> None:
    from robin.reporting.plotting import _is_gene_cnv_outlier

    # 3 copies against a diploid baseline is log2(3/2) = 0.58, past the cut-off.
    assert _is_gene_cnv_outlier(
        3.0, chromosome="chr1", use_log2=False, baseline=2.0
    ) is True
    assert _is_gene_cnv_outlier(
        2.1, chromosome="chr1", use_log2=False, baseline=2.0
    ) is False


def test_mean_target_coverage_uses_all_targets() -> None:
    from robin.reporting.plotting import _mean_target_coverage

    df = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "coverage": [10.0, 20.0, 30.0],
        }
    )
    assert _mean_target_coverage(df) == 20.0
    assert _mean_target_coverage(None) is None
    assert _mean_target_coverage(pd.DataFrame()) is None


def test_add_panel_coverage_points_chromosome_axis_uses_mid_mb() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_panel_coverage_points

    ax_cnv = MagicMock()

    panel_points = [
        {
            "mid_mb": 12.5,
            "coverage_val": 30.0,
            "y_norm": 0.6,
            "baseline_y": 0.0,
            "direction": "gain",
            "label": "GENE1",
        },
    ]

    assert (
        _add_panel_coverage_points(
            ax_cnv,
            panel_points,
            120.0,
            x_key="mid_mb",
            min_x_spacing=1.8,
            y_min=-2.0,
            y_max=2.0,
        )
        is True
    )
    # Markers share the CNV axis; no secondary coverage axis is created.
    ax_cnv.twinx.assert_not_called()
    scatter_x = ax_cnv.scatter.call_args[0][0]
    assert scatter_x == [12.5]


def test_add_genome_panel_coverage_points_draws_scatter_and_labels() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_genome_panel_coverage_points

    ax_cnv = MagicMock()

    panel_points = [
        {"position_bp": 1_000_000.0, "coverage_val": 20.0, "y_norm": 0.5,
         "baseline_y": 0.0, "direction": "gain", "label": "GENE1"},
        {"position_bp": 2_000_000.0, "coverage_val": 40.0, "y_norm": -0.5,
         "baseline_y": 0.0, "direction": "loss", "label": "GENE2"},
        {"position_bp": 3_000_000.0, "coverage_val": 30.0, "y_norm": 0.7,
         "baseline_y": 0.0, "direction": "gain", "label": "GENE3"},
    ]

    assert (
        _add_genome_panel_coverage_points(
            ax_cnv, panel_points, 250_000_000.0, y_min=-2.0, y_max=2.0
        )
        is True
    )
    # One scatter call per direction, one text label per gene, no second axis.
    ax_cnv.twinx.assert_not_called()
    assert ax_cnv.scatter.call_count == 2
    assert ax_cnv.text.call_count == 3


def test_panel_markers_sit_on_the_profile_at_the_gene_cnv_value() -> None:
    """Markers take their height from the gene's own CNV value, not from depth."""
    from robin.reporting.plotting import _attach_normalised_coverage

    panel_points = [
        {"position_bp": 1_000_000.0, "coverage_val": 20.0, "cnv_val": 0.8, "label": "GENE1"},
        {"position_bp": 2_000_000.0, "coverage_val": 40.0, "cnv_val": -0.6, "label": "GENE2"},
        {"position_bp": 3_000_000.0, "coverage_val": 30.0, "cnv_val": 0.0, "label": "GENE3"},
    ]

    placed = _attach_normalised_coverage(
        panel_points, mean_cov=30.0, scale_mean_cnv=0.0, use_log2=True
    )

    by_label = {point["label"]: point for point in placed}
    assert by_label["GENE1"]["y_norm"] == pytest.approx(0.8)
    assert by_label["GENE2"]["y_norm"] == pytest.approx(-0.6)
    assert by_label["GENE3"]["y_norm"] == pytest.approx(0.0)
    # Depth is kept for the caption as a ratio to the panel mean.
    assert by_label["GENE1"]["coverage_ratio"] == pytest.approx(20.0 / 30.0)


def test_panel_markers_survive_missing_coverage() -> None:
    """A gene with no target coverage is still marked at its CNV value."""
    from robin.reporting.plotting import _attach_normalised_coverage

    placed = _attach_normalised_coverage(
        [{"position_bp": 1.0, "coverage_val": None, "cnv_val": 0.5, "label": "GENE1"}],
        mean_cov=None,
        scale_mean_cnv=0.0,
        use_log2=True,
    )
    assert placed[0]["y_norm"] == pytest.approx(0.5)
    assert placed[0]["coverage_ratio"] is None


def test_downsample_cnv_for_plot_groups_values() -> None:
    from robin.analysis.cnv_analysis import downsample_cnv_for_plot

    values = np.array([1.0, 3.0, 5.0, 7.0], dtype=float)
    x_bp, out = downsample_cnv_for_plot(values, analysis_bin_width=12_000, plot_bin_width=24_000)
    assert len(out) == 2
    assert out[0] == 2.0
    assert out[1] == 6.0
    assert x_bp[0] == 12_000
    assert x_bp[1] == 36_000


def test_downsample_cnv_chromosome_track_keeps_full_x_axis() -> None:
    from robin.analysis.cnv_analysis import downsample_cnv_chromosome_track

    analysis_bw = 12_000
    n_bins = 1000
    values = np.linspace(0.0, 1.0, n_bins)
    x_mb, out, x_max_mb = downsample_cnv_chromosome_track(
        values, analysis_bw, plot_bin_width=500_000,
    )
    assert x_max_mb == n_bins * analysis_bw / 1_000_000
    assert len(out) < n_bins
    assert len(x_mb) == len(out)
    assert float(x_mb[-1]) < x_max_mb


def test_collect_chromosome_significant_panel_points_requires_cnv_outlier() -> None:
    from robin.reporting.plotting import _collect_chromosome_significant_panel_points

    panel_genes = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start_pos": [1_000_000, 50_000_000],
            "end_pos": [2_000_000, 51_000_000],
            "gene": ["OUTLIER", "NORMAL"],
        }
    )
    values = np.zeros(100, dtype=float)
    values[1] = 1.0
    target_coverage = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "startpos": [1_000_000, 50_000_000],
            "endpos": [2_000_000, 51_000_000],
            "name": ["OUTLIER", "NORMAL"],
            "length": [1_000_000, 1_000_000],
            "coverage": [40.0, 20.0],
            "bases": [40_000_000, 20_000_000],
        }
    )
    points = _collect_chromosome_significant_panel_points(
        panel_genes,
        "chr1",
        values,
        1_000_000,
        [],
        target_coverage,
        use_log2=True,
        baseline=0.0,
    )

    assert len(points) == 1
    assert points[0]["label"] == "OUTLIER"


def test_add_chromosome_panel_coverage_overlay_uses_shared_cnv_axis() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_chromosome_panel_coverage_overlay

    ax_cnv = MagicMock()
    ax_cnv.get_xlim.return_value = (0.0, 120.0)

    panel_points = [
        {
            "mid_mb": 12.5,
            "coverage_val": 30.0,
            "y_norm": 0.6,
            "baseline_y": 0.0,
            "direction": "gain",
            "label": "GENE1",
        },
    ]

    assert (
        _add_chromosome_panel_coverage_overlay(
            ax_cnv,
            panel_points,
            120.0,
            y_min=-2.0,
            y_max=2.0,
        )
        is True
    )
    # Coverage is normalised onto the CNV axis rather than a second right-hand axis.
    ax_cnv.twinx.assert_not_called()
    assert ax_cnv.scatter.call_args[0][0] == [12.5]


def test_create_cnv_plot_per_chromosome_log2_mode_returns_jpeg() -> None:
    from robin.reporting.plotting import create_CNV_plot_per_chromosome

    chr1_vals = np.zeros(40, dtype=float)
    chr1_vals[5] = 1.0
    sample = {
        "chr1": chr1_vals,
        "chr2": np.linspace(0.1, -0.1, 30),
    }
    log2 = {
        "chr1": chr1_vals,
        "chr2": np.linspace(0.1, -0.1, 30),
    }
    panel_genes = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "start_pos": [5_000_000],
            "end_pos": [6_000_000],
            "gene": ["TEST1"],
        }
    )
    target_coverage = pd.DataFrame(
        {
            "chrom": ["chr1"],
            "startpos": [5_000_000],
            "endpos": [6_000_000],
            "name": ["TEST1"],
            "length": [1_000_000],
            "coverage": [25.0],
            "bases": [25_000_000],
        }
    )

    class _FakeResult:
        cnv = sample

    plots = create_CNV_plot_per_chromosome(
        _FakeResult(),
        {"bin_width": 1_000_000},
        chromosomes=["chr1"],
        panel_genes_df=panel_genes,
        normalized_cnv=log2,
        target_coverage_df=target_coverage,
        use_log2_ratio=True,
        sex_estimate="Female",
    )
    assert len(plots) == 1
    assert plots[0][0] == "chr1"
    assert plots[0][1].getvalue()[:2] == b"\xff\xd8"


def test_per_chromosome_plot_uses_analysis_bin_width_by_default() -> None:
    from unittest.mock import patch

    from robin.reporting.plotting import create_CNV_plot_per_chromosome

    n_bins = 80
    chr1_vals = np.linspace(-0.2, 0.2, n_bins)

    class _FakeResult:
        cnv = {"chr1": chr1_vals}

    with patch(
        "robin.analysis.cnv_analysis.downsample_cnv_chromosome_track",
        wraps=__import__(
            "robin.analysis.cnv_analysis", fromlist=["downsample_cnv_chromosome_track"]
        ).downsample_cnv_chromosome_track,
    ) as mock_downsample:
        plots = create_CNV_plot_per_chromosome(
            _FakeResult(),
            {"bin_width": 100_000},
            chromosomes=["chr1"],
            normalized_cnv={"chr1": chr1_vals},
            use_log2_ratio=True,
            sex_estimate="Female",
        )

    assert len(plots) == 1
    mock_downsample.assert_called_once()
    assert mock_downsample.call_args.args[1] == 100_000
    assert mock_downsample.call_args.args[2] == 100_000


def test_cnv_chromosome_fig_height_fits_four_per_page() -> None:
    from robin.reporting.plotting import (
        CNV_CHROMOSOME_PLOTS_PER_PAGE,
        CNV_CHROMOSOME_PLOT_SPACER_PT,
        CNV_REPORT_FRAME_PADDING_PT,
        cnv_chromosome_fig_height_for_page,
    )

    page_height = 9.34
    plot_height = cnv_chromosome_fig_height_for_page(page_height)
    spacer_inch = (CNV_CHROMOSOME_PLOTS_PER_PAGE - 1) * CNV_CHROMOSOME_PLOT_SPACER_PT / 72.0
    frame_inch = page_height - CNV_REPORT_FRAME_PADDING_PT / 72.0
    assert (
        CNV_CHROMOSOME_PLOTS_PER_PAGE * plot_height + spacer_inch
        <= frame_inch + 1e-6
    )
    assert plot_height < 2.5


def test_twelve_chromosome_pdf_images_fit_three_pages() -> None:
    import io

    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, SimpleDocTemplate, Spacer

    from robin.reporting.plotting import (
        cnv_chromosome_fig_height_for_page,
        cnv_chromosome_fig_width_for_page,
    )

    doc = SimpleDocTemplate(
        "/tmp/chrom_layout_test.pdf",
        pagesize=A4,
        rightMargin=1.0 * inch,
        leftMargin=1.0 * inch,
        topMargin=1.35 * inch,
        bottomMargin=1.0 * inch,
    )
    width_inch = cnv_chromosome_fig_width_for_page(doc.width / inch)
    height_inch = cnv_chromosome_fig_height_for_page(doc.height / inch)
    width = width_inch * inch
    height = height_inch * inch

    elements = []
    for plot_idx in range(12):
        buf = io.BytesIO()
        PILImage.new("RGB", (int(width_inch * 100), int(height_inch * 100)), "white").save(
            buf,
            format="JPEG",
        )
        buf.seek(0)
        elements.append(Image(buf, width=width, height=height))
        last_on_page = (plot_idx + 1) % 4 == 0
        if plot_idx < 11 and not last_on_page:
            elements.append(Spacer(1, 6))

    doc.build(elements)
    from PyPDF2 import PdfReader

    page_count = len(PdfReader("/tmp/chrom_layout_test.pdf").pages)
    assert page_count == 3


def _chromosome_axes(use_log2=True, **kwargs):
    """Render a few per-chromosome figures and return {chromosome: ylim}."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    rng = np.random.default_rng(11)
    bw = 1_000_000
    lengths = {"chr1": 120, "chr7": 100, "chr9": 90, "chr10": 80}
    cnv = {}
    for name, n in lengths.items():
        values = np.abs(rng.normal(2.0, 0.2, n))
        if name == "chr7":
            values[:] = np.abs(rng.normal(2.9, 0.2, n))     # gain
        if name == "chr10":
            values[:] = np.abs(rng.normal(1.15, 0.2, n))    # loss
        if name == "chr9":
            values[20:24] = 0.28                            # deep deletion, off scale
        cnv[name] = values
    log2 = {c: np.log2(v / 2.0) for c, v in cnv.items()}

    class _Result:
        pass

    result = _Result()
    result.cnv = cnv

    out = {}
    for contig, fig in iter_CNV_chromosome_figures(
        result,
        {"bin_width": bw},
        normalized_cnv=log2,
        use_log2_ratio=use_log2,
        sex_estimate="XY",
        **kwargs,
    ):
        out[contig] = tuple(round(v, 3) for v in fig.axes[0].get_ylim())
        plt.close(fig)
    return out


def test_per_chromosome_plots_share_one_fixed_axis() -> None:
    """Pages have to be comparable with each other, so the window is fixed."""
    from robin.cnv_plot_style import CNV_CHROMOSOME_AXIS_LOG2

    axes = _chromosome_axes()
    assert len(axes) == 4
    assert set(axes.values()) == {
        (-CNV_CHROMOSOME_AXIS_LOG2, CNV_CHROMOSOME_AXIS_LOG2)
    }


def test_fixed_axis_holds_even_with_a_deep_deletion() -> None:
    """A homozygous deletion must not stretch the axis away from its neighbours."""
    axes = _chromosome_axes()
    assert axes["chr9"] == axes["chr1"]


def test_per_chromosome_axis_can_be_data_driven() -> None:
    axes = _chromosome_axes(fixed_axis_log2=None)
    assert len(set(axes.values())) > 1


def test_ploidy_mode_shares_the_equivalent_fixed_window() -> None:
    axes = _chromosome_axes(use_log2=False)
    assert len(set(axes.values())) == 1
    low, high = next(iter(axes.values()))
    assert low > 0
    assert high > 2.0


def test_offscale_bins_are_flagged_at_the_panel_edge() -> None:
    """Nothing may be silently hidden by the fixed window."""
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _mark_offscale_bins

    ax = MagicMock()
    flagged = _mark_offscale_bins(
        ax,
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 2.5, -2.5, 0.1],
        -1.2,
        1.2,
    )
    assert flagged == 2
    assert ax.plot.call_count == 2


def test_offscale_marker_is_skipped_when_everything_fits() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _mark_offscale_bins

    ax = MagicMock()
    assert _mark_offscale_bins(ax, [1.0, 2.0], [0.1, -0.2], -1.2, 1.2) == 0
    ax.plot.assert_not_called()


def test_centromere_boundaries_cover_every_chromosome() -> None:
    """The p/q divider needs a boundary for each reportable chromosome."""
    from robin.analysis.cnv_regional import load_centromere_boundaries

    boundaries = load_centromere_boundaries()
    for chrom in ("chr1", "chr7", "chr9", "chr13", "chrX", "chrY"):
        assert chrom in boundaries
        assert boundaries[chrom] > 0
    # chr1's centromere is around 123 Mb on GRCh38.
    assert 120e6 < boundaries["chr1"] < 126e6


def test_centromere_line_is_drawn_faint_and_behind_the_data() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_centromere_line

    ax = MagicMock()
    _add_centromere_line(ax, [123.4, 60.1])
    assert ax.axvline.call_count == 2
    kwargs = ax.axvline.call_args.kwargs
    assert kwargs["zorder"] == 0          # behind the bins and the segment line
    assert kwargs["linestyle"] != "-"     # dashed
    assert kwargs["alpha"] < 1.0          # faint


def test_centromere_line_skips_missing_positions() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_centromere_line

    ax = MagicMock()
    _add_centromere_line(ax, [None, float("nan"), 12.0])
    assert ax.axvline.call_count == 1


def test_report_uses_blue_for_gain_and_red_for_loss() -> None:
    """The PDF report convention, agreed with the reporting team."""
    from robin.reporting.plotting import CNV_COLORS

    def _channel(colour, index):
        return int(colour.lstrip("#")[index * 2 : index * 2 + 2], 16)

    gain, loss = CNV_COLORS["plot_gain"], CNV_COLORS["plot_loss"]
    assert _channel(gain, 2) > _channel(gain, 0)   # gain is blue-dominant
    assert _channel(loss, 0) > _channel(loss, 2)   # loss is red-dominant


def test_clinical_trial_targets_are_purple_whichever_way_they_went() -> None:
    from robin.reporting.plotting import CNV_COLORS, _panel_point_color

    trial_gain = {"direction": "gain", "clinical_trial": True}
    trial_loss = {"direction": "loss", "clinical_trial": True}
    assert _panel_point_color(trial_gain) == CNV_COLORS["plot_trial"]
    assert _panel_point_color(trial_loss) == CNV_COLORS["plot_trial"]


def test_ordinary_panel_genes_keep_the_gain_loss_colours() -> None:
    from robin.reporting.plotting import CNV_COLORS, _panel_point_color

    assert _panel_point_color({"direction": "gain"}) == CNV_COLORS["plot_gain"]
    assert _panel_point_color({"direction": "loss"}) == CNV_COLORS["plot_loss"]
    assert (
        _panel_point_color({"direction": "gain", "clinical_trial": False})
        == CNV_COLORS["plot_gain"]
    )


def test_trial_tagging_matches_gene_names_case_insensitively() -> None:
    from robin.reporting.plotting import _tag_clinical_trial_points

    tagged = _tag_clinical_trial_points(
        [{"label": "erbb2"}, {"label": "TP53"}], ["ERBB2", "MET"]
    )
    assert tagged[0]["clinical_trial"] is True
    assert tagged[1]["clinical_trial"] is False


def test_trial_tagging_is_a_no_op_without_a_list() -> None:
    from robin.reporting.plotting import _tag_clinical_trial_points

    points = [{"label": "ERBB2"}]
    assert _tag_clinical_trial_points(points, []) is points


def test_report_caption_names_the_purple_convention() -> None:
    from robin.gui.plotting_preferences import cnv_report_plot_caption

    with_trials = cnv_report_plot_caption(
        "normalized_difference", has_clinical_trial_genes=True
    )
    without = cnv_report_plot_caption("normalized_difference")

    assert "purple" in with_trials
    assert "Step 2" in with_trials
    assert "purple" not in without
    # Both state the swapped gain/loss convention.
    for caption in (with_trials, without):
        assert "blue = gain" in caption
        assert "red = loss" in caption


def test_clinical_trial_gene_defaults_cover_the_agreed_panel() -> None:
    from robin.workflow_config import get_cnv_clinical_trial_genes

    genes = get_cnv_clinical_trial_genes({})
    for gene in ("ERBB2", "MET", "BRCA1", "BRCA2", "CDK12", "MTAP", "FGFR1"):
        assert gene in genes


def test_clinical_trial_genes_can_be_overridden_or_disabled() -> None:
    from robin.workflow_config import get_cnv_clinical_trial_genes

    assert get_cnv_clinical_trial_genes(
        {"cnv": {"clinical_trial_genes": ["EGFR"]}}
    ) == ("EGFR",)
    assert get_cnv_clinical_trial_genes({"cnv": {"clinical_trial_genes": []}}) == ()


def test_clinical_trial_legend_is_drawn_on_the_figure() -> None:
    """The legend must travel with exported PDFs, not just report body text."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import _add_clinical_trial_legend

    fig = plt.figure()
    try:
        assert _add_clinical_trial_legend(fig, [{"clinical_trial": True}]) is True
        texts = [t.get_text() for t in fig.texts]
        assert any("Step 2 targets" in t for t in texts)
    finally:
        plt.close(fig)


def test_clinical_trial_legend_is_skipped_without_a_purple_gene() -> None:
    """No claim about purple when nothing purple is on the plot."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import _add_clinical_trial_legend

    fig = plt.figure()
    try:
        assert _add_clinical_trial_legend(fig, [{"clinical_trial": False}]) is False
        assert _add_clinical_trial_legend(fig, []) is False
        assert fig.texts == []
    finally:
        plt.close(fig)


def test_scatter_alpha_is_strong_enough_to_read() -> None:
    """The reporting team asked twice for more intense points."""
    from robin.reporting.plotting import (
        CNV_POINT_ALPHA_CALLED,
        CNV_POINT_ALPHA_NEUTRAL,
    )

    assert CNV_POINT_ALPHA_CALLED >= 0.6
    assert CNV_POINT_ALPHA_NEUTRAL >= 0.3
    # Called points must still stand out from the neutral background.
    assert CNV_POINT_ALPHA_CALLED > CNV_POINT_ALPHA_NEUTRAL


def test_clamped_gene_marker_records_its_true_value() -> None:
    """Regression: a pinned marker read as its clamped value.

    The same gene then appeared at a different level on the genome-wide panel,
    whose axis is fitted to the data rather than fixed.
    """
    from robin.reporting.plotting import _clamp_coverage_points_to_band

    points = [
        {"label": "EGFR", "y_norm": 1.433},
        {"label": "TP53", "y_norm": -0.4},
    ]
    clamped = _clamp_coverage_points_to_band(points, -1.2, 1.2)

    egfr, tp53 = clamped
    assert egfr["y_norm"] == pytest.approx(1.2)      # pinned to the edge
    assert egfr["true_value"] == pytest.approx(1.433)  # but the truth is kept
    assert egfr["offscale"] is True
    # A marker inside the window is untouched and not flagged.
    assert tp53["y_norm"] == pytest.approx(-0.4)
    assert tp53["offscale"] is False


def test_clamping_leaves_non_finite_markers_alone() -> None:
    from robin.reporting.plotting import _clamp_coverage_points_to_band

    clamped = _clamp_coverage_points_to_band(
        [{"label": "X", "y_norm": float("nan")}], -1.0, 1.0
    )
    assert "offscale" not in clamped[0]


def test_offscale_gene_marker_is_drawn_as_an_arrow() -> None:
    """An arrow says 'beyond the axis'; a dot would claim it sits at the edge."""
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_panel_coverage_points

    ax = MagicMock()
    _add_panel_coverage_points(
        ax,
        [
            {
                "mid_mb": 55.0,
                "coverage_val": 120.0,
                "y_norm": 1.2,
                "true_value": 1.433,
                "offscale": True,
                "baseline_y": 0.0,
                "direction": "gain",
                "label": "EGFR",
            }
        ],
        159.0,
        x_key="mid_mb",
        min_x_spacing=1.8,
        y_min=-1.2,
        y_max=1.2,
    )
    markers = [call.kwargs.get("marker") for call in ax.scatter.call_args_list]
    assert "^" in markers
    assert "o" not in markers


def test_gene_markers_inside_the_window_stay_round() -> None:
    from unittest.mock import MagicMock

    from robin.reporting.plotting import _add_panel_coverage_points

    ax = MagicMock()
    _add_panel_coverage_points(
        ax,
        [
            {
                "mid_mb": 55.0,
                "coverage_val": 120.0,
                "y_norm": 0.5,
                "offscale": False,
                "baseline_y": 0.0,
                "direction": "gain",
                "label": "EGFR",
            }
        ],
        159.0,
        x_key="mid_mb",
        min_x_spacing=1.8,
        y_min=-1.2,
        y_max=1.2,
    )
    markers = [call.kwargs.get("marker") for call in ax.scatter.call_args_list]
    assert markers == ["o"]


def test_multi_interval_gene_keeps_its_most_extreme_value() -> None:
    """A gene with several panel intervals must not lose a deletion to a gain.

    ALK, GNAQ, ABL1, ARID1B, FOXO3 and MGMT all have two or three intervals in
    the rCNS2 panel. The value within an interval is taken by magnitude, so the
    choice between intervals has to be made the same way.
    """
    from robin.reporting.plotting import _collect_panel_gene_points

    values = np.zeros(180)
    values[156] = -1.80          # ARID1B interval 1: deep loss
    values[157] = +0.15          # ARID1B interval 2: mild gain
    panel = pd.DataFrame(
        [
            {"chrom": "chr6", "start_pos": 156_777_967, "end_pos": 156_935_623,
             "gene": "ARID1B"},
            {"chrom": "chr6", "start_pos": 157_036_772, "end_pos": 157_208_027,
             "gene": "ARID1B"},
        ]
    )

    point = _collect_panel_gene_points(
        panel, "chr6", values, 1_000_000, use_max_abs=True
    )[0]
    # The loss still wins over the gain, which is what this guards. The value is
    # no longer the lone bin's own depth: one bin is not enough to carry a gene
    # call (see CNV_GENE_FOCAL_RESCUE_MIN_BINS), so the window is averaged and
    # the deletion is reported shallower than -1.80 while staying well past the
    # cut-off.
    assert point["cnv_val"] < -0.3
    assert point["cnv_val"] > -1.80
    # And it is drawn at the locus the value came from, not the first interval.
    assert 156.7 < point["mid_mb"] < 156.95


def test_multi_interval_gene_collapses_to_one_marker() -> None:
    from robin.reporting.plotting import _collect_panel_gene_points

    values = np.zeros(180)
    panel = pd.DataFrame(
        [
            {"chrom": "chr2", "start_pos": 29_531_853, "end_pos": 29_532_207, "gene": "ALK"},
            {"chrom": "chr2", "start_pos": 29_694_883, "end_pos": 29_717_768, "gene": "ALK"},
            {"chrom": "chr2", "start_pos": 29_919_911, "end_pos": 29_920_790, "gene": "ALK"},
        ]
    )
    points = _collect_panel_gene_points(
        panel, "chr2", values, 1_000_000, use_max_abs=True
    )
    assert [p["label"] for p in points] == ["ALK"]


def test_ploidy_mode_merge_still_takes_the_signed_maximum() -> None:
    """In ploidy mode the value is a copy number, so the largest one wins."""
    from robin.reporting.plotting import _collect_panel_gene_points

    values = np.zeros(180)
    values[156] = 0.4
    values[157] = 3.2
    panel = pd.DataFrame(
        [
            {"chrom": "chr6", "start_pos": 156_777_967, "end_pos": 156_935_623,
             "gene": "ARID1B"},
            {"chrom": "chr6", "start_pos": 157_036_772, "end_pos": 157_208_027,
             "gene": "ARID1B"},
        ]
    )
    point = _collect_panel_gene_points(
        panel, "chr6", values, 1_000_000, use_max_abs=False
    )[0]
    assert point["cnv_val"] == pytest.approx(3.2)


def test_genome_and_per_chromosome_agree_on_every_gene() -> None:
    """Standing guard: the two figures must never disagree about a gene's value.

    The user saw a gene appear at different heights on the genome-wide panel and
    its chromosome page. This pins the two code paths together for a whole panel.
    """
    from robin.reporting.plotting import (
        _collect_chromosome_significant_panel_points,
        _collect_genome_significant_panel_points,
    )

    rng = np.random.default_rng(2)
    lengths = {"chr1": 248, "chr7": 159, "chr9": 138, "chr17": 83}
    log2 = {
        chrom: np.log2(np.abs(rng.normal(2.0, 0.30, n)) / 2.0)
        for chrom, n in lengths.items()
    }

    rows = []
    for chrom, n in lengths.items():
        for i in range(6):
            start = (5 + i * 17) * 1_000_000
            if start // 1_000_000 >= n - 2:
                continue
            # Deliberately spans more than one bin, as real targets do.
            rows.append(
                {
                    "chrom": chrom,
                    "start_pos": start,
                    "end_pos": start + 1_400_000,
                    "gene": f"{chrom[3:]}G{i}",
                }
            )
    panel = pd.DataFrame(rows)
    coverage = pd.DataFrame(
        {
            "chrom": panel["chrom"],
            "startpos": panel["start_pos"],
            "endpos": panel["end_pos"],
            "name": panel["gene"],
            "coverage": [50.0] * len(panel),
        }
    )
    genes = tuple(panel["gene"])

    genome = {
        point["label"]: point["y_norm"]
        for point in _collect_genome_significant_panel_points(
            panel,
            log2,
            list(lengths),
            {chrom: 0.0 for chrom in lengths},
            1_000_000,
            {},
            coverage,
            use_log2=True,
            scale_mean_cnv=0.0,
            sex_estimate="XY",
            configured_genes=genes,
        )
    }
    per_chromosome = {}
    for chrom in lengths:
        for point in _collect_chromosome_significant_panel_points(
            panel,
            chrom,
            np.asarray(log2[chrom]),
            1_000_000,
            [],
            coverage,
            use_log2=True,
            baseline=0.0,
            sex_estimate="XY",
            configured_genes=genes,
        ):
            per_chromosome[point["label"]] = point["y_norm"]

    assert set(genome) == set(genes)
    assert genome == pytest.approx(per_chromosome)


def test_genome_figure_size_is_caller_controlled() -> None:
    """The report sizes the figure to its page, so labels land at true size."""
    from robin.reporting.plotting import build_CNV_genome_figure

    result = _FakeResult(_sample_cnv())
    fig = build_CNV_genome_figure(
        result, {"bin_width": 1_000_000}, fig_width=10.5, fig_height=4.5
    )
    try:
        width, height = fig.get_size_inches()
        assert width == pytest.approx(10.5)
        assert height == pytest.approx(4.5)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_standalone_genome_download_keeps_a_wide_canvas() -> None:
    """The download has no page to fit, so width is free room for labels."""
    from robin.reporting.plotting import (
        CNV_GENOME_FIG_ASPECT,
        CNV_GENOME_FIG_WIDTH,
        build_CNV_genome_figure,
    )

    # Narrowing this to the report's page width (10.5in) crowded the gene
    # labels without making any of them larger — fonts are absolute points.
    # Width is the lever for label crowding, so the download stays generous.
    assert CNV_GENOME_FIG_WIDTH >= 20.0
    # Crowding here is horizontal, so height is not scaled up with the width:
    # the panel stays flat enough that a genome profile does not read as
    # stretched, while leaving room for stacked gene labels.
    height = CNV_GENOME_FIG_WIDTH / CNV_GENOME_FIG_ASPECT
    assert 4.0 <= height <= 6.5

    result = _FakeResult(_sample_cnv())
    fig = build_CNV_genome_figure(result, {"bin_width": 1_000_000})
    try:
        width, height = fig.get_size_inches()
        assert width == pytest.approx(CNV_GENOME_FIG_WIDTH)
        assert width / height == pytest.approx(CNV_GENOME_FIG_ASPECT, rel=1e-3)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_report_places_the_genome_summary_at_its_true_aspect() -> None:
    """Placing a figure at a guessed aspect stretches it; measure it instead."""
    import io

    from robin.reporting.sections.cnv import GENOME_SUMMARY_ASPECT, _image_aspect

    result = _FakeResult(_sample_cnv())
    buf = create_CNV_plot(
        result, {"bin_width": 1_000_000}, fig_width=10.0, fig_height=4.0
    )
    aspect = _image_aspect(buf, GENOME_SUMMARY_ASPECT)
    # Measured from the rendered image, close to but not exactly the requested
    # 4:10 because savefig trims surrounding whitespace.
    assert 0.3 < aspect < 0.55
    # The buffer is left readable for the caller that places the image.
    assert buf.read(2) == b"\xff\xd8"
    assert isinstance(buf, io.BytesIO)

    empty = io.BytesIO(b"not an image")
    assert _image_aspect(empty, GENOME_SUMMARY_ASPECT) == pytest.approx(
        1.0 / GENOME_SUMMARY_ASPECT
    )


def test_report_landscape_frame_is_wider_than_the_portrait_frame() -> None:
    """The genome summary page exists to buy horizontal room; pin the gain."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch

    from robin.reporting.report import RobinReport

    portrait_width = A4[0] - 2 * inch
    landscape_width = (
        landscape(RobinReport.LANDSCAPE_PAGESIZE)[0]
        - 2 * RobinReport.LANDSCAPE_SIDE_MARGIN
    )
    # A4 rotated: the whole report stays one paper size, and the figure still
    # gets 10.5in against the portrait column's 6.3in. A3 reached 15.3in but
    # made the report mixed-format, which is awkward to print.
    assert landscape_width > portrait_width * 1.5
    assert landscape_width / inch > 10.0
    assert RobinReport.LANDSCAPE_PAGESIZE == A4


def test_report_document_builds_mixed_portrait_and_landscape_pages(tmp_path) -> None:
    """Smoke test for the two-template document the genome summary needs."""
    from reportlab.platypus import NextPageTemplate, PageBreak, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from PyPDF2 import PdfReader

    from robin.reporting.report import RobinReport

    out = tmp_path / "sample"
    out.mkdir()
    pdf = tmp_path / "report.pdf"
    report = RobinReport(
        str(pdf), str(out), center="TEST", plotting_preferences={}
    )
    doc = report._create_document()

    body = getSampleStyleSheet()["Normal"]
    doc.multiBuild(
        [
            Paragraph("portrait body", body),
            NextPageTemplate(report.LANDSCAPE_TEMPLATE),
            PageBreak(),
            Paragraph("wide figure would go here", body),
            NextPageTemplate(report.PORTRAIT_TEMPLATE),
            PageBreak(),
            Paragraph("back to portrait", body),
        ]
    )

    pages = PdfReader(str(pdf)).pages
    assert len(pages) == 3
    shapes = [
        "landscape" if p.mediabox.width > p.mediabox.height else "portrait"
        for p in pages
    ]
    assert shapes == ["portrait", "landscape", "portrait"]
    # Only the middle page changes paper size; the report stays A4 elsewhere.
    assert float(pages[1].mediabox.width) > float(pages[0].mediabox.width)
    assert float(pages[2].mediabox.width) == pytest.approx(
        float(pages[0].mediabox.width)
    )
    # The whole point: the landscape frame is wider than the portrait one.
    assert doc.landscape_width > doc.width


def test_genome_panel_fills_the_canvas_at_any_width() -> None:
    """Matplotlib's fractional margins left a wide empty strip on the right."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import (
        CNV_GENOME_MARGIN_LEFT_IN,
        CNV_GENOME_MARGIN_RIGHT_IN,
        build_CNV_genome_figure,
    )

    result = _FakeResult(_sample_cnv())
    for width, height in ((24.0, 6.0), (15.34, 5.9), (10.0, 4.0)):
        fig = build_CNV_genome_figure(
            result, {"bin_width": 1_000_000}, fig_width=width, fig_height=height
        )
        try:
            position = fig.axes[0].get_position()
            panel_width = fig.get_figwidth() * position.width
            reserved = CNV_GENOME_MARGIN_LEFT_IN + CNV_GENOME_MARGIN_RIGHT_IN
            # Axis furniture is reserved in inches, not as a fraction, so the
            # panel keeps essentially all of any extra width it is given.
            assert panel_width == pytest.approx(width - reserved, abs=0.01)
            # Well past matplotlib's default 0.775 of the figure, and the
            # fraction climbs as the figure widens rather than staying fixed.
            assert position.width > 0.85
        finally:
            plt.close(fig)


def test_wider_figure_becomes_a_wider_panel_not_a_wider_margin() -> None:
    """The point of widening the canvas is panel room, not empty space."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import build_CNV_genome_figure

    result = _FakeResult(_sample_cnv())
    widths = {}
    for width in (12.0, 24.0):
        fig = build_CNV_genome_figure(
            result, {"bin_width": 1_000_000}, fig_width=width, fig_height=5.0
        )
        try:
            widths[width] = fig.get_figwidth() * fig.axes[0].get_position().width
        finally:
            plt.close(fig)
    # Doubling the canvas should hand almost all of the extra inches to the
    # panel; under fractional margins it would have kept 22.5% as margin.
    assert widths[24.0] - widths[12.0] == pytest.approx(12.0, abs=0.01)


def test_rotated_gene_label_clears_its_own_marker() -> None:
    """A rotated name centred on its anchor hid the marker it was naming."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import _add_panel_coverage_points

    fig, ax = plt.subplots(figsize=(12.0, 4.0))
    try:
        ax.set_xlim(0, 159)
        ax.set_ylim(-1.5, 1.5)
        points = [
            {
                "label": "EGFR",
                "position_mb": 55.1,
                "y_norm": 1.0,
                "baseline_y": 0.0,
                "direction": "gain",
            }
        ]
        _add_panel_coverage_points(
            ax,
            points,
            159.0,
            x_key="position_mb",
            min_x_spacing=1.0,
            y_min=-1.5,
            y_max=1.5,
            rotated=True,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inverse = ax.transData.inverted()
        label = next(t for t in ax.texts if t.get_text() == "EGFR")
        box = label.get_window_extent(renderer)
        label_bottom = inverse.transform((box.x0, box.y0))[1]
        # With ha="center" the box straddled the anchor and ran down to 0.975,
        # covering a marker drawn at 1.0.
        assert label_bottom > 1.0, label_bottom
    finally:
        plt.close(fig)


def test_gene_marker_is_drawn_large_enough_to_find_in_a_dense_cloud() -> None:
    """The marker is correct but useless if it looks like any other point."""
    from robin.reporting.plotting import (
        CNV_CHROMOSOME_SCATTER_SIZE,
        CNV_PANEL_MARKER_EDGE_WIDTH,
        CNV_PANEL_MARKER_SIZE,
    )

    # Larger than a bin point and ringed in white, but kept small so a row of
    # lollipops does not read as a line of blobs. Below about 9 the marker stops
    # being findable in a densely binned cloud.
    assert CNV_PANEL_MARKER_SIZE > CNV_CHROMOSOME_SCATTER_SIZE
    assert CNV_PANEL_MARKER_SIZE >= 9.0
    assert CNV_PANEL_MARKER_EDGE_WIDTH >= 0.6


#: Analysis bin widths actually observed across archived samples. The width is
#: sized from read depth, so it is not knowable in advance — a display rule tuned
#: at one of these has to hold at all of them.
OBSERVED_ANALYSIS_BIN_WIDTHS = (7_000, 11_000, 60_000, 235_000, 431_000, 1_148_000)


@pytest.mark.parametrize("analysis_bin", OBSERVED_ANALYSIS_BIN_WIDTHS)
def test_genome_cloud_is_not_smoothed_far_past_its_gene_markers(analysis_bin) -> None:
    """Markers come from raw bins, so the cloud must stay comparable to them.

    Run at every analysis bin width seen in real data, because the default
    display width is derived from it. A fixed 400 kb passed this at 50 kb and
    failed badly at 7 kb, where it averaged 57 bins into every drawn point.
    """
    from robin.analysis.cnv_analysis import (
        CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING,
        downsample_cnv_for_plot,
        resolve_cnv_report_genome_plot_bin_width,
    )
    from robin.reporting.plotting import _collect_panel_gene_points

    rng = np.random.default_rng(3)
    chrom = rng.normal(0.75, 0.18, max(159_000_000 // analysis_bin, 32))
    # A focal event one analysis bin wide — the hardest case for a cloud that
    # averages, and the shape a real focal amplification actually has.
    start = min(55_000_000 // analysis_bin, len(chrom) - 1)
    chrom[start] = 1.02

    panel = pd.DataFrame(
        [
            {
                "chrom": "chr7",
                "start_pos": start * analysis_bin,
                "end_pos": (start + 1) * analysis_bin,
                "gene": "EGFR",
            }
        ]
    )
    marker = _collect_panel_gene_points(
        panel, "chr7", chrom, analysis_bin, use_max_abs=True, target_coverage_df=None
    )[0]["cnv_val"]

    display_bin = resolve_cnv_report_genome_plot_bin_width(analysis_bin)
    _, cloud = downsample_cnv_for_plot(chrom, analysis_bin, display_bin)

    # Never average away more than the agreed number of analysis bins. This is
    # the property that keeps the cloud within reach of its markers; the reach
    # itself degrades smoothly with depth of averaging, so bounding the
    # averaging is what can be asserted at every width.
    bins_averaged = max(1, display_bin // analysis_bin)
    assert bins_averaged <= CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING, (
        analysis_bin,
        display_bin,
    )
    # Readability ceiling still respected, and a track already coarser than the
    # ceiling is never smoothed further.
    assert display_bin >= analysis_bin
    assert display_bin <= max(analysis_bin, 400_000)

    # A single-bin event survives averaging well enough to remain visible
    # against the surrounding cloud rather than vanishing into it.
    assert cloud.max() >= marker * 0.75, (cloud.max(), marker, display_bin)
    # The marker is a max over the gene's bins, so it reaches the injected event
    # (it can exceed it when a noisier neighbouring bin falls inside the gene).
    # The marker averages a minimum-width window, so a one-bin event is reported
    # below its own peak by design; it must still carry most of the signal.
    assert marker >= 0.5


def test_genome_display_width_tracks_the_analysis_bin() -> None:
    """The default display width is derived, not fixed — that was the bug."""
    from robin.analysis.cnv_analysis import (
        CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING,
        resolve_cnv_report_genome_plot_bin_width,
    )

    resolve = resolve_cnv_report_genome_plot_bin_width
    # Deep run: fine analysis bins must not be smoothed 57x to hit a fixed 400 kb.
    assert resolve(7_000) == 7_000 * CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING
    assert resolve(11_000) == 11_000 * CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING
    # Mid: the readability ceiling starts to bind.
    assert resolve(235_000) == 400_000
    # Shallow run: already coarser than the ceiling, so leave it alone rather
    # than smoothing a sparse track further.
    assert resolve(431_000) == 431_000
    assert resolve(1_148_000) == 1_148_000
    # Monotonic: a coarser analysis bin never yields a finer display bin.
    widths = [resolve(w) for w in OBSERVED_ANALYSIS_BIN_WIDTHS]
    assert widths == sorted(widths)


def test_genome_figure_honours_a_chosen_y_range_exactly() -> None:
    """A chosen window is used as given; only a fitted one is adjusted."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import build_CNV_genome_figure

    log2 = {
        "chr1": np.random.default_rng(1).normal(0.0, 0.15, 249),
        "chr7": np.random.default_rng(2).normal(0.5, 0.15, 159),
    }
    result = _FakeResult({name: 2.0 * 2**vals for name, vals in log2.items()})
    panel = pd.DataFrame(
        [
            {
                "chrom": "chr7",
                "start_pos": 55_000_000,
                "end_pos": 55_200_000,
                "gene": "EGFR",
            }
        ]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr7"],
            "startpos": [55_000_000],
            "endpos": [55_200_000],
            "name": ["EGFR"],
            "coverage": [60.0],
        }
    )

    for span in (0.6, 1.0, 1.5, 4.0):
        fig = build_CNV_genome_figure(
            result,
            {"bin_width": 1_000_000},
            log2,
            use_normalized_difference=True,
            panel_genes_df=panel,
            target_coverage_df=coverage,
            sex_estimate="XY",
            configured_genes=("EGFR",),
            fixed_axis_log2=span,
        )
        try:
            # Neither tick snapping nor marker expansion may move a chosen
            # window: both used to, giving +-1.0 as +-1.1.
            assert fig.axes[0].get_ylim() == pytest.approx((-span, span))
        finally:
            plt.close(fig)


def test_genome_figure_still_fits_the_data_by_default() -> None:
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import build_CNV_genome_figure

    log2 = {"chr1": np.random.default_rng(1).normal(0.0, 0.15, 249)}
    result = _FakeResult({"chr1": 2.0 * 2 ** log2["chr1"]})
    fig = build_CNV_genome_figure(
        result, {"bin_width": 1_000_000}, log2, use_normalized_difference=True
    )
    try:
        low, high = fig.axes[0].get_ylim()
        # Fitted to a quiet profile, so nowhere near the +-2 chromosome default.
        assert high < 2.0
        assert low == pytest.approx(-high)
    finally:
        plt.close(fig)


def test_chromosome_names_are_anchored_at_the_start_of_each_chromosome() -> None:
    """Centred names sit a long way from either edge on a 24in panel."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import build_CNV_genome_figure

    bin_width = 1_000_000
    # 100 Mb and 60 Mb, so start and centre are far apart.
    log2 = {
        "chr1": np.zeros(100, dtype=float),
        "chr2": np.zeros(60, dtype=float),
    }
    result = _FakeResult({name: np.full(len(v), 2.0) for name, v in log2.items()})
    fig = build_CNV_genome_figure(
        result, {"bin_width": bin_width}, log2, use_normalized_difference=True
    )
    try:
        axis = fig.axes[0]
        positions = {
            text.get_text(): text.get_position()[0]
            for text in axis.texts
            if text.get_text() in {"1", "2"}
        }
        assert set(positions) == {"1", "2"}
        # chr1 spans 0-100 Mb, chr2 100-160 Mb.
        assert positions["1"] < 5_000_000, positions
        assert 100_000_000 <= positions["2"] < 105_000_000, positions
        # Not the old centres (50 Mb and 130 Mb).
        assert abs(positions["1"] - 50_000_000) > 40_000_000
        assert abs(positions["2"] - 130_000_000) > 20_000_000
        # Padded clear of the boundary line rather than sitting on it.
        assert positions["2"] > 100_000_000
        for text in axis.texts:
            if text.get_text() in {"1", "2"}:
                assert text.get_ha() == "left"
    finally:
        plt.close(fig)


def test_contig_and_centromere_guides_are_darker_than_the_grid() -> None:
    """Reviewers asked for the chromosome and p/q splits to read at a glance."""
    from robin.reporting.plotting import (
        CNV_CENTROMERE_WIDTH,
        CNV_COLORS,
        CNV_CONTIG_BOUNDARY_WIDTH,
        MODERN_COLORS,
    )

    def _luma(value: str) -> float:
        value = value.lstrip("#")
        red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        return 0.299 * red + 0.587 * green + 0.114 * blue

    grid = _luma(MODERN_COLORS["grid"])
    assert _luma(CNV_COLORS["contig_boundary"]) < grid
    assert _luma(CNV_COLORS["centromere"]) < grid
    # The boundary is the stronger of the two: it separates chromosomes, while
    # the centromere only splits one.
    assert _luma(CNV_COLORS["contig_boundary"]) < _luma(CNV_COLORS["centromere"])
    assert CNV_CONTIG_BOUNDARY_WIDTH >= CNV_CENTROMERE_WIDTH


def test_centromere_lines_sit_on_the_real_pq_boundary() -> None:
    """The arm divider must be the p/q transition, not a guess at the middle."""
    from robin.analysis.cnv_regional import (
        load_centromere_boundaries,
        load_cytobands_bed,
    )

    bands = load_cytobands_bed()
    assert not bands.empty, "cytoband resource missing"
    boundaries = load_centromere_boundaries()

    for chrom, chrom_bands in bands.groupby("chrom"):
        names = chrom_bands["name"].astype(str)
        p_arm = chrom_bands[names.str.startswith("p")]
        q_arm = chrom_bands[names.str.startswith("q")]
        assert not p_arm.empty and not q_arm.empty, chrom

        last_p = int(p_arm["end_pos"].max())
        first_q = int(q_arm["start_pos"].min())
        # The arms meet exactly, with no gap to place the line in.
        assert last_p == first_q, (chrom, last_p, first_q)
        assert boundaries[str(chrom)] == first_q, chrom
        # And they meet inside the centromere, not at some arbitrary band.
        assert p_arm.loc[p_arm["end_pos"].idxmax(), "stain"] == "acen", chrom
        assert q_arm.loc[q_arm["start_pos"].idxmin(), "stain"] == "acen", chrom


def test_centromere_boundaries_match_known_hg38_positions() -> None:
    """Spot-check against hg38, including the acrocentric chromosomes."""
    from robin.analysis.cnv_regional import load_centromere_boundaries

    boundaries = load_centromere_boundaries()
    # hg38 p/q boundaries. chr1 distinguishes hg38 (123.4 Mb) from hg19 (125.0).
    expected = {
        "chr1": 123_400_000,
        "chr2": 93_900_000,
        "chr7": 60_100_000,
        "chr17": 25_100_000,
        # Acrocentric: a tiny p arm, so a midpoint guess would be badly wrong.
        "chr13": 17_700_000,
        "chr14": 17_200_000,
        "chr15": 19_000_000,
        "chr21": 12_000_000,
        "chr22": 15_000_000,
        "chrX": 61_000_000,
        "chrY": 10_400_000,
    }
    for chrom, position in expected.items():
        assert boundaries[chrom] == position, (chrom, boundaries[chrom], position)

    # The acrocentrics must land in the first quarter of the chromosome; drawing
    # them near the middle would misrepresent every 13q/14q/21q call.
    lengths = {"chr13": 114_364_328, "chr14": 107_043_718, "chr21": 46_709_983}
    for chrom, length in lengths.items():
        assert boundaries[chrom] / length < 0.30, chrom


def test_cnv_load_uses_the_calling_thresholds() -> None:
    """Load must agree with the cut-off drawn on the plots, not a data-adaptive rule."""
    from robin.analysis.cnv_regional import compute_cnv_load

    bin_width = 1_000_000
    # 100 Mb: 20 Mb gained, 10 Mb lost, 5 Mb with no data, rest quiet.
    values = np.zeros(100)
    values[:20] = 0.6
    values[20:30] = -0.8
    values[95:] = np.nan

    load = compute_cnv_load({"chr1": values}, bin_width, "XY")
    assert load["assessed_mb"] == pytest.approx(95.0)
    assert load["gain_mb"] == pytest.approx(20.0)
    assert load["loss_mb"] == pytest.approx(10.0)
    assert load["total_mb"] == pytest.approx(30.0)
    # Percentages are over the assessed span, not the nominal chromosome length,
    # so partial coverage does not silently dilute the load toward zero.
    assert load["gain_percent"] == pytest.approx(20 / 95 * 100)
    assert load["total_percent"] == pytest.approx(30 / 95 * 100)


def test_cnv_load_is_zero_for_a_genome_inside_the_calling_window() -> None:
    """The regional cytoband table calls events here; the load must not."""
    from robin.analysis.cnv_regional import compute_cnv_load

    quiet = {"chr1": np.full(100, -0.10), "chr2": np.full(100, 0.12)}
    load = compute_cnv_load(quiet, 1_000_000, "XY")
    assert load["total_mb"] == 0.0
    assert load["total_percent"] == 0.0
    assert load["assessed_mb"] == pytest.approx(200.0)


def test_cnv_load_ignores_non_reportable_contigs_and_empty_input() -> None:
    from robin.analysis.cnv_regional import compute_cnv_load

    mixed = {
        "chr1": np.full(10, 0.6),
        "chrUn_KI270302v1": np.full(1000, 0.6),
    }
    load = compute_cnv_load(mixed, 1_000_000, "XY")
    assert load["assessed_mb"] == pytest.approx(10.0)
    assert load["gain_mb"] == pytest.approx(10.0)

    empty = compute_cnv_load({}, 1_000_000, "XY")
    assert empty["assessed_mb"] == 0.0
    assert empty["total_percent"] == 0.0


def test_clinical_trial_legend_can_avoid_the_x_axis_label() -> None:
    """On a 2in-tall report panel the bottom edge already carries the x-label."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import _add_clinical_trial_legend

    points = [{"clinical_trial": True}]
    for corner, expect_high in (("bottom", False), ("top", True)):
        fig = plt.figure(figsize=(6.1, 2.23))
        try:
            assert _add_clinical_trial_legend(fig, points, corner=corner) is True
            text = next(t for t in fig.texts if "purple" in t.get_text())
            assert (text.get_position()[1] > 0.5) is expect_high
        finally:
            plt.close(fig)

    # Still suppressed entirely when no trial gene is on the figure.
    fig = plt.figure(figsize=(6.1, 2.23))
    try:
        assert _add_clinical_trial_legend(fig, [{"clinical_trial": False}]) is False
        assert not fig.texts
    finally:
        plt.close(fig)


def _regional_states(track: np.ndarray, bin_width: int = 1_000_000):
    from robin.analysis.cnv_regional import (
        analyze_cytoband_cnv,
        load_centromere_bed,
        load_cytobands_bed,
        load_gene_bed,
    )

    frame = analyze_cytoband_cnv(
        {"chr1": track},
        "chr1",
        {"bin_width": bin_width},
        load_cytobands_bed(),
        load_centromere_bed(),
        load_gene_bed(),
        "XY",
    )
    if frame.empty:
        return frame
    # The regional frame reports only non-normal regions, in ``cnv_state``.
    assert "cnv_state" in frame.columns, list(frame.columns)
    return frame[frame["cnv_state"].isin(["GAIN", "LOSS"])]


def test_regional_events_are_gated_on_the_calling_cutoff() -> None:
    """The adaptive rule called events on a chromosome with nothing on it."""
    # Flat at log2 -0.10: entirely inside the +-0.3 calling window. Before the
    # gate this returned three GAIN/LOSS regions, because the adaptive rule
    # measures against the chromosome's own mean and SD.
    flat = np.full(249, -0.10) + np.random.default_rng(0).normal(0, 0.05, 249)
    assert len(_regional_states(flat)) == 0


def test_regional_events_still_report_real_changes() -> None:
    """Gating must not cost sensitivity to genuine events.

    A region is reported in pieces where an unmappable block interrupts it. On
    chr1 that is the centromere plus the 1q12 heterochromatic block, together
    about 21 Mb of sequence with no uniquely mappable content — a call must not
    claim to span it, so an event crossing it comes back as two rows.
    """
    loss = np.zeros(249)
    loss[10:60] = -1.2
    called = _regional_states(loss)
    # 10-60 Mb is entirely within 1p, so it is uninterrupted.
    assert len(called) == 1
    assert float(called["mean_cnv"].iloc[0]) < -0.3

    gain = np.zeros(249)
    gain[100:180] = 0.75
    called = _regional_states(gain)
    assert len(called) == 2, called[["start_pos", "end_pos", "cnv_state"]]
    assert set(called["cnv_state"]) == {"GAIN"}
    assert (called["mean_cnv"] > 0.3).all()
    # The pieces sit either side of the unmappable block, and none spans it.
    assert called["start_pos"].min() <= 100_000_000
    assert called["end_pos"].max() >= 179_000_000
    gap_start = int(called.sort_values("start_pos")["end_pos"].iloc[0])
    gap_end = int(called.sort_values("start_pos")["start_pos"].iloc[1])
    assert gap_end > gap_start, (gap_start, gap_end)

    both = np.zeros(249)
    both[10:60] = -1.2
    both[100:180] = 0.75
    called = _regional_states(both)
    assert set(called["cnv_state"]) == {"GAIN", "LOSS"}
    assert (called["cnv_state"] == "LOSS").sum() == 1
    assert (called["cnv_state"] == "GAIN").sum() == 2


def test_unmappable_bands_are_never_called_as_losses() -> None:
    """Centromeric, heterochromatic and stalk bands are not copy number.

    Read depth there reflects mappability. On the log2 track they sit at -3 to
    -5, and before they were excluded every one of the top ten regional events
    on a real sample was one of them — and the same fourteen recurred in 8 of 8
    samples, which is the signature of an artefact rather than biology.
    """
    from robin.analysis.cnv_regional import UNMAPPABLE_CYTOBAND_STAINS

    from robin.analysis.cnv_regional import load_cytobands_bed

    assert UNMAPPABLE_CYTOBAND_STAINS == {"acen", "gvar", "stalk"}

    bands = load_cytobands_bed()
    chr1 = bands[bands["chrom"] == "chr1"]
    unmappable = chr1[chr1["stain"].isin(UNMAPPABLE_CYTOBAND_STAINS)]
    assert not unmappable.empty, "chr1 must have acen/gvar bands to exercise this"

    # Deeply negative across unmappable sequence only. Bins are 1 Mb, so only
    # whole bins strictly inside a band are set — a partially covered bin at the
    # edge belongs to the neighbouring mappable band and would contaminate it.
    track = np.zeros(249)
    for _, band in unmappable.iterrows():
        lo = -(-int(band["start_pos"]) // 1_000_000)  # ceil
        hi = int(band["end_pos"]) // 1_000_000  # floor
        track[lo:hi] = -4.5

    assert (track < 0).any(), "test set nothing"
    called = _regional_states(track)
    assert called.empty, called[["start_pos", "end_pos", "mean_cnv"]]


def test_every_regional_call_crosses_the_calling_threshold() -> None:
    """No row may appear whose own mean sits inside the calling window."""
    from robin.classification_config import get_cnv_thresholds

    gain_threshold, loss_threshold = get_cnv_thresholds("chr1", "XY")
    rng = np.random.default_rng(11)
    track = rng.normal(0.0, 0.25, 249)
    track[30:70] = -1.4
    track[150:200] = 0.9

    for _, row in _regional_states(track).iterrows():
        mean_cnv = float(row["mean_cnv"])
        if row["cnv_state"] == "GAIN":
            assert mean_cnv >= gain_threshold, row.to_dict()
        else:
            assert mean_cnv <= loss_threshold, row.to_dict()


def test_cnv_load_summary_text_is_shared_and_degrades_safely() -> None:
    from robin.analysis.cnv_regional import CNV_LOAD_TITLE, cnv_load_summary_text

    text = cnv_load_summary_text(
        {
            "assessed_mb": 417.0,
            "gain_mb": 13.55,
            "loss_mb": 125.85,
            "total_mb": 139.4,
            "gain_percent": 3.25,
            "loss_percent": 30.18,
            "total_percent": 33.43,
        }
    )
    assert CNV_LOAD_TITLE in text
    assert "33.4%" in text and "139 Mb" in text
    assert "417 Mb assessed" in text
    # Nothing to say when there is no data — callers stamp this straight onto a
    # figure, so it must not produce a stray label.
    assert cnv_load_summary_text(None) == ""
    assert cnv_load_summary_text({"assessed_mb": 0.0}) == ""


def test_cnv_load_follows_the_cutoff_override() -> None:
    """Load must agree with the cut-off lines actually drawn on the plots."""
    from robin.analysis.cnv_regional import compute_cnv_load

    # 60 Mb of low-level change at log2 +0.20: real, but under the +-0.3
    # calling cut-off. This is the low-purity case the Cut-off menu exists for.
    track = {"chr1": np.concatenate([np.full(60, 0.20), np.full(40, 0.0)])}

    default = compute_cnv_load(track, 1_000_000, "XY")
    assert default["total_percent"] == pytest.approx(0.0)
    assert default["cutoff"] is None
    assert "calling" in default["cutoff_label"]

    lowered = compute_cnv_load(track, 1_000_000, "XY", cutoff_override=0.15)
    assert lowered["total_percent"] == pytest.approx(60.0)
    assert lowered["gain_mb"] == pytest.approx(60.0)
    assert lowered["cutoff"] == pytest.approx(0.15)
    assert lowered["cutoff_label"] == "±0.15, non-default"

    # A cut-off above the change hides it again.
    raised = compute_cnv_load(track, 1_000_000, "XY", cutoff_override=0.5)
    assert raised["total_percent"] == pytest.approx(0.0)

    # Sign is irrelevant — the menu offers a symmetric cut-off.
    assert compute_cnv_load(track, 1_000_000, "XY", cutoff_override=-0.15)[
        "total_percent"
    ] == pytest.approx(60.0)


def test_cnv_load_always_names_the_cutoff_it_used() -> None:
    """A load figure is not comparable between different cut-offs."""
    from robin.analysis.cnv_regional import compute_cnv_load, cnv_load_summary_text

    track = {"chr1": np.full(100, 0.5)}
    for override, expected in ((None, "calling"), (0.15, "±0.15")):
        load = compute_cnv_load(track, 1_000_000, "XY", cutoff_override=override)
        assert expected in load["cutoff_label"]
        assert f"cut-off {load['cutoff_label']}" in cnv_load_summary_text(load)


def test_cnv_load_ignores_a_nonsense_cutoff_override() -> None:
    from robin.analysis.cnv_regional import compute_cnv_load

    track = {"chr1": np.full(100, 0.5)}
    baseline = compute_cnv_load(track, 1_000_000, "XY")["total_percent"]
    for bad in ("not a number", float("nan"), float("inf"), None):
        load = compute_cnv_load(track, 1_000_000, "XY", cutoff_override=bad)
        assert load["total_percent"] == pytest.approx(baseline), bad


def _genome_with_chr10(level: float, noise: float, n_other_aberrant: int):
    lengths = {
        "chr1": 249, "chr2": 242, "chr7": 159, "chr10": 134,
        "chr12": 133, "chr16": 90, "chr17": 83,
    }
    other = {"chr7": 0.66, "chr12": 0.33, "chr16": -0.64, "chr17": -0.30}
    picked = dict(list(other.items())[:n_other_aberrant])
    rng = np.random.default_rng(7)
    return {
        name: rng.normal(level if name == "chr10" else picked.get(name, 0.0), noise, n)
        for name, n in lengths.items()
    }


def _regional_whole_chromosome_called(data, chromosome="chr10") -> bool:
    from robin.analysis.cnv_regional import (
        analyze_cytoband_cnv,
        load_centromere_bed,
        load_cytobands_bed,
        load_gene_bed,
    )

    frame = analyze_cytoband_cnv(
        data,
        chromosome,
        {"bin_width": 1_000_000},
        load_cytobands_bed(),
        load_centromere_bed(),
        load_gene_bed(),
        "XY",
    )
    if frame.empty:
        return False
    called = frame[frame["cnv_state"].isin(["GAIN", "LOSS"])]
    return any("WHOLE" in str(name).upper() for name in called.get("name", []))


def test_whole_chromosome_call_does_not_depend_on_the_rest_of_the_genome() -> None:
    """The old rule lost the call as the genome became more aberrant."""
    # A chromosome uniformly gained at log2 +0.55 is a whole-chromosome gain
    # whatever else is going on. Under the previous rule — chromosome mean
    # against the spread of all chromosome means — this was reported on a quiet
    # genome and dropped once four other chromosomes were aberrant, because
    # their spread raised the bar.
    for noise in (0.15, 0.30):
        quiet = _regional_whole_chromosome_called(
            _genome_with_chr10(0.55, noise, n_other_aberrant=0)
        )
        aberrant = _regional_whole_chromosome_called(
            _genome_with_chr10(0.55, noise, n_other_aberrant=4)
        )
        assert quiet is True, noise
        assert aberrant == quiet, noise


def test_regional_and_arm_tables_agree_on_whole_chromosome_events() -> None:
    """Two detectors reporting different things on one page is indefensible."""
    from robin.analysis.cnv_classification import detect_cnv_events
    from robin.analysis.cnv_regional import load_cytobands_bed, load_gene_bed

    cytobands, gene_bed = load_cytobands_bed(), load_gene_bed()
    for level in (0.25, 0.32, 0.40, 0.55, 0.80):
        for noise in (0.15, 0.30):
            data = _genome_with_chr10(level, noise, n_other_aberrant=4)
            events = detect_cnv_events(data, 1_000_000, "XY", cytobands, gene_bed)
            from_detector = any(
                event.chromosome == "chr10" and "WHOLE" in event.event_type
                for event in events
            )
            from_regional = _regional_whole_chromosome_called(data)
            assert from_detector == from_regional, (level, noise)


def test_a_quiet_chromosome_is_never_a_whole_chromosome_event() -> None:
    for noise in (0.15, 0.30):
        assert not _regional_whole_chromosome_called(
            _genome_with_chr10(0.0, noise, n_other_aberrant=4)
        )
        # And a level under the calling cut-off does not qualify either.
        assert not _regional_whole_chromosome_called(
            _genome_with_chr10(0.25, noise, n_other_aberrant=0)
        )


def _ngtd_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"chrom": "chr7", "start_pos": 55_000_000, "end_pos": 55_300_000, "gene": "EGFR"},
            {"chrom": "chr9", "start_pos": 21_800_000, "end_pos": 22_000_000, "gene": "MTAP"},
            {"chrom": "chr17", "start_pos": 39_700_000, "end_pos": 39_800_000, "gene": "ERBB2"},
            # On the panel but on a contig with no CNV data in the fixture.
            {"chrom": "chr4", "start_pos": 1_800_000, "end_pos": 1_810_000, "gene": "FGFR3"},
        ]
    )


def _ngtd_track(bin_width: int = 50_000) -> dict:
    lengths = {"chr7": 159, "chr9": 138, "chr17": 83}
    track = {
        name: np.zeros(mb * 1_000_000 // bin_width) for name, mb in lengths.items()
    }
    for chrom, mb, value in (("chr7", 55.0, 1.10), ("chr9", 21.9, -1.40)):
        start = int(mb * 1e6) // bin_width
        track[chrom][start : start + 8] = value
    return track


def test_ngtd_table_reports_every_target_including_quiet_ones() -> None:
    """A target with nothing on it must still say it was looked at."""
    from robin.analysis.cnv_regional import (
        CNV_NO_EVENT_LABEL,
        compute_target_gene_cnv_states,
    )

    requested = ["EGFR", "MTAP", "ERBB2", "FGFR3", "MADEUPGENE"]
    rows = compute_target_gene_cnv_states(
        _ngtd_track(), 50_000, requested, "XY", gene_frames=[_ngtd_panel()]
    )
    # Every requested gene comes back, in the order asked for.
    assert [row["gene"] for row in rows] == requested

    states = {row["gene"]: row["state"] for row in rows}
    assert states["EGFR"] == "GAIN"
    assert states["MTAP"] == "LOSS"
    # On the panel, covered, nothing past the cut-off.
    assert states["ERBB2"] == CNV_NO_EVENT_LABEL


def test_ngtd_table_never_calls_an_unassessed_gene_negative() -> None:
    """'Not assessed' and 'assessed, nothing found' are different statements."""
    from robin.analysis.cnv_regional import (
        CNV_GENE_NO_DATA_LABEL,
        CNV_GENE_UNKNOWN_LABEL,
        CNV_NO_EVENT_LABEL,
        compute_target_gene_cnv_states,
    )

    rows = {
        row["gene"]: row
        for row in compute_target_gene_cnv_states(
            _ngtd_track(),
            50_000,
            ["FGFR3", "MADEUPGENE"],
            "XY",
            gene_frames=[_ngtd_panel()],
        )
    }
    # On the panel but no bins cover it — must not read as "No CNVs Detected".
    assert rows["FGFR3"]["state"] == CNV_GENE_NO_DATA_LABEL
    assert rows["FGFR3"]["chrom"] == "chr4"
    assert rows["FGFR3"]["state"] != CNV_NO_EVENT_LABEL
    # Not in the panel or any reference — never assessed, no value invented.
    assert rows["MADEUPGENE"]["state"] == CNV_GENE_UNKNOWN_LABEL
    assert rows["MADEUPGENE"]["value"] is None
    assert rows["MADEUPGENE"]["state"] != CNV_NO_EVENT_LABEL


def test_ngtd_table_agrees_with_the_cutoff_in_force() -> None:
    """The table must not contradict the cut-off lines on the plots."""
    from robin.analysis.cnv_regional import (
        CNV_NO_EVENT_LABEL,
        compute_target_gene_cnv_states,
    )

    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 55_000_000, "end_pos": 55_300_000, "gene": "EGFR"}]
    )
    track = {"chr7": np.zeros(159_000_000 // 50_000)}
    start = 55_000_000 // 50_000
    track["chr7"][start : start + 8] = 0.20  # real, but under +-0.3

    default = compute_target_gene_cnv_states(
        track, 50_000, ["EGFR"], "XY", gene_frames=[panel]
    )[0]
    assert default["state"] == CNV_NO_EVENT_LABEL

    lowered = compute_target_gene_cnv_states(
        track, 50_000, ["EGFR"], "XY", gene_frames=[panel], cutoff_override=0.15
    )[0]
    assert lowered["state"] == "GAIN"


def test_ngtd_gene_list_is_configurable() -> None:
    from robin.workflow_config import DEFAULT_CNV_NGTD_GENES, get_cnv_ngtd_genes

    # The supplied NGTD list, kept in the order given. 48 gene names, with the
    # three overlapping antisense transcripts joined to their partner locus.
    assert len(DEFAULT_CNV_NGTD_GENES) == 45
    assert DEFAULT_CNV_NGTD_GENES[0] == "ALK"
    assert DEFAULT_CNV_NGTD_GENES[-1] == "YWHAE"
    assert len(set(DEFAULT_CNV_NGTD_GENES)) == len(DEFAULT_CNV_NGTD_GENES)
    assert "CDKN2B/CDKN2B-AS1" in DEFAULT_CNV_NGTD_GENES
    assert "MYB/MYB-AS1" in DEFAULT_CNV_NGTD_GENES
    assert "MYCN/MYCNOS" in DEFAULT_CNV_NGTD_GENES
    # The partners are not also listed on their own.
    assert "CDKN2B" not in DEFAULT_CNV_NGTD_GENES
    assert "MYB" not in DEFAULT_CNV_NGTD_GENES
    assert "MYCN" not in DEFAULT_CNV_NGTD_GENES
    names = {n for entry in DEFAULT_CNV_NGTD_GENES for n in entry.split("/")}
    assert len(names) == 48
    assert get_cnv_ngtd_genes({}) == DEFAULT_CNV_NGTD_GENES
    assert get_cnv_ngtd_genes({"cnv": {"ngtd_genes": ["EGFR", "MTAP"]}}) == (
        "EGFR",
        "MTAP",
    )
    # An explicitly empty list suppresses the table rather than falling back.
    assert get_cnv_ngtd_genes({"cnv": {"ngtd_genes": []}}) == ()


def test_offpanel_and_unlocatable_targets_are_reported_differently() -> None:
    """'Not a target here' and 'cannot be placed' say different things."""
    from robin.analysis.cnv_regional import (
        CNV_GENE_NOT_LOCATED_LABEL,
        CNV_GENE_UNKNOWN_LABEL,
        compute_target_gene_cnv_states,
    )

    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 55_000_000, "end_pos": 55_300_000, "gene": "EGFR"}]
    )
    reference = pd.DataFrame(
        [{"chrom": "chr9", "start_pos": 21_800_000, "end_pos": 22_000_000, "gene": "MTAP"}]
    )
    track = {"chr7": np.zeros(159_000_000 // 50_000), "chr9": np.zeros(138_000_000 // 50_000)}
    track["chr9"][21_900_000 // 50_000 : 21_900_000 // 50_000 + 8] = -1.4

    rows = {
        row["gene"]: row
        for row in compute_target_gene_cnv_states(
            track, 50_000, ["EGFR", "MTAP", "CDKN2B-AS1"], "XY",
            gene_frames=[panel, reference],
        )
    }
    assert rows["EGFR"]["on_panel"] is True

    # Placed by the reference but not a panel target: no state is invented for
    # it even though the track carries a large deflection there.
    assert rows["MTAP"]["state"] == CNV_GENE_NOT_LOCATED_LABEL
    assert rows["MTAP"]["chrom"] == "chr9"
    assert rows["MTAP"]["value"] is None
    assert rows["MTAP"]["on_panel"] is False

    # Not placeable at all — says nothing about the locus either way.
    assert rows["CDKN2B-AS1"]["state"] == CNV_GENE_UNKNOWN_LABEL
    assert rows["CDKN2B-AS1"]["chrom"] is None


def test_combined_locus_label_resolves_through_its_alternatives() -> None:
    """Overlapping transcripts share CNV bins, so they get one row."""
    from robin.analysis.cnv_regional import compute_target_gene_cnv_states

    # Only the protein-coding partner is on the panel; the antisense transcript
    # is in no reference at all, which is the real CDKN2B-AS1 situation.
    panel = pd.DataFrame(
        [{"chrom": "chr9", "start_pos": 22_002_903, "end_pos": 22_009_305, "gene": "CDKN2B"}]
    )
    track = {"chr9": np.zeros(138_000_000 // 50_000)}
    start = 22_002_903 // 50_000
    track["chr9"][start : start + 4] = -1.45

    row = compute_target_gene_cnv_states(
        track, 50_000, ["CDKN2B/CDKN2B-AS1"], "XY", gene_frames=[panel]
    )[0]
    # The combined label is preserved for display, and it resolves via the
    # partner rather than reporting as unplaceable.
    assert row["gene"] == "CDKN2B/CDKN2B-AS1"
    assert row["state"] == "LOSS"
    assert row["chrom"] == "chr9"
    assert row["on_panel"] is True

    # Order within the label does not matter — whichever side is found wins.
    reversed_label = compute_target_gene_cnv_states(
        track, 50_000, ["CDKN2B-AS1/CDKN2B"], "XY", gene_frames=[panel]
    )[0]
    assert reversed_label["state"] == "LOSS"


def test_every_supplied_ngtd_target_resolves_on_the_cns_panel() -> None:
    """The shipped list must not produce blank rows on the panel in use."""
    import os

    from robin import resources
    from robin.analysis.cnv_regional import (
        CNV_GENE_NOT_LOCATED_LABEL,
        CNV_GENE_UNKNOWN_LABEL,
        compute_target_gene_cnv_states,
    )
    from robin.workflow_config import DEFAULT_CNV_NGTD_GENES

    bed = os.path.join(
        os.path.dirname(os.path.abspath(resources.__file__)),
        "CNS_TD_v1_Jul26_panel_name_uniq.bed",
    )
    if not os.path.exists(bed):
        pytest.skip("CNS_TD panel BED not packaged")
    panel = pd.read_csv(
        bed, sep="\t", header=None, names=["chrom", "start_pos", "end_pos", "gene"]
    )
    lengths = {str(c): 250 for c in panel["chrom"].unique()}
    track = {name: np.zeros(mb * 1_000_000 // 50_000) for name, mb in lengths.items()}

    rows = compute_target_gene_cnv_states(
        track, 50_000, DEFAULT_CNV_NGTD_GENES, "XY", gene_frames=[panel]
    )
    unresolved = [
        row["gene"]
        for row in rows
        if row["state"] in (CNV_GENE_UNKNOWN_LABEL, CNV_GENE_NOT_LOCATED_LABEL)
    ]
    assert unresolved == [], unresolved
    assert len(rows) == len(DEFAULT_CNV_NGTD_GENES)


def _report_chromosome_figure(**kwargs):
    """One per-chromosome figure at the size the PDF report uses."""
    from robin.reporting.plotting import iter_CNV_chromosome_figures

    bin_width = 50_000
    rng = np.random.default_rng(6)
    values = rng.normal(0.0, 0.18, 138_000_000 // bin_width)
    values[430:470] = -2.9  # homozygous-depth loss
    log2 = {"chr9": values}
    result = _FakeResult({"chr9": 2.0 * 2**values})
    panel = pd.DataFrame(
        [{"chrom": "chr9", "start_pos": 21_900_000, "end_pos": 22_100_000, "gene": "CDKN2A"}]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr9"], "startpos": [21_900_000], "endpos": [22_100_000],
            "name": ["CDKN2A"], "coverage": [60.0],
        }
    )
    defaults = dict(
        chromosomes=["chr9"],
        panel_genes_df=panel,
        target_coverage_df=coverage,
        chromosome_status={"chr9": "LOSS: chr9 p21.3-p21.3"},
        use_log2_ratio=True,
        sex_estimate="XY",
        configured_genes=("CDKN2A",),
        clinical_trial_genes=("CDKN2A",),
        full_range_pages=False,
        fig_width=6.10,
        fig_height=2.23,
    )
    defaults.update(kwargs)
    figures = list(
        iter_CNV_chromosome_figures(
            result, {"bin_width": bin_width}, normalized_cnv=log2, **defaults
        )
    )
    return figures, values


def test_report_chromosome_axis_reaches_the_deepest_bin() -> None:
    """A percentile fit clipped a homozygous deletion to an edge marker."""
    import matplotlib.pyplot as plt

    figures, values = _report_chromosome_figure(
        fixed_axis_log2=None, full_range_axis=True
    )
    try:
        low, high = figures[0][1].axes[0].get_ylim()
        finite = values[np.isfinite(values)]
        assert low <= finite.min(), (low, finite.min())
        assert high >= finite.max(), (high, finite.max())
        # Well past what a percentile fit or the +-2 window would have shown.
        assert low < -2.9
    finally:
        for _contig, fig in figures:
            plt.close(fig)


def test_report_chromosome_figures_can_omit_the_trial_legend() -> None:
    """Four plots to a page: the note belongs in the body, not on each figure."""
    import matplotlib.pyplot as plt

    for figure_legend, expected in ((True, 1), (False, 0)):
        figures, _ = _report_chromosome_figure(
            fixed_axis_log2=None, full_range_axis=True, figure_legend=figure_legend
        )
        try:
            fig = figures[0][1]
            legends = [t for t in fig.texts if "purple" in t.get_text()]
            assert len(legends) == expected, figure_legend
        finally:
            for _contig, closing in figures:
                plt.close(closing)


def test_full_range_axis_page_is_not_labelled_as_a_second_page() -> None:
    """It is the report's only page for that chromosome, so no suffix."""
    import matplotlib.pyplot as plt

    figures, _ = _report_chromosome_figure(
        fixed_axis_log2=None, full_range_axis=True
    )
    try:
        titles = [t.get_text() for t in figures[0][1].texts]
        assert any(t.strip() == "Chromosome 9" for t in titles), titles
        assert not any("full range" in t for t in titles), titles
    finally:
        for _contig, fig in figures:
            plt.close(fig)


def test_step2_naming_replaces_clinical_trial_in_user_facing_text() -> None:
    """The site renamed these targets to "Step 2" to avoid confusion with WGS trials.

    Only the wording changed: the ``[cnv].clinical_trial_genes`` config key stays
    as it is, so existing workflow settings keep working.
    """
    from robin.gui.plotting_preferences import CNV_STEP2_LEGEND
    from robin.reporting.sections.cnv import CNVSection

    assert "Step 2" in CNV_STEP2_LEGEND
    assert "clinical trial" not in CNV_STEP2_LEGEND.casefold()
    assert "Step 2" in CNVSection.NGTD_TABLE_TITLE
    assert "clinical trial" not in CNVSection.NGTD_TABLE_TITLE.casefold()
