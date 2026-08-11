"""Live CNV chart layout: mirrored axis, label gutters, trend and reference lines."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robin.gui.components.cnv import (
    _apply_cnv_scatter_performance,
    _cnv_diff_y_window,
    _cnv_echarts_option_to_json,
    _cnv_reference_line_series,
    _cnv_scatter_value_range,
    _cnv_trend_series,
    _plan_lollipop_layout,
    _prepare_lollipop_points,
    _upsert_cnv_reference_lines,
    _upsert_configured_gene_coverage_lollipops,
)


class _FakeChart:
    """Stands in for a NiceGUI echart: the option dict is all the code touches."""

    def __init__(self) -> None:
        self.options = {
            "grid": {"left": "5%", "right": "5%", "containLabel": True},
            "xAxis": {"type": "value", "min": 0},
            "yAxis": [
                {"type": "value", "name": "Log2 ratio (ploidy / expected)"},
                {"type": "value", "name": "Coverage (x)", "position": "right",
                 "show": False, "min": 0},
            ],
            "dataZoom": [
                {"type": "slider", "xAxisIndex": [0]},
                {"type": "slider", "yAxisIndex": [0], "startValue": -4, "endValue": 4},
            ],
            "series": [],
        }


def _scatter(name="chr7", n=300, level=0.0):
    rng = np.random.default_rng(4)
    xs = np.arange(n) * 1e6
    ys = rng.normal(level, 0.12, n)
    return {
        "type": "scatter",
        "name": name,
        "data": [[float(x), float(y)] for x, y in zip(xs, ys)],
    }


def _gene(name, x, y, direction):
    return {
        "gene": name,
        "x": x,
        "coverage": 50.0,
        "y": y,
        "baseline_y": 0.0,
        "direction": direction,
    }


def _series_by_name(chart, name):
    return next(s for s in chart.options["series"] if s.get("name") == name)


def test_value_range_ignores_overlay_series() -> None:
    series = [
        _scatter(),
        {"type": "scatter", "name": "cytobands_highlight", "data": [[0.0, 99.0]]},
    ]
    lo, hi = _cnv_scatter_value_range(series)
    assert -1.0 < lo < hi < 1.0


def test_value_range_is_robust_to_a_single_extreme_bin() -> None:
    series = _scatter()
    series["data"][0][1] = -20.0
    lo, _hi = _cnv_scatter_value_range([series])
    assert lo > -1.0


def test_mirror_axis_matches_the_primary_axis() -> None:
    chart = _FakeChart()
    chart.options["series"] = [_scatter()]
    _upsert_configured_gene_coverage_lollipops(
        chart, [], use_log=True, dark=False, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3),
    )
    primary, mirror = chart.options["yAxis"][0], chart.options["yAxis"][1]
    assert mirror["show"] is True
    assert mirror["position"] == "right"
    assert mirror["min"] == primary["min"]
    assert mirror["max"] == primary["max"]
    assert mirror["interval"] == primary["interval"]
    # Only the primary axis draws gridlines, so they are not doubled up.
    assert mirror["splitLine"] == {"show": False}


def test_axis_window_tightens_around_a_quiet_profile() -> None:
    chart = _FakeChart()
    chart.options["series"] = [_scatter()]
    _upsert_configured_gene_coverage_lollipops(
        chart, [], use_log=True, dark=False, scale_mean_cnv=0.0,
        data_range=(-0.25, 0.25),
    )
    axis = chart.options["yAxis"][0]
    assert axis["max"] - axis["min"] <= 1.5
    assert axis["interval"] <= 0.25
    assert axis["minorTick"]["show"] is True


def test_y_slider_follows_the_pinned_window() -> None:
    chart = _FakeChart()
    chart.options["series"] = [_scatter()]
    _upsert_configured_gene_coverage_lollipops(
        chart, [], use_log=True, dark=False, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3),
    )
    axis = chart.options["yAxis"][0]
    slider = chart.options["dataZoom"][1]
    assert slider["startValue"] == axis["min"]
    assert slider["endValue"] == axis["max"]


def test_gene_labels_ride_on_their_marker() -> None:
    """Rotated names are anchored to the marker, not parked in a reserved band."""
    points = [_gene("EGFR", 2.0e8, 0.9, "gain"), _gene("CDKN2A", 2.02e8, -0.8, "loss")]
    prepared = _prepare_lollipop_points(points, use_log=True, scale_mean_cnv=0.0)
    layout = _plan_lollipop_layout(
        prepared, use_log=True, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3),
    )
    for marker, placement in zip(layout["points"], layout["placements"]):
        assert placement["y"] == pytest.approx(marker["y_disp"])
        assert layout["y_lo"] <= placement["y"] <= layout["y_hi"]


def test_labels_never_widen_the_axis() -> None:
    """The axis must be identical with and without gene markers configured."""
    bare = _plan_lollipop_layout(
        [], use_log=True, scale_mean_cnv=0.0, data_range=(-0.3, 0.3),
    )
    many = _plan_lollipop_layout(
        _prepare_lollipop_points(
            [_gene(f"GENE{i}", 5e7 + i * 5e5, 0.25, "gain") for i in range(40)],
            use_log=True, scale_mean_cnv=0.0,
        ),
        use_log=True, scale_mean_cnv=0.0, data_range=(-0.3, 0.3),
    )
    assert (many["y_lo"], many["y_hi"]) == (bare["y_lo"], bare["y_hi"])
    assert many["ticks"].major == bare["ticks"].major


def test_label_side_flips_when_the_preferred_side_has_no_room() -> None:
    from robin.gui.components.cnv import _lollipop_label_side

    high_gain = {"gene": "KIAA1549", "direction": "gain", "y_disp": 0.95,
                 "baseline_y": 0.0}
    assert _lollipop_label_side(high_gain) == "above"
    assert _lollipop_label_side(high_gain, y_lo=-1.0, y_hi=1.0) == "below"


def test_offscale_gene_marker_is_clamped_and_flagged() -> None:
    points = [_gene("CDKN2A", 2.0e8, -6.0, "loss")]
    prepared = _prepare_lollipop_points(points, use_log=True, scale_mean_cnv=0.0)
    layout = _plan_lollipop_layout(
        prepared, use_log=True, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3),
    )
    marker = layout["points"][0]
    assert marker["y_disp"] > -6.0
    assert marker["capped"] is True
    # The rest of the profile still gets a usable share of the panel.
    assert layout["y_hi"] - layout["y_lo"] < 6.0


def test_linear_mode_never_drops_below_zero_ploidy() -> None:
    points = [_gene("PTEN", 1.0e8, 0.2, "loss")]
    prepared = _prepare_lollipop_points(points, use_log=False, scale_mean_cnv=2.0)
    layout = _plan_lollipop_layout(
        prepared, use_log=False, scale_mean_cnv=2.0,
        data_range=(1.6, 2.4),
    )
    assert layout["y_lo"] >= 0.0


def test_reference_lines_are_added_for_both_scales() -> None:
    log_series = _cnv_reference_line_series(
        use_log=True, dark=False, chromosome="chr7", sex_estimate="XY",
        y_lo=-1.5, y_hi=1.5,
    )
    values = [d["yAxis"] for d in log_series["markLine"]["data"]]
    assert 0.0 in values
    assert len(values) > 1  # baseline plus calling thresholds

    ploidy_series = _cnv_reference_line_series(
        use_log=False, dark=False, chromosome="All", sex_estimate="XY",
        y_lo=0.0, y_hi=5.0,
    )
    assert 2.0 in [d["yAxis"] for d in ploidy_series["markLine"]["data"]]


def test_reference_series_is_replaced_not_duplicated() -> None:
    chart = _FakeChart()
    chart.options["series"] = [_scatter()]
    for _ in range(3):
        _upsert_cnv_reference_lines(
            chart, use_log=True, dark=False, chromosome="chr7",
            sex_estimate="XY", y_lo=-1.0, y_hi=1.0,
        )
    names = [s.get("name") for s in chart.options["series"]]
    assert names.count("cnv_reference_lines") == 1


def test_segment_series_is_flat_within_each_segment() -> None:
    """The array-plot look: one constant level per segment."""
    rng = np.random.default_rng(1)
    xs = (np.arange(200) * 1e6).tolist()
    ys = np.concatenate([rng.normal(0.0, 0.1, 100), rng.normal(0.8, 0.1, 100)]).tolist()
    series = _cnv_trend_series("chr7", xs, ys, dark=False)

    assert series["type"] == "line"
    assert series["showSymbol"] is False
    assert series["connectNulls"] is False
    levels = sorted({pt[1] for pt in series["data"] if pt[1] is not None})
    assert len(levels) == 2
    assert levels[0] == pytest.approx(0.0, abs=0.06)
    assert levels[1] == pytest.approx(0.8, abs=0.06)


def test_segment_series_draws_detached_bars_without_risers() -> None:
    """Regression: a connected step spiked wherever a dropout formed a segment."""
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 0.12, 200)
    values[100:103] = -5.0          # unmappable dropout, as at a centromere
    xs = (np.arange(200) * 1e6).tolist()
    series = _cnv_trend_series("chr1", xs, values.tolist(), dark=False)

    data = series["data"]
    risers = [
        (data[i], data[i + 1])
        for i in range(len(data) - 1)
        if data[i][1] is not None
        and data[i + 1][1] is not None
        and data[i][0] == data[i + 1][0]
        and data[i][1] != data[i + 1][1]
    ]
    assert risers == []
    # The dropout is still represented, just as its own detached bar.
    assert any(pt[1] is not None and pt[1] < -4.0 for pt in data)


def test_segment_series_does_not_step_on_flat_noise() -> None:
    rng = np.random.default_rng(2)
    xs = (np.arange(250) * 1e6).tolist()
    ys = rng.normal(0.0, 0.15, 250).tolist()
    series = _cnv_trend_series("chr1", xs, ys, dark=False)
    levels = {pt[1] for pt in series["data"] if pt[1] is not None}
    assert len(levels) == 1


def test_trend_series_is_skipped_for_a_stub_track() -> None:
    assert _cnv_trend_series("chr7", [0.0], [1.0], dark=False) is None


def test_difference_window_is_symmetric_and_bounded() -> None:
    assert _cnv_diff_y_window((-0.4, 0.5)) == (-1.0, 1.0)
    assert _cnv_diff_y_window(None) == (-1.0, 1.0)
    lo, hi = _cnv_diff_y_window((-50.0, 50.0))
    assert lo == -hi and hi <= 6.0


def test_dense_scatter_uses_the_batched_render_path() -> None:
    chart = _FakeChart()
    chart.options["series"] = [
        _scatter(),
        {"type": "scatter", "name": "cytobands_highlight", "data": []},
    ]
    _apply_cnv_scatter_performance(chart)
    assert _series_by_name(chart, "chr7")["large"] is True
    # Overlay series must keep per-point styling.
    assert "large" not in _series_by_name(chart, "cytobands_highlight")


def test_full_chart_option_is_json_serialisable() -> None:
    """The option dict is pushed to the browser via setOption, so it must serialise."""
    chart = _FakeChart()
    chart.options["series"] = [_scatter()]
    _upsert_configured_gene_coverage_lollipops(
        chart,
        [_gene("EGFR", 2.0e8, 0.9, "gain")],
        use_log=True, dark=False, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3),
    )
    _upsert_cnv_reference_lines(
        chart, use_log=True, dark=False, chromosome="chr7",
        sex_estimate="XY", y_lo=-1.0, y_hi=1.0,
    )
    json.dumps(_cnv_echarts_option_to_json(chart.options))


def _track(level: float, n: int = 150, sd: float = 0.15, seed: int = 3):
    return np.random.default_rng(seed).normal(level, sd, n)


def _gene_points(filter_mode: str, *, use_log: bool = True, baseline: float = 0.0):
    from robin.gui.components.cnv import _build_configured_gene_coverage_points

    bw = 1_000_000
    tracks = {
        "chr1": _track(0.0),                 # flat
        "chr7": _track(0.50, seed=4),        # whole-chromosome gain
        "chr10": _track(-0.45, seed=5),      # whole-chromosome loss
    }
    if not use_log:
        tracks = {c: 2.0 * np.power(2.0, v) for c, v in tracks.items()}
    offsets, off = {}, 0.0
    for c, v in tracks.items():
        offsets[c] = off
        off += len(v) * bw
    locs = [
        {"gene": "EGFR", "chrom": "chr7", "start_pos": 55_000_000, "end_pos": 55_200_000},
        {"gene": "PTEN", "chrom": "chr10", "start_pos": 89_000_000, "end_pos": 89_200_000},
        {"gene": "MYCN", "chrom": "chr1", "start_pos": 15_000_000, "end_pos": 15_200_000},
    ]
    points, _ = _build_configured_gene_coverage_points(
        locs,
        selected="All",
        chrom_offsets=offsets,
        abs_plot_map=tracks,
        bin_width=bw,
        coverage_by_gene={"EGFR": 60.0, "PTEN": 40.0, "MYCN": 50.0},
        filter_mode=filter_mode,
        use_log=use_log,
        scale_mean_cnv=baseline,
        sex_estimate="XY",
    )
    return {p["gene"] for p in points}


def test_outliers_filter_reports_genes_on_a_gained_chromosome() -> None:
    """Regression: a whole-chromosome gain used to hide every gene on it.

    The rule compared a gene with its own chromosome's mean, so on a chromosome
    that is gained end to end the deviation was ~0 and nothing was reported.
    """
    assert _gene_points("outliers") == {"EGFR", "PTEN"}


def test_outliers_filter_still_hides_genes_on_flat_ground() -> None:
    assert "MYCN" not in _gene_points("outliers")


def test_all_filter_reports_every_configured_gene() -> None:
    assert _gene_points("all") == {"EGFR", "PTEN", "MYCN"}


def test_outliers_filter_works_on_the_ploidy_scale() -> None:
    assert _gene_points("outliers", use_log=False, baseline=2.0) == {"EGFR", "PTEN"}


def test_gene_without_target_coverage_is_still_marked() -> None:
    """The marker is the gene's CNV value, so depth is no longer required."""
    from robin.gui.components.cnv import _build_configured_gene_coverage_points

    bw = 1_000_000
    tracks = {"chr7": _track(0.5, seed=4)}
    points, _ = _build_configured_gene_coverage_points(
        [{"gene": "EGFR", "chrom": "chr7", "start_pos": 55_000_000,
          "end_pos": 55_200_000}],
        selected="All",
        chrom_offsets={"chr7": 0.0},
        abs_plot_map=tracks,
        bin_width=bw,
        coverage_by_gene={},
        filter_mode="outliers",
        use_log=True,
        scale_mean_cnv=0.0,
        sex_estimate="XY",
    )
    assert [p["gene"] for p in points] == ["EGFR"]
    assert points[0]["coverage"] is None
    assert points[0]["coverage_ratio"] is None


def test_single_chromosome_view_can_use_a_fixed_window() -> None:
    """The Chr Y-range control fixes the live view the same way as the report."""
    points = _prepare_lollipop_points(
        [_gene("CDKN2A", 2.0e7, -2.8, "loss")], use_log=True, scale_mean_cnv=0.0
    )
    for span in (2.0, 1.2):
        layout = _plan_lollipop_layout(
            points, use_log=True, scale_mean_cnv=0.0,
            data_range=(-0.3, 0.3), fixed_axis_log2=span,
        )
        assert (layout["y_lo"], layout["y_hi"]) == (-span, span)
        # A deeper event is pinned to the edge and flagged, never dropped.
        assert layout["points"][0]["y_disp"] == pytest.approx(-span)
        assert layout["points"][0]["capped"] is True


def test_fixed_window_is_independent_of_the_data() -> None:
    quiet = _plan_lollipop_layout(
        [], use_log=True, scale_mean_cnv=0.0,
        data_range=(-0.05, 0.05), fixed_axis_log2=2.0,
    )
    busy = _plan_lollipop_layout(
        [], use_log=True, scale_mean_cnv=0.0,
        data_range=(-1.8, 1.9), fixed_axis_log2=2.0,
    )
    assert (quiet["y_lo"], quiet["y_hi"]) == (busy["y_lo"], busy["y_hi"]) == (-2.0, 2.0)


def test_auto_window_still_fits_the_data() -> None:
    layout = _plan_lollipop_layout(
        [], use_log=True, scale_mean_cnv=0.0,
        data_range=(-0.3, 0.3), fixed_axis_log2=None,
    )
    assert layout["y_hi"] - layout["y_lo"] < 4.0


def test_centromere_lines_mark_each_chromosome_in_the_genome_view() -> None:
    from robin.gui.components.cnv import (
        _cnv_centromere_boundaries,
        _upsert_cnv_centromere_lines,
    )

    boundaries = _cnv_centromere_boundaries()
    offsets = {"chr1": 0.0, "chr2": 248_956_422.0}
    chart = _FakeChart()
    _upsert_cnv_centromere_lines(
        chart, dark=False, selected="All", chrom_offsets=offsets
    )
    series = _series_by_name(chart, "cnv_centromere_lines")
    lines = series["markLine"]["data"]
    assert len(lines) == 2
    assert lines[0]["xAxis"] == pytest.approx(boundaries["chr1"])
    assert lines[1]["xAxis"] == pytest.approx(
        offsets["chr2"] + boundaries["chr2"]
    )
    # Faint, dashed and behind the data.
    assert lines[0]["lineStyle"]["type"] == "dashed"
    assert series["z"] < 2


def test_centromere_line_for_a_single_chromosome_is_not_offset() -> None:
    from robin.gui.components.cnv import (
        _cnv_centromere_boundaries,
        _upsert_cnv_centromere_lines,
    )

    chart = _FakeChart()
    _upsert_cnv_centromere_lines(
        chart, dark=True, selected="chr9", chrom_offsets={}
    )
    lines = _series_by_name(chart, "cnv_centromere_lines")["markLine"]["data"]
    assert len(lines) == 1
    assert lines[0]["xAxis"] == pytest.approx(_cnv_centromere_boundaries()["chr9"])


def test_centromere_series_is_replaced_not_duplicated() -> None:
    from robin.gui.components.cnv import _upsert_cnv_centromere_lines

    chart = _FakeChart()
    for _ in range(3):
        _upsert_cnv_centromere_lines(
            chart, dark=False, selected="chr1", chrom_offsets={}
        )
    names = [s.get("name") for s in chart.options["series"]]
    assert names.count("cnv_centromere_lines") == 1


def test_no_centromere_series_for_an_unknown_contig() -> None:
    from robin.gui.components.cnv import _upsert_cnv_centromere_lines

    chart = _FakeChart()
    _upsert_cnv_centromere_lines(
        chart, dark=False, selected="chrZZ", chrom_offsets={}
    )
    names = [s.get("name") for s in chart.options["series"]]
    assert "cnv_centromere_lines" not in names
