"""Shared CNV plot conventions used by both the live GUI and the PDF report."""

from __future__ import annotations

import numpy as np
import pytest

from robin.cnv_plot_style import (
    CNV_LOG2_MIN_AXIS_SPAN,
    assign_label_lanes,
    clamp_band_for_markers,
    cnv_log2_deviation,
    cnv_segment_bounds,
    cnv_segment_line,
    cnv_segment_points,
    cnv_segment_spans,
    gene_crosses_cutoff,
    cnv_axis_tick_spec,
    cnv_axis_ticks,
    cnv_chromosome_axis_window,
    cnv_reference_levels,
    cnv_trend_line,
    cnv_trend_points,
    gutter_label_y,
    clear_reference_levels,
    horizontal_label_placements,
    horizontal_lane_height,
    horizontal_label_proximity_frac,
    lane_counts,
    reserve_label_gutters,
    resolve_tick_interval,
    snap_axis_window_to_ticks,
    solve_lane_height,
)


def test_tick_interval_gets_finer_as_the_window_tightens() -> None:
    """The whole point of the tighter axis: a narrow window earns finer ticks."""
    assert resolve_tick_interval(0.5) <= 0.05
    assert resolve_tick_interval(1.2) <= 0.125
    assert resolve_tick_interval(8.0) <= 1.0


def test_axis_ticks_are_anchored_on_zero() -> None:
    ticks = cnv_axis_ticks(-0.62, 0.62, 0.125)
    assert 0.0 in ticks
    assert min(ticks) >= -0.62
    assert max(ticks) <= 0.62


def test_tick_spec_minor_interval_subdivides_major() -> None:
    spec = cnv_axis_tick_spec(-1.0, 1.0, use_log=True)
    assert spec.minor == pytest.approx(spec.major / 2.0)
    assert 0.0 in spec.major_ticks()


def test_snap_window_rounds_outwards_onto_ticks() -> None:
    lo, hi = snap_axis_window_to_ticks(-0.53, 0.41, 0.25)
    assert lo == pytest.approx(-0.75)
    assert hi == pytest.approx(0.5)


def test_snap_window_respects_the_linear_floor() -> None:
    lo, hi = snap_axis_window_to_ticks(-0.2, 4.1, 0.5, clamp_min=0.0)
    assert lo == 0.0
    assert hi >= 4.1


def test_log2_reference_levels_carry_baseline_and_thresholds() -> None:
    levels = cnv_reference_levels(
        use_log=True, gain_threshold=0.3, loss_threshold=-0.3
    )
    assert (0.0, "baseline") in levels
    assert (0.3, "threshold") in levels
    assert (-0.3, "threshold") in levels


def test_ploidy_reference_levels_mark_diploid_as_the_baseline() -> None:
    levels = cnv_reference_levels(use_log=False)
    assert (2.0, "baseline") in levels
    assert (1.0, "ploidy") in levels


def test_reference_levels_outside_the_window_are_dropped() -> None:
    levels = cnv_reference_levels(
        use_log=True, gain_threshold=0.3, loss_threshold=-0.3, y_lo=-0.2, y_hi=0.2
    )
    assert levels == [(0.0, "baseline")]


def test_trend_line_tracks_a_step_change() -> None:
    values = np.concatenate([np.zeros(100), np.full(100, 0.8)])
    trend = cnv_trend_line(values, target_segments=20)
    assert trend is not None
    assert trend[10] == pytest.approx(0.0, abs=0.05)
    assert trend[-10] == pytest.approx(0.8, abs=0.05)


def test_trend_line_preserves_gaps_rather_than_bridging_them() -> None:
    values = np.zeros(100)
    values[40:60] = np.nan
    trend = cnv_trend_line(values, target_segments=20)
    assert trend is not None
    assert np.all(np.isnan(trend[40:60]))
    assert np.isfinite(trend[0])


def test_trend_line_needs_enough_bins() -> None:
    assert cnv_trend_line(np.zeros(3)) is None
    assert cnv_trend_line(np.full(50, np.nan)) is None


def test_trend_points_break_the_line_with_none() -> None:
    values = np.zeros(60)
    values[20:30] = np.nan
    points = cnv_trend_points(np.arange(60) * 1e6, values)
    assert len(points) == 60
    assert points[25][1] is None
    assert points[0][1] is not None


def test_label_lanes_use_the_axis_span_not_the_label_spread() -> None:
    """Genes clustered in a few Mb of a 3.1 Gb genome must still be separated."""
    xs = [5e7 + i * 5e5 for i in range(5)]
    sides = ["above"] * 5
    lanes = assign_label_lanes(xs, sides, x_span=3.1e9)
    assert lanes == [0, 1, 2, 3, 4]


def test_label_lanes_leave_distant_genes_on_the_same_lane() -> None:
    lanes = assign_label_lanes([1e7, 1.5e9, 3.0e9], ["above"] * 3, x_span=3.1e9)
    assert lanes == [0, 0, 0]


def test_lane_counts_split_gains_above_and_losses_below() -> None:
    top, bottom = lane_counts([0, 1, 0], ["above", "above", "below"])
    assert (top, bottom) == (2, 1)


def test_gutters_place_every_label_outside_the_data_band() -> None:
    y_lo, y_hi, lane_height = reserve_label_gutters(
        -0.5, 0.5, top_lanes=2, bottom_lanes=2
    )
    assert y_lo < -0.5
    assert y_hi > 0.5
    for lane in range(2):
        above = gutter_label_y(
            side="above", lane=lane, data_lo=-0.5, data_hi=0.5, lane_height=lane_height
        )
        below = gutter_label_y(
            side="below", lane=lane, data_lo=-0.5, data_hi=0.5, lane_height=lane_height
        )
        assert above > 0.5 and above <= y_hi
        assert below < -0.5 and below >= y_lo


def test_gutters_do_not_grow_on_a_side_with_no_labels() -> None:
    y_lo, y_hi, _ = reserve_label_gutters(-0.5, 0.5, top_lanes=2, bottom_lanes=0)
    assert y_lo == pytest.approx(-0.5)
    assert y_hi > 0.5


def test_gutters_respect_the_linear_floor() -> None:
    y_lo, _y_hi, _ = reserve_label_gutters(
        0.0, 4.0, top_lanes=1, bottom_lanes=2, clamp_min=0.0
    )
    assert y_lo == 0.0


def test_lane_height_solves_for_the_final_expanded_span() -> None:
    band = 2.0
    frac = 0.06
    total_lanes = 3.5
    lane = solve_lane_height(band, total_lanes=total_lanes, lane_height_frac=frac)
    final_span = band + total_lanes * lane
    assert lane / final_span == pytest.approx(frac)


def test_lane_height_caps_the_gutter_when_labels_would_swallow_the_panel() -> None:
    lane = solve_lane_height(2.0, total_lanes=40.0, lane_height_frac=0.06)
    assert 40.0 * lane <= 2.0 * 0.56


def test_markers_may_widen_the_band_only_so_far() -> None:
    """A deeply deleted gene must not flatten the rest of the profile."""
    lo, hi = clamp_band_for_markers(-0.5, 0.5, [-4.0, 0.6])
    assert lo > -4.0
    assert lo < -0.5
    assert hi >= 0.6


def test_markers_inside_the_allowance_widen_the_band_exactly() -> None:
    lo, hi = clamp_band_for_markers(-1.0, 1.0, [-1.1, 1.05])
    assert lo == pytest.approx(-1.1)
    assert hi == pytest.approx(1.05)


def test_band_is_unchanged_without_finite_markers() -> None:
    assert clamp_band_for_markers(-1.0, 1.0, []) == (-1.0, 1.0)
    assert clamp_band_for_markers(-1.0, 1.0, [np.nan]) == (-1.0, 1.0)


def test_min_axis_span_stays_tight_enough_to_resolve_small_changes() -> None:
    """Guards the resolution fix: the floor must stay well inside a real gain."""
    assert CNV_LOG2_MIN_AXIS_SPAN <= 1.0


def test_gene_crosses_cutoff_uses_the_calling_thresholds() -> None:
    call = dict(gain_threshold=0.3, loss_threshold=-0.3, use_log=True)
    assert gene_crosses_cutoff(0.5, **call) is True
    assert gene_crosses_cutoff(-0.5, **call) is True
    assert gene_crosses_cutoff(0.2, **call) is False
    assert gene_crosses_cutoff(0.0, **call) is False


def test_gene_on_a_gained_chromosome_crosses_the_cutoff() -> None:
    """The whole point of judging against the genome, not the chromosome.

    A gene sitting mid-way through a chromosome that is gained end to end has
    almost no deviation from *that chromosome's* mean, so a per-chromosome
    standard-deviation rule reports it as normal. Against the genome-wide
    baseline it is what it is: a gain.
    """
    assert gene_crosses_cutoff(
        0.55, gain_threshold=0.3, loss_threshold=-0.3, use_log=True
    ) is True


def test_gene_crosses_cutoff_on_the_ploidy_scale() -> None:
    call = dict(gain_threshold=0.3, loss_threshold=-0.3, use_log=False, baseline=2.0)
    # 3 copies against diploid is log2(3/2) = 0.58, past the cut-off.
    assert gene_crosses_cutoff(3.0, **call) is True
    # A single copy is log2(0.5) = -1.0.
    assert gene_crosses_cutoff(1.0, **call) is True
    assert gene_crosses_cutoff(2.1, **call) is False


def test_gene_crosses_cutoff_is_safe_on_bad_input() -> None:
    call = dict(gain_threshold=0.3, loss_threshold=-0.3)
    assert gene_crosses_cutoff(float("nan"), **call) is False
    # Ploidy scale with no usable baseline cannot be judged.
    assert gene_crosses_cutoff(3.0, use_log=False, baseline=0.0, **call) is False
    assert gene_crosses_cutoff(0.0, use_log=False, baseline=2.0, **call) is False


def test_log2_deviation_passes_log_values_through() -> None:
    assert cnv_log2_deviation(0.4, use_log=True) == pytest.approx(0.4)
    assert cnv_log2_deviation(4.0, use_log=False, baseline=2.0) == pytest.approx(1.0)
    assert cnv_log2_deviation(4.0, use_log=False, baseline=None) is None


def _noisy(level, n, sd=0.15, seed=0):
    return np.random.default_rng(seed).normal(level, sd, n)


def test_flat_track_is_a_single_segment() -> None:
    """A rolling average wanders on noise; a segmentation must not."""
    assert len(cnv_segment_bounds(_noisy(0.0, 250))) == 1
    assert np.nanstd(cnv_segment_line(_noisy(0.0, 250))) == pytest.approx(0.0)


def test_whole_chromosome_gain_is_a_single_flat_segment() -> None:
    line = cnv_segment_line(_noisy(0.5, 200, seed=4))
    assert len(cnv_segment_bounds(_noisy(0.5, 200, seed=4))) == 1
    assert float(np.nanmedian(line)) == pytest.approx(0.5, abs=0.05)


def test_a_real_step_is_split_at_the_right_place() -> None:
    values = np.concatenate([_noisy(0.0, 120, seed=1), _noisy(0.55, 130, seed=2)])
    bounds = cnv_segment_bounds(values)
    assert len(bounds) == 2
    # The break lands within a few bins of the true boundary.
    assert abs(bounds[0][1] - 120) <= 5


def test_focal_deletion_survives_segmentation() -> None:
    """A four-bin homozygous deletion must not be smoothed away."""
    values = np.concatenate([_noisy(0.0, 100, 0.12, 3), np.full(4, -2.4),
                             _noisy(0.0, 96, 0.12, 5)])
    line = cnv_segment_line(values)
    assert len(cnv_segment_bounds(values)) == 3
    assert line[101] < -2.0
    assert line[50] == pytest.approx(0.0, abs=0.05)


def test_segment_line_preserves_gaps() -> None:
    values = _noisy(0.0, 200)
    values[80:100] = np.nan
    line = cnv_segment_line(values)
    assert np.all(np.isnan(line[80:100]))
    assert np.isfinite(line[0])


def test_segment_line_needs_enough_bins() -> None:
    assert cnv_segment_line(np.zeros(3)) is None
    assert cnv_segment_line(np.full(50, np.nan)) is None


def test_segment_points_are_detached_bars() -> None:
    """Each segment is emitted as start/end/break, so no riser is ever drawn."""
    values = np.concatenate([_noisy(0.0, 100, 0.12, 3), _noisy(0.6, 100, 0.12, 4)])
    points = cnv_segment_points(np.arange(200) * 1e6, values)

    assert len(points) % 3 == 0
    for i in range(0, len(points), 3):
        start, end, gap = points[i], points[i + 1], points[i + 2]
        assert start[1] == end[1]      # bar is horizontal
        assert start[0] < end[0]
        assert gap[1] is None          # break before the next bar


def test_segment_points_skip_gaps() -> None:
    values = _noisy(0.0, 60)
    values[20:30] = np.nan
    points = cnv_segment_points(np.arange(60) * 1e6, values)
    levels = [pt[1] for pt in points if pt[1] is not None]
    assert levels
    assert all(np.isfinite(v) for v in levels)


def test_segment_spans_cover_each_level_once() -> None:
    values = np.concatenate([_noisy(0.0, 100, 0.12, 3), np.full(3, -5.0),
                             _noisy(0.0, 97, 0.12, 5)])
    spans = cnv_segment_spans(np.arange(200) * 1e6, values)
    # The dropout is its own span rather than a riser on the neighbouring one.
    dropouts = [s for s in spans if s[2] < -4.0]
    assert len(dropouts) == 1
    start, end, _level = dropouts[0]
    assert end - start <= 3e6


def test_segment_line_is_flatter_than_a_rolling_average() -> None:
    """The complaint that motivated this: the rolling trace looked shaky."""
    from robin.cnv_plot_style import cnv_trend_line

    values = _noisy(0.0, 250, seed=7)
    assert np.nanstd(cnv_segment_line(values)) < np.nanstd(cnv_trend_line(values))


def test_chromosome_axis_window_is_symmetric_in_log2() -> None:
    assert cnv_chromosome_axis_window(use_log=True, span_log2=1.2) == (-1.2, 1.2)


def test_chromosome_axis_window_converts_to_ploidy() -> None:
    """The same window expressed as copy number, so both scales are comparable."""
    lo, hi = cnv_chromosome_axis_window(use_log=False, span_log2=1.2, baseline=2.0)
    assert hi == pytest.approx(2.0 * 2**1.2)
    assert lo == pytest.approx(2.0 * 2**-1.2)
    assert lo > 0


def test_chromosome_axis_window_survives_a_bad_baseline() -> None:
    lo, hi = cnv_chromosome_axis_window(use_log=False, span_log2=1.2, baseline=0.0)
    assert lo > 0 and hi > lo


def test_horizontal_label_width_scales_with_name_and_font() -> None:
    """A horizontal name needs room along x; a rotated one does not."""
    narrow = horizontal_label_proximity_frac(
        ["TP53"], font_size=4.5, panel_width_pt=1152.0
    )
    wide = horizontal_label_proximity_frac(
        ["KIAA1549"], font_size=4.5, panel_width_pt=1152.0
    )
    bigger_font = horizontal_label_proximity_frac(
        ["TP53"], font_size=8.0, panel_width_pt=1152.0
    )
    assert wide > narrow
    assert bigger_font > narrow


def test_horizontal_labels_stack_when_they_would_collide() -> None:
    names = ["KIAA1549", "CDKN2A", "MSH6"]
    placements = horizontal_label_placements(
        [1.0e8, 1.02e8, 1.04e8],
        names,
        ["above"] * 3,
        [0.5, 0.5, 0.5],
        x_span=3.1e9,
        lane_height=0.05,
        font_size=4.5,
        panel_width_pt=1152.0,
    )
    lanes = [p["lane"] for p in placements]
    assert lanes == [0, 1, 2]
    # Each lane sits further from the marker than the last.
    ys = [p["y"] for p in placements]
    assert ys == sorted(ys)


def test_horizontal_labels_share_a_lane_when_far_apart() -> None:
    placements = horizontal_label_placements(
        [1.0e7, 1.5e9, 3.0e9],
        ["TP53", "EGFR", "PTEN"],
        ["above"] * 3,
        [0.4, 0.4, 0.4],
        x_span=3.1e9,
        lane_height=0.05,
        font_size=4.5,
        panel_width_pt=1152.0,
    )
    assert [p["lane"] for p in placements] == [0, 0, 0]


def test_horizontal_labels_read_away_from_the_baseline() -> None:
    placements = horizontal_label_placements(
        [1.0e7, 2.0e9],
        ["GAIN", "LOSS"],
        ["above", "below"],
        [0.5, -0.5],
        x_span=3.1e9,
        lane_height=0.05,
        font_size=4.5,
        panel_width_pt=1152.0,
    )
    assert placements[0]["y"] > 0.5
    assert placements[1]["y"] < -0.5


def test_horizontal_labels_stay_inside_the_panel() -> None:
    """Stacking must not push a name off the axis."""
    placements = horizontal_label_placements(
        [1.0e8 + i * 1e6 for i in range(8)],
        ["KIAA1549"] * 8,
        ["above"] * 8,
        [0.9] * 8,
        x_span=3.1e9,
        lane_height=0.2,
        font_size=4.5,
        panel_width_pt=1152.0,
        y_lo=-1.0,
        y_hi=1.0,
    )
    assert all(-1.0 <= p["y"] <= 1.0 for p in placements)


def test_label_ladder_clears_a_reference_line() -> None:
    """A gene name printed along the cut-off line is effectively unreadable."""
    on_the_line = horizontal_label_placements(
        [1.0e8],
        ["MTAP"],
        ["above"],
        [0.15],
        x_span=1.4e8,
        lane_height=0.165,
        font_size=4.5,
        panel_width_pt=648.0,
        reference_levels=[0.0, 0.3, -0.3],
    )[0]["y"]
    # Without the guides the lane would land at 0.315, right on the +0.3 cut-off.
    assert abs(on_the_line - 0.3) > 0.06


def test_label_ladder_keeps_its_spacing_after_clearing() -> None:
    placements = horizontal_label_placements(
        [1.0e8, 1.01e8, 1.02e8],
        ["MTAP", "CDKN2A", "CDKN2B"],
        ["above"] * 3,
        [0.15] * 3,
        x_span=1.4e8,
        lane_height=0.165,
        font_size=4.5,
        panel_width_pt=648.0,
        reference_levels=[0.0, 0.3, -0.3],
    )
    ys = sorted(p["y"] for p in placements)
    gaps = [round(b - a, 6) for a, b in zip(ys, ys[1:])]
    assert len(set(gaps)) == 1          # evenly stacked, none collapsed together
    assert all(abs(y - 0.3) > 0.06 for y in ys)


def test_clearing_pushes_away_from_the_marker_only() -> None:
    above = clear_reference_levels(
        0.31, side="above", reference_levels=[0.3], clearance=0.1
    )
    below = clear_reference_levels(
        -0.31, side="below", reference_levels=[-0.3], clearance=0.1
    )
    assert above > 0.31
    assert below < -0.31


def test_clearing_is_a_no_op_without_reference_lines() -> None:
    assert clear_reference_levels(
        0.5, side="above", reference_levels=[], clearance=0.1
    ) == 0.5
    assert clear_reference_levels(
        0.5, side="above", reference_levels=[0.0], clearance=0.0
    ) == 0.5


def test_lane_spacing_is_constant_in_points_across_y_ranges() -> None:
    """The user changes the Y range; the gap between names must not change.

    A fixed fraction of the span looks stable in data units but is only stable
    on screen if the panel height is fixed too — and it ignores the font size.
    """
    panel_height_pt = 108.0
    gaps = []
    for span in (1.2, 3.0, 4.0, 8.0):
        lane = horizontal_lane_height(
            y_span=span, panel_height_pt=panel_height_pt, font_size=4.5
        )
        gaps.append(round(lane / span * panel_height_pt, 6))
    assert len(set(gaps)) == 1


def test_lane_spacing_scales_with_the_label_size() -> None:
    small = horizontal_lane_height(y_span=3.0, panel_height_pt=108.0, font_size=3.5)
    large = horizontal_lane_height(y_span=3.0, panel_height_pt=108.0, font_size=8.0)
    assert large > small


def test_lane_spacing_clears_the_text_height() -> None:
    """Lanes must be further apart than the text is tall, or names touch."""
    panel_height_pt = 108.0
    span = 3.0
    for font in (3.5, 4.5, 6.0, 8.0):
        lane_pt = (
            horizontal_lane_height(
                y_span=span, panel_height_pt=panel_height_pt, font_size=font
            )
            / span
            * panel_height_pt
        )
        assert lane_pt > font * 1.3


def test_lane_spacing_survives_an_unknown_panel_height() -> None:
    lane = horizontal_lane_height(y_span=3.0, panel_height_pt=0.0, font_size=4.5)
    assert lane > 0


def test_colliding_labels_step_from_one_shared_base() -> None:
    """A lane index only separates names if every label starts from one height."""
    from robin.cnv_plot_style import horizontal_label_placements

    # Six genes packed onto a short chromosome, at differing CNV values —
    # the chr17 case (NF1, RAD51D, ERBB2, CDK12, BRCA1, RAD51C).
    xs = [31e6, 35e6, 39e6, 39.5e6, 43e6, 58e6]
    labels = ["NF1", "RAD51D", "ERBB2", "CDK12", "BRCA1", "RAD51C"]
    marker_ys = [0.45, 0.50, 0.55, 0.52, 0.62, 0.58]
    lane_height = 0.0355

    placements = horizontal_label_placements(
        xs,
        labels,
        ["above"] * 6,
        marker_ys,
        x_span=3086e6,
        lane_height=lane_height,
        font_size=4.5,
        panel_width_pt=22.96 * 72,
        y_lo=-1.0,
        y_hi=3.0,
    )
    ys = sorted(p["y"] for p in placements)
    # Every name gets its own line: consecutive labels a full lane apart.
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    assert min(gaps) >= lane_height * 0.99, gaps
    # Measuring each ladder from its own marker put RAD51C (marker 0.58, lane 0)
    # and RAD51D (marker 0.50, lane 2) within 0.01 of each other.
    assert len({round(y, 6) for y in ys}) == 6


def test_ladder_slides_back_to_fit_instead_of_piling_at_the_edge() -> None:
    """Clamping each lane to the ceiling stacks the top of the ladder."""
    from robin.cnv_plot_style import horizontal_label_placements

    xs = [31e6, 35e6, 39e6, 39.5e6, 43e6]
    labels = ["NF1", "RAD51D", "ERBB2", "CDK12", "BRCA1"]
    lane_height = 0.05
    placements = horizontal_label_placements(
        xs,
        labels,
        ["above"] * 5,
        [0.6] * 5,
        x_span=3086e6,
        lane_height=lane_height,
        font_size=4.5,
        panel_width_pt=1000.0,
        y_lo=-1.0,
        # Deliberately tight: the full ladder does not fit above the markers.
        y_hi=0.8,
    )
    ys = sorted(p["y"] for p in placements)
    assert len({round(y, 6) for y in ys}) == 5, ys
    assert all(y <= 0.8 + 1e-9 for y in ys), ys


def test_cluster_ids_split_groups_that_are_far_apart() -> None:
    from robin.cnv_plot_style import label_cluster_ids

    ids = label_cluster_ids(
        [0.0, 1.0e6, 500.0e6],
        ["above", "above", "above"],
        x_span=1000.0e6,
        x_proximity_frac=0.01,
    )
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]

    # Labels on opposite sides of the baseline never share a ladder.
    ids = label_cluster_ids(
        [0.0, 0.0], ["above", "below"], x_span=1000.0e6, x_proximity_frac=0.5
    )
    assert ids[0] != ids[1]


def test_robust_gene_value_averages_away_single_bin_noise() -> None:
    """Taking the peak bin is a max-of-N statistic and drifts away from zero."""
    from robin.cnv_plot_style import robust_gene_value

    # One noisy bin past the cut-off among otherwise neutral ones. The peak rule
    # returned -0.35 and called this a loss; the gene is not lost.
    neutral = [0.10, -0.35, 0.05, 0.20, -0.10]
    assert abs(robust_gene_value(neutral, cutoff=0.3)) < 0.3

    # A gene genuinely at -0.5 is unaffected by the change.
    real = [-0.50, -0.45, -0.55, -0.50, -0.48]
    assert robust_gene_value(real, cutoff=0.3) == pytest.approx(-0.496, abs=0.01)


def test_robust_gene_value_keeps_a_focal_event_at_its_own_depth() -> None:
    """Averaging alone would dilute a homozygous deletion out of existence."""
    from robin.cnv_plot_style import robust_gene_value

    # Two consecutive deep bins inside a large gene: the mean is -0.28, inside
    # the cut-off, so the focal rescue has to carry it. A single bin no longer
    # qualifies: a lone deep bin is a coverage dropout rather than a deletion
    # (CNV_GENE_FOCAL_RESCUE_MIN_BINS).
    focal = [-2.5, -2.5] + [0.0] * 16
    assert robust_gene_value(focal, cutoff=0.3) == pytest.approx(-2.5)
    assert abs(sum(focal) / len(focal)) < 0.3

    # The rescue is scaled to the cut-off in force, not hard-coded.
    from robin.cnv_plot_style import CNV_GENE_FOCAL_RESCUE_MULTIPLE

    under = -0.3 * CNV_GENE_FOCAL_RESCUE_MULTIPLE + 0.01
    just_under = [under, under] + [0.0] * 16
    assert abs(robust_gene_value(just_under, cutoff=0.3)) < 0.3
    over = -0.3 * CNV_GENE_FOCAL_RESCUE_MULTIPLE - 0.01
    just_over = [over, over] + [0.0] * 16
    assert abs(robust_gene_value(just_over, cutoff=0.3)) > 0.3


def test_robust_gene_value_handles_gaps_and_empty_input() -> None:
    import numpy as np

    from robin.cnv_plot_style import robust_gene_value

    assert robust_gene_value([], cutoff=0.3) == 0.0
    assert robust_gene_value([np.nan, np.nan], cutoff=0.3) == 0.0
    # Non-finite bins are ignored rather than poisoning the mean.
    assert robust_gene_value([np.nan, -0.5, -0.5], cutoff=0.3) == pytest.approx(-0.5)


def test_robust_gene_value_cuts_the_false_positive_rate() -> None:
    """The measured effect that motivated the change, pinned as a property."""
    import numpy as np

    from robin.cnv_plot_style import robust_gene_value

    rng = np.random.default_rng(0)
    trials = 4000
    peak_calls = robust_calls = 0
    for _ in range(trials):
        window = rng.normal(0.0, 0.18, 5)  # neutral gene, realistic noise
        peak = window[np.argmax(np.abs(window))]
        peak_calls += abs(peak) >= 0.3
        robust_calls += abs(robust_gene_value(window, cutoff=0.3)) >= 0.3

    peak_rate = peak_calls / trials
    robust_rate = robust_calls / trials
    # The peak rule calls a neutral five-bin gene roughly 40% of the time.
    assert peak_rate > 0.30, peak_rate
    # The robust rule should be far below 1%.
    assert robust_rate < 0.01, robust_rate
