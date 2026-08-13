"""Display bin width for the genome-wide CNV summary.

The width is a multiplier on the analysis bin, capped by a ceiling and now held
up by a floor. Without the floor a deep run - where cnv_from_bam sizes analysis
bins at 1 kb - drew the summary at 4 kb, holding ~76 reads per point. That is
11.5% Poisson noise, 0.17 log2, against a 0.30 calling cut-off: the panel filled
with scatter that was counting statistics, not copy number.

The floor is set by the opposing constraint. A homozygous deletion drops a bin's
reads to (1 - deletion/bin) of expected, so a wider bin dilutes a focal event
until it disappears. 50 kb is the widest bin that keeps a 20 kb deletion clear
of the scatter.
"""

import math

import pytest

from robin.analysis.cnv_analysis import (
    CNV_REPORT_GENOME_PLOT_BIN_WIDTH,
    CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING,
    CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH,
    resolve_cnv_report_genome_plot_bin_width as resolve,
)

CUTOFF = 0.30
NOISE_SD = 0.21          # measured on a real sample at 50 kb
STAND_OUT = -0.60        # roughly what a dot must reach to be read as an event


def _deletion_depth(deletion_bp, bin_bp):
    """log2 a homozygous deletion reads at, diluted into a bin of this width."""
    remaining = max(0.0, 1 - min(deletion_bp, bin_bp) / bin_bp)
    return -math.inf if remaining <= 0 else math.log2(remaining)


def test_floor_is_50kb():
    assert CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH == 50_000


def test_deep_run_no_longer_draws_at_4kb():
    """1 kb analysis bins used to yield a 4 kb display bin."""
    assert resolve(1_000) == 50_000


@pytest.mark.parametrize("analysis_bw", [1_000, 2_000, 5_000])
def test_fine_tracks_are_lifted_to_the_floor(analysis_bw):
    assert resolve(analysis_bw) == CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH


@pytest.mark.parametrize("analysis_bw", [7_000, 11_000, 60_000, 235_000])
def test_tuned_range_is_untouched(analysis_bw):
    """The 4x rule was measured at these widths; the floor must not disturb it."""
    expected = min(
        CNV_REPORT_GENOME_PLOT_BIN_WIDTH,
        analysis_bw * CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING,
    )
    assert resolve(analysis_bw) == expected


@pytest.mark.parametrize("analysis_bw", [50_000, 100_000, 200_000])
def test_coarser_tracks_still_follow_the_multiplier(analysis_bw):
    expected = min(
        CNV_REPORT_GENOME_PLOT_BIN_WIDTH,
        analysis_bw * CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING,
    )
    assert resolve(analysis_bw) == expected


def test_shallow_run_is_never_smoothed_further():
    """An analysis bin coarser than the ceiling is returned unchanged."""
    assert resolve(500_000) == 500_000
    assert resolve(1_000_000) == 1_000_000


def test_display_bin_is_never_finer_than_the_analysis_bin():
    for analysis_bw in (1_000, 50_000, 500_000, 2_000_000):
        assert resolve(analysis_bw) >= analysis_bw


def test_a_focal_deletion_still_clears_the_scatter_at_the_floor():
    """The constraint that sets the floor: 20 kb and 40 kb must stay visible."""
    width = CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH
    assert _deletion_depth(20_000, width) < STAND_OUT
    assert _deletion_depth(41_000, width) < STAND_OUT    # CDKN2A/B span


def test_a_wider_floor_would_lose_a_small_deletion():
    """Documents why 100 kb was rejected, so raising it is a deliberate act."""
    assert _deletion_depth(20_000, 100_000) > -NOISE_SD * 2
    assert _deletion_depth(20_000, 400_000) > -NOISE_SD


def test_floor_sits_below_the_ceiling():
    assert CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH < CNV_REPORT_GENOME_PLOT_BIN_WIDTH


def test_floor_can_be_overridden_per_call():
    assert resolve(1_000, floor=10_000) == 10_000
    assert resolve(1_000, floor=0) == 4_000     # back to the old behaviour


def test_floor_engages_only_below_the_tuned_range():
    from robin.analysis.cnv_analysis import CNV_REPORT_GENOME_PLOT_FINE_TRACK_BELOW

    boundary = CNV_REPORT_GENOME_PLOT_FINE_TRACK_BELOW
    assert resolve(boundary - 1) == CNV_REPORT_GENOME_PLOT_MIN_BIN_WIDTH
    assert resolve(boundary) == boundary * CNV_REPORT_GENOME_PLOT_MAX_SMOOTHING


def test_width_never_decreases_within_the_tuned_range():
    widths = [7_000, 11_000, 60_000, 235_000, 431_000, 1_148_000]
    resolved = [resolve(w) for w in widths]
    assert resolved == sorted(resolved), resolved


def test_there_is_a_deliberate_step_at_the_boundary():
    """A 5 kb track gets a wider display bin than a 7 kb one.

    Not an oversight. Below the tuned range the floor lifts the width to 50 kb;
    at 7 kb and above the measured 4x rule takes over and gives 28 kb. Removing
    the step would mean applying the floor inside the range where the 4x rule
    was validated against gene-marker reach, which this change has no evidence
    to overturn. Both widths sit comfortably below the calling cut-off in noise
    terms, so the step costs nothing in practice.
    """
    assert resolve(5_000) > resolve(7_000)


# ---------------------------------------------------------------- chromosome panels

def test_chromosome_panels_do_not_draw_at_the_analysis_width():
    """They used to draw every 1 kb analysis bin - 0.33 log2 of counting noise."""
    from robin.analysis.cnv_analysis import resolve_cnv_plot_bin_width

    assert resolve_cnv_plot_bin_width(1_000, None) == 50_000
    assert resolve_cnv_plot_bin_width(5_000, None) == 50_000


def test_chromosome_panels_honour_an_explicit_choice():
    from robin.analysis.cnv_analysis import resolve_cnv_plot_bin_width

    assert resolve_cnv_plot_bin_width(1_000, 1_000_000) == 1_000_000
    assert resolve_cnv_plot_bin_width(1_000, 500_000) == 500_000


def test_chromosome_panels_leave_coarse_tracks_alone():
    from robin.analysis.cnv_analysis import resolve_cnv_plot_bin_width

    for analysis_bw in (7_000, 60_000, 431_000):
        assert resolve_cnv_plot_bin_width(analysis_bw, None) == analysis_bw


def test_display_width_never_finer_than_analysis_width():
    from robin.analysis.cnv_analysis import resolve_cnv_plot_bin_width

    for analysis_bw in (1_000, 50_000, 431_000, 2_000_000):
        assert resolve_cnv_plot_bin_width(analysis_bw, 1_000) >= analysis_bw
