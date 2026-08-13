"""How readily the segment trend line steps.

Analysts read whole-chromosome and arm-level events off this line, so it has to
follow the profile without chasing scatter. Two constants govern that, and they
do different jobs: the penalty decides whether a split is proposed at all, and
the merge test decides whether two neighbouring levels are distinguishable
afterwards. Only the penalty was relaxed; the merge test is what stops the line
stepping with noise and is deliberately untouched.
"""

import numpy as np
import pytest

from robin.cnv_plot_style import (
    CNV_SEGMENT_MERGE_SIGMAS,
    CNV_SEGMENT_MIN_BINS,
    CNV_SEGMENT_PENALTY,
    cnv_segment_bounds,
)


def test_penalty_stays_high_to_protect_long_levels():
    """Lowering this is what destroys the long neutral lines.

    Measured on a real sample: at 20, 38% of the genome sits in runs longer than
    10 Mb; at 10 that falls to 12% and at 5 to 2%. Focal events are recovered by
    the focal pass instead, so the two behaviours do not compete for one number.
    """
    assert CNV_SEGMENT_PENALTY == 20.0


def test_noise_guard_is_unchanged():
    assert CNV_SEGMENT_MERGE_SIGMAS == 3.0
    assert CNV_SEGMENT_MIN_BINS == 3


def test_a_flat_track_stays_one_segment():
    """The line must not invent structure in noise."""
    rng = np.random.default_rng(0)
    flat = rng.normal(0.0, 0.2, 4000)
    assert len(cnv_segment_bounds(flat)) <= 3


def test_a_real_step_is_found():
    rng = np.random.default_rng(1)
    track = np.concatenate([rng.normal(0.0, 0.2, 2000), rng.normal(0.6, 0.2, 2000)])
    bounds = cnv_segment_bounds(track)
    edges = {start for start, _ in bounds} | {end for _, end in bounds}
    assert any(abs(edge - 2000) < 60 for edge in edges), bounds


def test_a_focal_spike_is_cut_out_of_a_long_level():
    """The shape analysts read off an array plot: long line, short interruption."""
    rng = np.random.default_rng(10)
    track = rng.normal(0.0, 0.05, 3000)
    track[1500:1508] = -1.2          # focal homozygous-style loss, 8 bins
    bounds = cnv_segment_bounds(track)

    focal = [
        (start, end)
        for start, end in bounds
        if float(np.mean(track[start:end])) < -0.5
    ]
    assert focal, bounds
    start, end = focal[0]
    assert end - start <= 24, "focal segment should stay short"
    assert start >= 1490 and end <= 1520, (start, end)

    # And the flat region either side is still carried by long segments.
    longest = max(end - start for start, end in bounds)
    assert longest > 1000, bounds


def test_a_flat_track_gains_no_focal_segments():
    """The focal pass must not manufacture events out of noise."""
    rng = np.random.default_rng(11)
    flat = rng.normal(0.0, 0.2, 4000)
    assert len(cnv_segment_bounds(flat)) <= 4


def test_focal_pass_leaves_a_broad_event_alone():
    """A broad shift is a level change, not a focal run; it must not fragment."""
    rng = np.random.default_rng(12)
    track = np.concatenate([
        rng.normal(0.0, 0.1, 1500),
        rng.normal(-0.6, 0.1, 1500),
    ])
    bounds = cnv_segment_bounds(track)
    assert len(bounds) <= 6, bounds


def test_lower_penalty_gives_at_least_as_many_segments():
    """The direction of the change, asserted rather than assumed."""
    rng = np.random.default_rng(2)
    track = np.concatenate([
        rng.normal(level, 0.2, 500)
        for level in (0.0, 0.3, 0.0, -0.3, 0.0, 0.25, -0.2, 0.0)
    ])
    loose = len(cnv_segment_bounds(track, penalty=10.0))
    tight = len(cnv_segment_bounds(track, penalty=20.0))
    assert loose >= tight


def test_stepped_track_recovers_its_levels():
    """Eight real levels must not be averaged into a coarse line."""
    rng = np.random.default_rng(3)
    levels = (0.0, 0.4, 0.0, -0.4, 0.0, 0.35, -0.3, 0.0)
    track = np.concatenate([rng.normal(v, 0.15, 500) for v in levels])
    bounds = cnv_segment_bounds(track)
    assert len(bounds) >= len(levels), bounds
    for start, end in bounds:
        segment_mean = float(np.mean(track[start:end]))
        assert min(levels) - 0.3 <= segment_mean <= max(levels) + 0.3


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_short_tracks_do_not_raise(n):
    assert isinstance(cnv_segment_bounds(np.zeros(n)), list)
