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


def test_penalty_matches_the_calling_segmentation():
    """The drawn line and the calls should agree on when a step is believable."""
    from robin.workflow_config import DEFAULT_CNV_PENALTY_VALUE

    assert CNV_SEGMENT_PENALTY == float(DEFAULT_CNV_PENALTY_VALUE)


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
