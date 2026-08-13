"""Shared CNV plot conventions for the live GUI (ECharts) and PDF reports (matplotlib).

The neuropathology team reads ROBIN's real-time CNV profiles alongside conumee-style
whole-genome methylation (WGM) CNV plots, so the two need to agree on how the copy
number axis is drawn: a fine, symmetric tick ladder, tick labels mirrored on both
sides of the panel, an explicit baseline at no-change, and a segment trend line over
the bin cloud.

Everything here is pure numpy/pandas so the same numbers drive both renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Tick intervals, finest first. The resolver walks these and takes the smallest
# interval that still yields a readable number of labels for the axis span.
CNV_LOG2_TICK_LADDER: Tuple[float, ...] = (
    0.05,
    0.1,
    0.125,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
)
CNV_PLOIDY_TICK_LADDER: Tuple[float, ...] = (
    0.1,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)

# Target label density. Genome-wide panels are short, so keep the ceiling modest;
# minor ticks carry the extra resolution without crowding the labels.
CNV_AXIS_TARGET_TICKS = 10
CNV_AXIS_MAX_TICKS = 16
CNV_AXIS_MINOR_SUBDIVISIONS = 2

# Values below this are treated as "no usable track" by the trend smoother.
CNV_TREND_MIN_POINTS = 8

# Segmentation. A split must explain more than CNV_SEGMENT_PENALTY times the
# bin-to-bin noise variance to be kept, which is what stops noise producing steps;
#: How many times the calling cut-off a single bin must reach before it can
#: carry a gene call on its own. Gene values are otherwise averaged across the
#: gene's bins, which is what keeps noise out; this rescues a genuine focal
#: event — a homozygous deletion covering part of a large gene — that averaging
#: would dilute below the cut-off.
CNV_GENE_FOCAL_RESCUE_MULTIPLE = 3.0

#: How many *consecutive* bins must reach the rescue threshold before a focal
#: event can carry the gene call. One bin is not enough: coverage dropouts
#: produce isolated bins at extreme negative log2 (a bin at an eighth of expected
#: depth reads as -3), and on the log2 scale that is indistinguishable from a
#: homozygous deletion by depth alone. In testing, single dropout bins sitting
#: among otherwise normal ones produced false gene-level deletion calls, while
#: genuine homozygous deletions were supported by runs of several bins, so
#: requiring two consecutive bins separates the two cleanly.
CNV_GENE_FOCAL_RESCUE_MIN_BINS = 2

#: Smallest window a gene value may be averaged over. Panel targets are often
#: shorter than one analysis bin, which leaves the mean with a single bin and no
#: noise protection at all — at the measured 0.175 per-bin noise a lone bin
#: crosses a 0.3 cut-off about 9% of the time. Padding the window symmetrically
#: to three bins puts the cut-off near three standard errors instead. It does not
#: hide focal amplification: a single bin at +4 still averages well past any
#: cut-off.
CNV_GENE_MIN_EVAL_BINS = 3


def _longest_run_past(arr, threshold: float):
    """(length, end index) of the longest run of consecutive bins past threshold.

    Non-finite bins break a run: a gap in coverage is not evidence of continuity.
    """
    import numpy as _np

    past = _np.isfinite(arr) & (_np.abs(arr) >= threshold)
    best = best_end = run = 0
    for i, flag in enumerate(past):
        run = run + 1 if flag else 0
        if run > best:
            best, best_end = run, i
    return best, best_end


def robust_gene_value(values, *, cutoff: float) -> float:
    """One representative log2 value for a gene, resistant to single-bin noise.

    Taking the most extreme bin is a max-of-N statistic: it drifts away from zero
    as a gene spans more bins, so a neutral gene covering five bins at 0.20 noise
    was called about half the time. The mean across the gene's bins removes that,
    at the cost of diluting a focal event — so a run of at least
    ``CNV_GENE_FOCAL_RESCUE_MIN_BINS`` consecutive bins reaching
    ``CNV_GENE_FOCAL_RESCUE_MULTIPLE`` times the cut-off is reported at its own
    depth instead. The value returned for a rescued event is the peak *within the
    supporting run*, so the number quoted is the one the evidence supports.

    ``values`` must be in genomic order and must not have had gaps removed, or
    bins that are not adjacent would be counted as a run.

    Returns 0.0 when there is nothing finite to summarise.
    """
    import numpy as _np

    arr = _np.asarray(values, dtype=float)
    finite = arr[_np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    threshold = abs(float(cutoff)) * CNV_GENE_FOCAL_RESCUE_MULTIPLE
    run, end = _longest_run_past(arr, threshold)
    if run >= CNV_GENE_FOCAL_RESCUE_MIN_BINS:
        block = arr[end - run + 1: end + 1]
        return float(block[_np.argmax(_np.abs(block))])
    return float(_np.mean(finite))


def gene_bin_window(start_bin: int, stop_bin: int, n_bins: int,
                    minimum: int = CNV_GENE_MIN_EVAL_BINS):
    """Clamp a gene's bin range to the track, widening it to ``minimum`` bins.

    Returns ``(start, stop)`` as a half-open slice. A gene shorter than the
    analysis bin would otherwise be judged on one bin; the window is grown
    symmetrically around it so the same noise protection applies to every gene.
    """
    start = max(int(start_bin), 0)
    stop = min(int(stop_bin), int(n_bins))
    if stop <= start:
        return start, stop
    want = max(int(minimum), 1)
    while (stop - start) < want:
        grew = False
        if start > 0:
            start -= 1
            grew = True
        if (stop - start) < want and stop < n_bins:
            stop += 1
            grew = True
        if not grew:
            break
    return start, stop


# CNV_SEGMENT_MIN_BINS keeps focal events (a four-bin CDKN2A deletion) callable.
CNV_SEGMENT_MIN_BINS = 3

#: How much of the bin-to-bin noise variance a split must explain to be kept.
#:
#: Lowered from 20 to 10 at the analysts' request: the line was averaging across
#: real steps and reading as a coarse level rather than following the profile.
#: On a chr1 track of 4,493 display bins this takes the line from 62 segments at
#: a 1.8 Mb median to 110 at 1.3 Mb.
#:
#: 10 rather than lower because ``CNV_SEGMENT_MERGE_SIGMAS`` below is what stops
#: the line stepping with noise, and that guard is left untouched: a split still
#: has to survive the merge test afterwards. Going to 5 with a weaker merge gave
#: 147 segments, where the line starts tracking scatter instead of structure.
#: It also matches ``DEFAULT_CNV_PENALTY_VALUE``, the penalty the calling
#: segmentation already uses, so the drawn line and the calls agree on how
#: readily a step is worth believing.
CNV_SEGMENT_PENALTY = 10.0

#: A jump between consecutive x positions larger than this multiple of the median
#: spacing is treated as a gap in coverage, and a segment may not cross it. The
#: check is on positions rather than on NaNs because callers may strip missing
#: bins before plotting: the chromosome pages do, which let a segment join two
#: clusters of bins 18 Mb apart across the chr1 heterochromatin and draw one level
#: over the whole span.
CNV_SEGMENT_GAP_SPACING_FACTOR = 3.0

#: How many standard errors two neighbouring segment levels must differ by before
#: the step between them is drawn. The binary splitter uses a fixed penalty with
#: no correction for how many split points it tried, so on a genome-wide track it
#: keeps hundreds of steps that noise alone explains — orders of magnitude more
#: segments than the underlying copy-number structure, with a median segment
#: length far shorter than any real event. Analysts read whole chromosome gains
#: and losses off this line, so a line that steps with the noise is read as
#: structure that is not there.
CNV_SEGMENT_MERGE_SIGMAS = 3.0

# A horizontal label landing this close to a reference line reads as sitting on
# it, so the label ladder starts beyond the line instead.
CNV_LABEL_LINE_CLEARANCE_FRAC = 0.6
# Lane spacing as a multiple of the label size: enough that the text plus its
# halo clears the lane below it.
CNV_LABEL_LINE_SPACING = 1.7

# Gutter geometry: lane 0 sits a full lane clear of the data band (a marker head
# can sit right on the band edge), and half a lane is left above the last lane.
GUTTER_FIRST_LANE_OFFSET = 1.0
GUTTER_TRAILING_PAD = 0.5

# Log2 axis window. A typical ONT CNV profile sits well inside +/-0.5, so the
# floor is kept tight: a wider minimum spends most of the panel on empty space and
# is what makes small gains and losses hard to separate.
CNV_LOG2_MIN_AXIS_SPAN = 0.6
CNV_LOG2_MAX_AXIS_SPAN = 4.0

# Per-chromosome plots use one fixed symmetric window so the pages can be compared
# with each other and with the methylation-array plots. Bins outside it are flagged
# at the panel edge rather than dropped. Users can change this from the CNV controls.
CNV_CHROMOSOME_AXIS_LOG2 = 2.0
# Expected copy number the fixed window is centred on when plotting ploidy.
CNV_PLOIDY_BASELINE = 2.0


def cnv_chromosome_axis_window(
    *,
    use_log: bool,
    span_log2: float = CNV_CHROMOSOME_AXIS_LOG2,
    baseline: float = CNV_PLOIDY_BASELINE,
) -> Tuple[float, float]:
    """Fixed per-chromosome axis window, in log2 or the equivalent ploidy range."""
    span = abs(float(span_log2))
    if use_log:
        return -span, span
    base = float(baseline) if baseline and baseline > 0 else CNV_PLOIDY_BASELINE
    return float(base * 2.0**-span), float(base * 2.0**span)


@dataclass(frozen=True)
class CnvAxisTicks:
    """Major/minor tick geometry for one copy-number axis."""

    lo: float
    hi: float
    major: float
    minor: float

    def major_ticks(self) -> List[float]:
        return cnv_axis_ticks(self.lo, self.hi, self.major)

    def minor_ticks(self) -> List[float]:
        return cnv_axis_ticks(self.lo, self.hi, self.minor)


def resolve_tick_interval(
    span: float,
    ladder: Sequence[float] = CNV_LOG2_TICK_LADDER,
    *,
    target_ticks: int = CNV_AXIS_TARGET_TICKS,
    max_ticks: int = CNV_AXIS_MAX_TICKS,
) -> float:
    """Smallest interval from ``ladder`` that keeps the label count readable."""
    span = float(abs(span))
    if not np.isfinite(span) or span <= 0:
        return float(ladder[0])
    for interval in ladder:
        if span / interval <= max(target_ticks, 2):
            return float(interval)
    # Nothing on the ladder is coarse enough: fall back to an even split.
    coarse = span / float(max(max_ticks, 2))
    return float(max(coarse, ladder[-1]))


def cnv_axis_ticks(lo: float, hi: float, interval: float) -> List[float]:
    """Tick values on ``interval`` multiples covering ``[lo, hi]``.

    Anchored on zero so the no-change line always carries a labelled tick.
    """
    lo = float(lo)
    hi = float(hi)
    interval = float(interval)
    if not np.isfinite(interval) or interval <= 0 or hi <= lo:
        return []
    first = np.ceil(lo / interval - 1e-9)
    last = np.floor(hi / interval + 1e-9)
    if last < first:
        return []
    steps = np.arange(first, last + 1, dtype=float)
    return [round(float(step * interval), 10) for step in steps]


def cnv_axis_tick_spec(
    lo: float,
    hi: float,
    *,
    use_log: bool,
    target_ticks: int = CNV_AXIS_TARGET_TICKS,
) -> CnvAxisTicks:
    """Resolve major/minor tick intervals for a copy-number axis window."""
    ladder = CNV_LOG2_TICK_LADDER if use_log else CNV_PLOIDY_TICK_LADDER
    major = resolve_tick_interval(
        float(hi) - float(lo),
        ladder,
        target_ticks=target_ticks,
    )
    minor = major / float(max(CNV_AXIS_MINOR_SUBDIVISIONS, 1))
    return CnvAxisTicks(lo=float(lo), hi=float(hi), major=major, minor=minor)


def snap_axis_window_to_ticks(
    lo: float,
    hi: float,
    interval: float,
    *,
    clamp_min: Optional[float] = None,
) -> Tuple[float, float]:
    """Round an axis window outwards to whole tick multiples.

    Keeps the top and bottom of the panel on a labelled gridline instead of an
    arbitrary data-driven value, which is what makes gains and losses easy to read
    off against the ticks.
    """
    interval = float(interval)
    if not np.isfinite(interval) or interval <= 0:
        return float(lo), float(hi)
    snapped_lo = float(np.floor(float(lo) / interval - 1e-9) * interval)
    snapped_hi = float(np.ceil(float(hi) / interval + 1e-9) * interval)
    if clamp_min is not None:
        snapped_lo = max(float(clamp_min), snapped_lo)
    if snapped_hi <= snapped_lo:
        snapped_hi = snapped_lo + interval
    return snapped_lo, snapped_hi


def cnv_reference_levels(
    *,
    use_log: bool,
    gain_threshold: Optional[float] = None,
    loss_threshold: Optional[float] = None,
    y_lo: Optional[float] = None,
    y_hi: Optional[float] = None,
) -> List[Tuple[float, str]]:
    """Horizontal reference levels as ``(value, kind)``.

    ``kind`` is ``baseline`` for the no-change line, ``threshold`` for the calling
    cut-offs, and ``ploidy`` for the integer copy-number guides in linear mode.
    """
    levels: List[Tuple[float, str]] = []
    if use_log:
        levels.append((0.0, "baseline"))
        for threshold in (gain_threshold, loss_threshold):
            if threshold is None or not np.isfinite(threshold):
                continue
            if abs(float(threshold)) < 1e-9:
                continue
            levels.append((float(threshold), "threshold"))
    else:
        levels.append((2.0, "baseline"))
        for ploidy in (1.0, 3.0, 4.0):
            levels.append((ploidy, "ploidy"))

    if y_lo is None and y_hi is None:
        return levels
    lo = -np.inf if y_lo is None else float(y_lo)
    hi = np.inf if y_hi is None else float(y_hi)
    return [(value, kind) for value, kind in levels if lo <= value <= hi]


def cnv_log2_deviation(
    cnv_val: float,
    *,
    use_log: bool,
    baseline: Optional[float] = None,
) -> Optional[float]:
    """Express a gene's copy number as a log2 deviation from normal.

    In log2 mode the value already is that deviation. In ploidy mode it is
    converted against the genome-wide baseline, so both scales can be judged
    against the same calling cut-offs.
    """
    value = float(cnv_val)
    if not np.isfinite(value):
        return None
    if use_log:
        return value
    if baseline is None or not np.isfinite(baseline) or baseline <= 0 or value <= 0:
        return None
    return float(np.log2(value / float(baseline)))


def gene_crosses_cutoff(
    cnv_val: float,
    *,
    gain_threshold: float,
    loss_threshold: float,
    use_log: bool = True,
    baseline: Optional[float] = None,
) -> bool:
    """True when a gene's copy number crosses the gain/loss calling cut-off.

    This is deliberately the same test that colours the points, fills the events
    table and draws the cut-off lines, and it is taken against the genome-wide
    baseline. Judging a gene against its *own* chromosome instead — as a
    standard-deviation rule does — cancels out whole-chromosome and arm-level
    events, so a gene sitting on a gained chromosome looks perfectly normal.
    """
    deviation = cnv_log2_deviation(cnv_val, use_log=use_log, baseline=baseline)
    if deviation is None:
        return False
    if np.isfinite(gain_threshold) and deviation > float(gain_threshold):
        return True
    if np.isfinite(loss_threshold) and deviation < float(loss_threshold):
        return True
    return False


def _robust_sigma(values: np.ndarray) -> float:
    """Bin-to-bin noise estimated from successive differences.

    Successive differencing is used so a real copy-number step does not inflate
    the estimate the way the plain standard deviation of the track would.
    """
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return float(1.4826 * mad / np.sqrt(2.0))


def _best_binary_split(
    segment: np.ndarray,
    min_size: int,
) -> Tuple[Optional[int], float]:
    """Index and sum-of-squares gain of the best single split of ``segment``."""
    n = segment.size
    if n < 2 * min_size:
        return None, 0.0
    cumulative = np.cumsum(segment)
    total = float(cumulative[-1])
    cuts = np.arange(min_size, n - min_size + 1)
    left_sum = cumulative[cuts - 1]
    left_n = cuts.astype(float)
    right_n = float(n) - left_n
    gain = (
        left_sum**2 / left_n
        + (total - left_sum) ** 2 / right_n
        - total**2 / float(n)
    )
    best = int(np.argmax(gain))
    return int(cuts[best]), float(gain[best])


def cnv_segment_bounds(
    values: Sequence[float],
    *,
    min_size: int = CNV_SEGMENT_MIN_BINS,
    penalty: float = CNV_SEGMENT_PENALTY,
) -> List[Tuple[int, int]]:
    """Split a CNV track into piecewise-constant segments.

    Recursive binary segmentation on the sum-of-squares criterion — the same idea
    as the circular binary segmentation behind the flat step lines on methylation
    array CNV plots. A split is kept only when it explains more than ``penalty``
    times the bin-to-bin noise variance, so noise alone does not create steps.

    Indices are into the array as given, so callers can map back to positions.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return []
    sigma = _robust_sigma(arr)
    # With no measurable noise, any split is "significant"; fall back to a floor
    # so a flat track stays a single segment.
    threshold = max(float(penalty) * sigma**2, 1e-12)

    bounds: List[Tuple[int, int]] = []
    stack: List[Tuple[int, int]] = [(0, n)]
    while stack:
        start, end = stack.pop()
        split, gain = _best_binary_split(arr[start:end], int(min_size))
        if split is None or gain <= threshold:
            bounds.append((start, end))
            continue
        stack.append((start + split, end))
        stack.append((start, start + split))
    bounds.sort()
    return _merge_indistinguishable_segments(arr, bounds, sigma)


def _merge_indistinguishable_segments(
    arr: np.ndarray,
    bounds: List[Tuple[int, int]],
    sigma: float,
    *,
    sigmas: float = CNV_SEGMENT_MERGE_SIGMAS,
) -> List[Tuple[int, int]]:
    """Join neighbouring segments whose levels the data cannot tell apart.

    The splitter asks "does a split here help?" at every position and keeps any
    that clears a fixed bar, with no allowance for the number of positions tried.
    This asks the complementary question of each surviving step — "is this
    difference larger than the uncertainty on the two means?" — and removes it
    when it is not. That is what the array's significance test achieves in effect.

    Merging repeats until nothing more can be joined, so a run of small steps
    collapses into one level rather than being halved.
    """
    if sigma <= 0 or len(bounds) < 2:
        return bounds
    merged = list(bounds)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        out: List[Tuple[int, int]] = [merged[0]]
        for start, end in merged[1:]:
            p_start, p_end = out[-1]
            a = arr[p_start:p_end]
            b = arr[start:end]
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size == 0 or b.size == 0:
                out.append((start, end))
                continue
            # Standard error of the difference between the two segment means.
            se = sigma * np.sqrt(1.0 / a.size + 1.0 / b.size)
            if abs(float(a.mean()) - float(b.mean())) < sigmas * se:
                out[-1] = (p_start, end)
                changed = True
            else:
                out.append((start, end))
        merged = out
    return merged


def cnv_segment_line(
    values: Sequence[float],
    *,
    min_size: int = CNV_SEGMENT_MIN_BINS,
    penalty: float = CNV_SEGMENT_PENALTY,
    min_points: int = CNV_TREND_MIN_POINTS,
) -> Optional[np.ndarray]:
    """Piecewise-constant segment level per bin, or None when the track is too short.

    Each segment is drawn flat at its **mean**, giving the straight step line of a
    methylation-array CNV plot rather than a rolling average that wanders with the
    noise. NaN bins stay NaN so the line breaks across gaps.

    The mean is used because every call ROBIN makes uses it — arm and
    whole-chromosome events from the arm mean, the regional table from the band
    mean, gene states from the gene-window mean. Drawing the median instead meant
    the level on the figure and the number in the table were different statistics,
    so they could disagree on the same region.
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if arr.size < int(min_points) or int(finite.sum()) < int(min_points):
        return None

    # Segment each contiguous run of finite bins on its own. Compacting the whole
    # track first makes bins on either side of a gap adjacent, so a segment could
    # group bins megabases apart and its level was then painted over regions whose
    # data looked nothing like it: on chr1 a span at 121.6 Mb was drawn at -4.00
    # where the bins there read -0.32, and one segment was drawn at two places
    # 18 Mb apart. A gap in coverage is not evidence that the two sides belong
    # together.
    out = np.full(arr.shape, np.nan, dtype=float)
    idx = np.flatnonzero(finite)
    if idx.size:
        # Split the finite positions wherever they are not consecutive.
        breaks = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, breaks):
            if run.size == 0:
                continue
            piece = arr[run]
            if run.size < int(min_size):
                # Too short to segment; report it flat at its own mean rather
                # than borrowing a level from across the gap.
                out[run] = float(np.mean(piece))
                continue
            for start, end in cnv_segment_bounds(
                piece, min_size=min_size, penalty=penalty
            ):
                if end <= start:
                    continue
                out[run[start:end]] = float(np.mean(piece[start:end]))
    if not np.isfinite(out).any():
        return None
    return out


def _positional_gap_cuts(
    xs: np.ndarray,
    *,
    factor: float = CNV_SEGMENT_GAP_SPACING_FACTOR,
) -> List[float]:
    """Split points where the x positions jump, i.e. where coverage is missing.

    Bins that are far apart on the genome are not neighbours however adjacent
    they are in the array, so a segment must not average across the jump.
    """
    if xs.size < 3:
        return []
    steps = np.diff(xs)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return []
    typical = float(np.median(steps))
    if typical <= 0:
        return []
    jumps = np.flatnonzero(np.diff(xs) > factor * typical)
    # Cut midway across each jump so both sides keep their own bins.
    return [float((xs[i] + xs[i + 1]) / 2.0) for i in jumps]


def cnv_arm_mean_spans(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    split_at: Sequence[float] = (),
) -> List[Tuple[float, float, float]]:
    """One span per arm, drawn at the arm's mean — the number the table reports.

    Used for the genome-wide figure, where analysts read whole-chromosome gains
    and losses off the line. Segmenting that track finely is not informative:
    ROBIN has no working reference normalisation, so it carries real, highly
    significant local structure that is coverage variation rather than copy
    number — measured on one chromosome, the surviving steps ran at a median
    10.9 sigma. A detailed line there draws artefact with conviction, and it can
    disagree with the arm table printed alongside it.

    The level is the plain mean of the finite bins in the arm, which is the
    statistic ``analyze_chromosome_arms`` calls on, so the figure and the arm and
    whole-chromosome table cannot show different levels for the same arm.
    """
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    if xs.size == 0 or xs.size != ys.size:
        return []
    cuts = sorted(
        c for c in (float(s) for s in split_at)
        if np.isfinite(c) and xs.min() < c < xs.max()
    )
    edges = [-np.inf] + cuts + [np.inf]
    spans: List[Tuple[float, float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        piece = (xs >= lo) & (xs < hi)
        if not piece.any():
            continue
        values = ys[piece]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        px = xs[piece]
        spans.append((float(px.min()), float(px.max()), float(finite.mean())))
    return spans


def cnv_segment_spans(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    min_size: int = CNV_SEGMENT_MIN_BINS,
    penalty: float = CNV_SEGMENT_PENALTY,
    min_points: int = CNV_TREND_MIN_POINTS,
    split_at: Sequence[float] = (),
) -> List[Tuple[float, float, float]]:
    """Segments as ``(x_start, x_end, level)`` spans.

    Returned as separate spans rather than one continuous line so each segment can
    be drawn as its own horizontal bar. A connected step line has to draw a
    vertical riser between levels, and a short run of near-zero-coverage bins — a
    centromere, an unmappable region — then produces a full-height spike off the
    segment line.

    ``split_at`` lists x positions a segment may not cross — the centromere, in
    practice. Arm and whole-chromosome events are called per arm, so a line that
    averaged a gained q arm together with a flat p arm would show a level that no
    table reports and that belongs to neither arm.
    """
    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    cuts = [c for c in (float(s) for s in split_at)
            if np.isfinite(c) and xs.size and xs.min() < c < xs.max()]
    cuts.extend(_positional_gap_cuts(xs))
    if cuts:
        spans: List[Tuple[float, float, float]] = []
        edges = [-np.inf] + sorted(cuts) + [np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            piece = (xs >= lo) & (xs < hi)
            if not piece.any():
                continue
            spans.extend(
                cnv_segment_spans(
                    xs[piece], ys[piece], min_size=min_size, penalty=penalty,
                    min_points=min_points,
                )
            )
        return spans

    levels = cnv_segment_line(
        y_values, min_size=min_size, penalty=penalty, min_points=min_points
    )
    if levels is None:
        return []
    xs = np.asarray(x_values, dtype=float)
    if xs.size != levels.size or xs.size == 0:
        return []

    spans: List[Tuple[float, float, float]] = []
    start = 0
    for idx in range(1, levels.size + 1):
        ended = idx == levels.size
        if not ended:
            previous, current = levels[idx - 1], levels[idx]
            same = (
                (np.isnan(previous) and np.isnan(current))
                or previous == current
            )
            if same:
                continue
        if not np.isnan(levels[start]):
            spans.append(
                (float(xs[start]), float(xs[idx - 1]), float(levels[start]))
            )
        start = idx
    return spans


def cnv_segment_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    min_size: int = CNV_SEGMENT_MIN_BINS,
    penalty: float = CNV_SEGMENT_PENALTY,
    min_points: int = CNV_TREND_MIN_POINTS,
) -> List[List[Optional[float]]]:
    """Segment bars as ECharts ``[x, y]`` pairs, ``None`` separating each bar."""
    spans = cnv_segment_spans(
        x_values,
        y_values,
        min_size=min_size,
        penalty=penalty,
        min_points=min_points,
    )
    points: List[List[Optional[float]]] = []
    for x_start, x_end, level in spans:
        points.append([x_start, level])
        points.append([x_end, level])
        # Break the polyline so no riser is drawn to the next segment.
        points.append([x_end, None])
    return points


def cnv_trend_window(n_points: int, *, target_segments: int = 40) -> int:
    """Rolling-median window (in bins) for a track of ``n_points`` bins."""
    if n_points <= 0:
        return 1
    window = int(round(n_points / float(max(target_segments, 1))))
    window = max(window, 5)
    window = min(window, max(n_points, 1))
    # Odd windows keep the centred median symmetric around each bin.
    if window % 2 == 0:
        window += 1
    return int(min(window, max(n_points, 1)))


def cnv_trend_line(
    values: Sequence[float],
    *,
    window: Optional[int] = None,
    target_segments: int = 40,
    min_points: int = CNV_TREND_MIN_POINTS,
) -> Optional[np.ndarray]:
    """Centred rolling median over a CNV track, or None when it is too short.

    NaN bins (unmappable / filtered) stay NaN in the output so the trend breaks
    across gaps instead of drawing a straight line through them.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < int(min_points):
        return None
    finite = np.isfinite(arr)
    if int(finite.sum()) < int(min_points):
        return None
    resolved_window = int(
        window if window else cnv_trend_window(arr.size, target_segments=target_segments)
    )
    if resolved_window < 1:
        return None
    series = pd.Series(arr, dtype=float)
    trend = series.rolling(
        resolved_window,
        center=True,
        min_periods=max(1, resolved_window // 3),
    ).median()
    out = trend.to_numpy(dtype=float)
    # Never invent a trend where there was no data.
    out[~finite] = np.nan
    if not np.isfinite(out).any():
        return None
    return out


def cnv_trend_points(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    window: Optional[int] = None,
    target_segments: int = 40,
    min_points: int = CNV_TREND_MIN_POINTS,
) -> List[List[Optional[float]]]:
    """Trend line as ECharts ``[x, y]`` pairs, with ``None`` breaking the line."""
    trend = cnv_trend_line(
        y_values,
        window=window,
        target_segments=target_segments,
        min_points=min_points,
    )
    if trend is None:
        return []
    xs = np.asarray(x_values, dtype=float)
    if xs.size != trend.size:
        return []
    return [
        [float(x), (float(y) if np.isfinite(y) else None)]
        for x, y in zip(xs.tolist(), trend.tolist())
    ]


def horizontal_label_proximity_frac(
    labels: Sequence[str],
    *,
    font_size: float,
    panel_width_pt: float,
    char_width_ratio: float = 0.62,
    min_frac: float = 0.008,
    max_frac: float = 0.2,
) -> float:
    """Fraction of the panel width a horizontal gene label occupies.

    A rotated label is narrow, so neighbouring genes rarely collide. A horizontal
    one is as wide as its name, so how far apart two genes must be before they can
    share a line depends on the text length and the font size.
    """
    longest = max((len(str(label)) for label in labels), default=1)
    width_pt = float(longest) * float(char_width_ratio) * float(font_size)
    frac = width_pt / max(float(panel_width_pt), 1.0)
    return float(min(max(frac, float(min_frac)), float(max_frac)))


def horizontal_lane_height(
    *,
    y_span: float,
    panel_height_pt: float,
    font_size: float,
    line_spacing: float = CNV_LABEL_LINE_SPACING,
    min_frac: float = 0.02,
) -> float:
    """Data-space spacing between stacked label lanes.

    Derived from the label size and the panel's real height in points, so the gap
    between two stacked names is a constant number of points however the user
    changes the Y range. A fixed fraction of the span cannot do this: it ignores
    the font size, and at a small label size the lanes end up closer together than
    the text is tall.
    """
    span = max(float(y_span), 1e-9)
    if panel_height_pt <= 0:
        return span * float(min_frac)
    points_per_unit = float(panel_height_pt) / span
    height = (float(font_size) * float(line_spacing)) / points_per_unit
    return float(max(height, span * float(min_frac)))


def clear_reference_levels(
    y: float,
    *,
    side: str,
    reference_levels: Sequence[float],
    clearance: float,
    max_steps: int = 6,
) -> float:
    """Push a label off any horizontal reference line it would sit on.

    A gene name printed along the cut-off line is effectively unreadable, so the
    label ladder starts beyond the nearest line rather than on it. Only ever
    pushed further from the marker, so labels keep their order.
    """
    if not reference_levels or clearance <= 0:
        return float(y)
    out = float(y)
    for _ in range(int(max_steps)):
        clashing = [
            float(level)
            for level in reference_levels
            if abs(float(level) - out) < clearance
        ]
        if not clashing:
            break
        out = (
            min(clashing) - clearance
            if str(side) == "below"
            else max(clashing) + clearance
        )
    return out


def horizontal_label_placements(
    x_positions: Sequence[float],
    labels: Sequence[str],
    sides: Sequence[str],
    marker_ys: Sequence[float],
    *,
    x_span: Optional[float],
    lane_height: float,
    font_size: float,
    panel_width_pt: float,
    y_lo: Optional[float] = None,
    y_hi: Optional[float] = None,
    reference_levels: Sequence[float] = (),
) -> List[dict]:
    """Stack horizontal gene labels into lanes so overlapping names step apart.

    Labels stay inside the existing axis window — the panel is never grown for
    them — so on a crowded panel they sit over the bin cloud. That is the inherent
    trade-off of horizontal labelling, and why rotated is the default.
    """
    n = len(x_positions)
    if n == 0:
        return []
    proximity = horizontal_label_proximity_frac(
        labels, font_size=font_size, panel_width_pt=panel_width_pt
    )
    lanes = assign_label_lanes(
        x_positions, sides, x_span=x_span, x_proximity_frac=proximity
    )
    # A lane index only separates two names if both step from the *same* height.
    # Measuring each ladder from its own marker let genes in different lanes
    # land on top of each other whenever their markers sat at different values —
    # which is exactly what happens on a gene-dense chromosome.
    clusters = label_cluster_ids(
        x_positions, sides, x_span=x_span, x_proximity_frac=proximity
    )
    cluster_bases = _cluster_ladder_bases(
        clusters,
        sides,
        marker_ys,
        lanes,
        lane_height=lane_height,
        y_lo=y_lo,
        y_hi=y_hi,
    )

    # Lane 0 starts clear of any reference line, then the ladder steps from
    # there — so the spacing between stacked names is preserved.
    clearance = float(lane_height) * CNV_LABEL_LINE_CLEARANCE_FRAC
    placements: List[dict] = []
    for lane, side, marker_y, cluster_base in zip(
        lanes, sides, marker_ys, cluster_bases
    ):
        base = (
            float(cluster_base) + float(lane_height)
            if str(side) != "below"
            else float(cluster_base) - float(lane_height)
        )
        base = clear_reference_levels(
            base,
            side=side,
            reference_levels=reference_levels,
            clearance=clearance,
        )
        offset = int(lane) * float(lane_height)
        label_y = base + offset if str(side) != "below" else base - offset
        if y_lo is not None:
            label_y = max(label_y, float(y_lo))
        if y_hi is not None:
            label_y = min(label_y, float(y_hi))
        placements.append({"y": float(label_y), "side": str(side), "lane": int(lane)})
    return placements


def _cluster_ladder_bases(
    clusters: Sequence[int],
    sides: Sequence[str],
    marker_ys: Sequence[float],
    lanes: Sequence[int],
    *,
    lane_height: float,
    y_lo: Optional[float],
    y_hi: Optional[float],
) -> List[float]:
    """One ladder base per label, shared across each colliding group.

    The base is the outermost marker in the group, so no name sits below the
    marker it belongs to. Where the resulting ladder would run off the end of the
    axis, the whole group is slid back so it fits: clamping each lane to the axis
    edge instead would pile the top of the ladder into a single line.
    """
    n = len(clusters)
    if n == 0:
        return []
    step = float(lane_height)
    outermost: dict[int, float] = {}
    top_lane: dict[int, int] = {}
    for key, side, marker_y, lane in zip(clusters, sides, marker_ys, lanes):
        y = float(marker_y)
        if key not in outermost:
            outermost[key] = y
            top_lane[key] = int(lane)
            continue
        outermost[key] = (
            max(outermost[key], y) if str(side) != "below" else min(outermost[key], y)
        )
        top_lane[key] = max(top_lane[key], int(lane))

    side_of: dict[int, str] = {}
    for key, side in zip(clusters, sides):
        side_of.setdefault(key, str(side))

    bases: dict[int, float] = {}
    for key, base in outermost.items():
        span = (top_lane[key] + 1) * step
        if side_of[key] != "below":
            if y_hi is not None and base + span > float(y_hi):
                # Never below the group's own markers, even if it still overflows.
                base = max(float(y_hi) - span, base - span)
        elif y_lo is not None and base - span < float(y_lo):
            base = min(float(y_lo) + span, base + span)
        bases[key] = base
    return [bases[key] for key in clusters]


def label_cluster_ids(
    x_positions: Sequence[float],
    sides: Sequence[str],
    *,
    x_span: Optional[float] = None,
    x_proximity_frac: float = 0.028,
) -> List[int]:
    """Group labels that sit close enough on x for their names to collide.

    Labels in one group must step from a single shared base, or the lane index
    does not guarantee they are separated: a lane-2 name hanging off a low marker
    can land at the same height as a lane-0 name off a high one.
    """
    n = len(x_positions)
    if n == 0:
        return []
    xs = [float(x) for x in x_positions]
    span = float(x_span) if x_span else (max(xs) - min(xs))
    gap = float(x_proximity_frac) * span if span > 0 else 0.0

    order = sorted(range(n), key=lambda i: (str(sides[i]), xs[i]))
    cluster_of = [0] * n
    cluster_id = -1
    prev_side: Optional[str] = None
    prev_x: Optional[float] = None
    for index in order:
        side = str(sides[index])
        x = xs[index]
        if prev_side != side or prev_x is None or (x - prev_x) > gap:
            cluster_id += 1
        cluster_of[index] = cluster_id
        prev_side, prev_x = side, x

    return cluster_of


def assign_label_lanes(
    x_positions: Sequence[float],
    sides: Sequence[str],
    *,
    x_span: Optional[float] = None,
    x_proximity_frac: float = 0.028,
    max_lanes: int = 12,
) -> List[int]:
    """Greedy lane packing for gene labels sharing a gutter.

    Labels close together on the x-axis are pushed into successive lanes so they
    stack instead of overprinting. Returns one lane index per input point.

    ``x_span`` should be the plotted axis width. Falling back to the spread of the
    labels themselves under-separates a tight cluster of genes, because what
    decides whether two badges collide is how far apart they are *on screen*.
    """
    n = len(x_positions)
    if n == 0:
        return []
    xs = [float(x) for x in x_positions]
    if x_span is not None and float(x_span) > 0:
        span = float(x_span)
    else:
        span = (max(xs) - min(xs)) if n > 1 else 1.0
    proximity = max(span * float(x_proximity_frac), 1e-9)
    order = sorted(range(n), key=lambda idx: xs[idx])
    lanes = [0] * n
    occupied: List[Tuple[float, str, int]] = []
    for idx in order:
        x_pos = xs[idx]
        side = str(sides[idx])
        lane = 0
        while any(
            abs(x_pos - other_x) < proximity
            and side == other_side
            and lane == other_lane
            for other_x, other_side, other_lane in occupied
        ):
            lane += 1
            if lane >= int(max_lanes):
                lane = int(max_lanes) - 1
                break
        occupied.append((x_pos, side, lane))
        lanes[idx] = lane
    return lanes


def lane_counts(lanes: Sequence[int], sides: Sequence[str]) -> Tuple[int, int]:
    """Number of lanes needed in the top (``above``) and bottom (``below``) gutters."""
    top = 0
    bottom = 0
    for lane, side in zip(lanes, sides):
        if str(side) == "below":
            bottom = max(bottom, int(lane) + 1)
        else:
            top = max(top, int(lane) + 1)
    return top, bottom


def clamp_band_for_markers(
    data_lo: float,
    data_hi: float,
    marker_values: Sequence[float],
    *,
    max_expansion_frac: float = 0.15,
    min_expansion: float = 0.25,
) -> Tuple[float, float]:
    """Widen a CNV band to reach gene markers, but only so far.

    A panel target's normalised coverage can sit far outside the CNV cloud — a
    homozygously deleted gene at 4x against a 78x mean lands several log2 units
    down. Letting that set the axis squeezes the whole profile into a couple of
    ticks, so markers may stretch the band by a bounded amount and anything beyond
    is drawn clamped at the edge.
    """
    data_lo = float(data_lo)
    data_hi = float(data_hi)
    values = [
        float(v) for v in marker_values if v is not None and np.isfinite(float(v))
    ]
    if not values:
        return data_lo, data_hi
    span = max(data_hi - data_lo, 1e-6)
    allow = max(span * float(max_expansion_frac), float(min_expansion))
    lo = min(data_lo, max(min(values), data_lo - allow))
    hi = max(data_hi, min(max(values), data_hi + allow))
    return float(lo), float(hi)


def solve_lane_height(
    band_span: float,
    *,
    total_lanes: float,
    lane_height_frac: float,
    max_gutter_frac: float = 0.55,
) -> float:
    """Lane height that ends up ``lane_height_frac`` of the *final* axis span.

    Reserving gutters grows the axis, which changes what fraction of it a
    fixed-height label occupies — so solve for the fixed point instead of sizing
    lanes against the pre-expansion band. ``lane_height_frac`` is the label height
    as a fraction of the panel, so labels stay legible on a tall genome-wide panel
    and on a short per-chromosome one alike.
    """
    band = max(float(band_span), 1e-6)
    frac = max(float(lane_height_frac), 1e-6)
    lanes = max(float(total_lanes), 0.0)
    if lanes <= 0:
        return band * frac
    denom = 1.0 - frac * lanes
    if denom <= (1.0 - float(max_gutter_frac)):
        # Too many lanes to honour the requested height; cap the gutters so the
        # labels crowd rather than swallowing the panel the CNV data needs.
        return band * float(max_gutter_frac) / lanes
    return band * frac / denom


def reserve_label_gutters(
    data_lo: float,
    data_hi: float,
    *,
    top_lanes: int,
    bottom_lanes: int,
    lane_height_frac: float = 0.055,
    min_lane_height: float = 0.0,
    clamp_min: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Grow an axis window so gene labels sit outside the data band.

    Returns ``(y_lo, y_hi, lane_height)``. Labels are then drawn inside the
    reserved gutters rather than on top of the CNV bin cloud.
    """
    data_lo = float(data_lo)
    data_hi = float(data_hi)
    band = max(data_hi - data_lo, 1e-6)
    top_lanes = int(top_lanes)
    bottom_lanes = int(bottom_lanes)
    total_lanes = (
        top_lanes + GUTTER_TRAILING_PAD if top_lanes > 0 else 0.0
    ) + (bottom_lanes + GUTTER_TRAILING_PAD if bottom_lanes > 0 else 0.0)
    lane_height = max(
        solve_lane_height(
            band,
            total_lanes=total_lanes,
            lane_height_frac=lane_height_frac,
        ),
        float(min_lane_height),
    )
    top_pad = (
        (top_lanes + GUTTER_TRAILING_PAD) * lane_height if top_lanes > 0 else 0.0
    )
    bottom_pad = (
        (bottom_lanes + GUTTER_TRAILING_PAD) * lane_height if bottom_lanes > 0 else 0.0
    )
    y_hi = data_hi + top_pad
    y_lo = data_lo - bottom_pad
    if clamp_min is not None:
        y_lo = max(float(clamp_min), y_lo)
    return float(y_lo), float(y_hi), float(lane_height)


def gutter_label_y(
    *,
    side: str,
    lane: int,
    data_lo: float,
    data_hi: float,
    lane_height: float,
) -> float:
    """Y coordinate for a gene label in the reserved gutter."""
    offset = (int(lane) + GUTTER_FIRST_LANE_OFFSET) * float(lane_height)
    if str(side) == "below":
        return float(data_lo) - offset
    return float(data_hi) + offset
