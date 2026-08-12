"""Hiding unmappable bins from the CNV plots.

The "rain" below the profile is bins the control profile cannot cover: ROBIN
normalises against ``HG01280_control_new.pkl``, so where that control is empty
the divisor is zero and the log2 ratio explodes to -3..-5. Those bins are hidden
from the plots only - the track, the segmentation and the reported calls keep
every value.

The clinically important tests are the ones asserting no panel target is ever
hidden. Two earlier approaches failed exactly there: ``cenSatRegions.bed`` spans
PDGFRA, KIT, KDR, NF1 and SUZ12, and a bare control threshold hides EZHIP and
TBL1X. Target protection is therefore structural, not a matter of picking a
lenient threshold.
"""

import glob
import os

import numpy as np
import pytest

from robin import resources
from robin.analysis.cnv_regional import (
    UNMAPPABLE_CONTROL_FRACTION,
    load_control_mappability,
    load_protected_target_intervals,
    unmappable_bin_mask,
)

PLOT_BIN = 50_000
CDKN2A_START, CDKN2B_END = 21_967_752, 22_009_312  # hg38


def _resource_dir():
    return os.path.dirname(os.path.abspath(resources.__file__))


def _packaged_targets():
    for bed in sorted(glob.glob(os.path.join(_resource_dir(), "*_panel_name_uniq.bed"))):
        with open(bed) as handle:
            for line in handle:
                if line.startswith(("#", "track", "browser")):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    start, end = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                yield os.path.basename(bed), parts[0], start, end, (
                    parts[3] if len(parts) > 3 else "?"
                )


def _bin_centres_overlapping(start, end):
    """Centres of the plot bins that actually overlap [start, end)."""
    half = PLOT_BIN / 2.0
    centres = np.arange((start // PLOT_BIN) * PLOT_BIN + half, end + PLOT_BIN,
                        PLOT_BIN, dtype=float)
    return centres[(centres - half < end) & (start < centres + half)]


def _hidden(contig, start, end, **kwargs):
    """Is any plot bin overlapping [start, end) hidden?"""
    centres = _bin_centres_overlapping(start, end)
    if len(centres) == 0:
        return np.zeros(0, dtype=bool)
    return unmappable_bin_mask(contig, centres, PLOT_BIN, **kwargs)


def test_control_profile_loads_on_a_1kb_grid():
    profile = load_control_mappability()
    assert len(profile) >= 23
    _, bin_size = profile["chr9"]
    assert bin_size == 1000


def test_no_packaged_panel_target_is_ever_hidden():
    """The guard. Any change that hides a target must fail here."""
    hidden = [
        f"{name} ({bed} {chrom}:{start}-{end})"
        for bed, chrom, start, end, name in _packaged_targets()
        if _hidden(chrom, start, end).any()
    ]
    assert not hidden, f"panel targets hidden: {hidden[:10]}"


@pytest.mark.parametrize("gene,chrom,start,end", [
    ("CDKN2A/B", "chr9", CDKN2A_START, CDKN2B_END),
    ("EZHIP", "chrX", 103_691_000, 103_698_000),
    ("TBL1X", "chrX", 9_463_000, 9_760_000),
])
def test_named_genes_stay_visible(gene, chrom, start, end):
    """Genes an earlier threshold-only approach would have hidden."""
    assert not _hidden(chrom, start, end).any(), gene


@pytest.mark.parametrize("gene,chrom,start,end", [
    ("U2AF1", "chr21", 43_092_956, 43_107_570),
    ("CRLF2", "chrX", 1_190_490, 1_212_723),
    ("P2RY8", "chrX", 1_462_581, 1_537_185),
    ("RSPH10B", "chr7", 5_925_550, 5_970_689),
])
def test_target_protection_is_doing_real_work(gene, chrom, start, end):
    """These sit in unmappable bins; only explicit protection keeps them drawn."""
    assert _hidden(chrom, start, end, protect_targets=False).any(), f"{gene} premise"
    assert not _hidden(chrom, start, end).any(), f"{gene} not protected"


def test_centromeric_bins_are_hidden():
    """chr9 pericentromeric rain, well clear of any target."""
    positions = np.arange(62_000_000, 66_000_000, PLOT_BIN, dtype=float)
    assert unmappable_bin_mask("chr9", positions, PLOT_BIN).mean() > 0.5


def test_ordinary_sequence_is_not_hidden():
    positions = np.arange(150_000_000, 160_000_000, PLOT_BIN, dtype=float)
    assert not unmappable_bin_mask("chr1", positions, PLOT_BIN).any()


def test_unknown_contig_hides_nothing():
    positions = np.arange(0, 5_000_000, PLOT_BIN, dtype=float)
    assert not unmappable_bin_mask("chrZZ", positions, PLOT_BIN).any()


def test_missing_control_hides_nothing(monkeypatch):
    """Fail open: no control profile means show everything."""
    from robin.analysis import cnv_regional

    load_control_mappability.cache_clear()
    monkeypatch.setattr(cnv_regional, "_resource_path", lambda name: "/nonexistent")
    try:
        assert load_control_mappability() == {}
        positions = np.arange(62_000_000, 66_000_000, PLOT_BIN, dtype=float)
        assert not unmappable_bin_mask("chr9", positions, PLOT_BIN).any()
    finally:
        load_control_mappability.cache_clear()


def test_zero_bin_width_hides_nothing():
    positions = np.arange(62_000_000, 63_000_000, PLOT_BIN, dtype=float)
    assert not unmappable_bin_mask("chr9", positions, 0).any()


def test_threshold_is_monotonic():
    positions = np.arange(60_000_000, 70_000_000, PLOT_BIN, dtype=float)
    lenient = unmappable_bin_mask("chr9", positions, PLOT_BIN, threshold=0.9).sum()
    strict = unmappable_bin_mask("chr9", positions, PLOT_BIN, threshold=0.2).sum()
    assert strict >= lenient


def test_default_threshold_is_half():
    assert UNMAPPABLE_CONTROL_FRACTION == 0.5


def test_protected_intervals_are_merged():
    for chrom, intervals in load_protected_target_intervals().items():
        assert intervals == tuple(sorted(intervals)), chrom
        for (_, a_end), (b_start, _) in zip(intervals, intervals[1:]):
            assert a_end < b_start, f"{chrom} not merged"


def test_report_plots_hide_bins_by_default():
    import inspect

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    parameters = inspect.signature(iter_CNV_chromosome_figures).parameters
    assert parameters["hide_unmappable_bands"].default is True


def _gui_series(hide):
    from robin.gui.components.cnv import _build_cnv_track_scatter_series

    track = {"chr9": np.zeros(140_000, dtype=float)}  # 1 kb bins across chr9
    series = _build_cnv_track_scatter_series(
        track,
        selected="chr9",
        binw_analysis=1_000,
        plot_bin_width=PLOT_BIN,
        chrom_palette=["#000000"],
        hide_unmappable_bands=hide,
    )
    return [point[0] for point in series[0]["data"]]


def test_gui_hides_centromeric_bins():
    """Most of the pericentromeric rain goes; mappable bins there still draw."""
    before = [x for x in _gui_series(hide=False) if 62_000_000 <= x < 66_000_000]
    after = [x for x in _gui_series(hide=True) if 62_000_000 <= x < 66_000_000]
    assert len(after) < len(before) / 2
    # CDKN2A/B is nowhere near it and must still be drawn.
    assert any(abs(x - CDKN2A_START) < PLOT_BIN for x in _gui_series(hide=True))


def test_gui_keeps_everything_when_disabled():
    assert len(_gui_series(hide=False)) > len(_gui_series(hide=True))


def test_gui_hiding_is_display_only():
    """The caller's track is never mutated."""
    from robin.gui.components.cnv import _build_cnv_track_scatter_series

    track = {"chr9": np.zeros(140_000, dtype=float)}
    before = track["chr9"].copy()
    _build_cnv_track_scatter_series(
        track,
        selected="chr9",
        binw_analysis=1_000,
        plot_bin_width=PLOT_BIN,
        chrom_palette=["#000000"],
    )
    assert np.array_equal(track["chr9"], before)
