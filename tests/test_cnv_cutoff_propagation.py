"""The review cut-off must reach every output that reports a copy-number call.

The Cut-off control is a single number a reviewer sets. If any one consumer
keeps calling at the configured default, a report contradicts itself: a figure
drawn at one threshold above a table called at another, with nothing to say
which is which. These tests pin each consumer to the cut-off it was given.
"""

import numpy as np
import pandas as pd
import pytest

from robin.analysis.cnv_classification import detect_cnv_events
from robin.analysis.cnv_regional import (
    CNV_NO_EVENT_LABEL,
    analyze_cytoband_cnv,
    compute_cnv_load,
    compute_target_gene_cnv_states,
    load_centromere_bed,
    load_cytobands_bed,
)
from robin.classification_config import resolve_cnv_thresholds


BIN_WIDTH = 1_000_000


def _flat_track(level: float, chrom: str = "chr1", bins: int = 250) -> dict:
    return {chrom: np.full(bins, float(level))}


def test_resolver_replaces_the_calling_thresholds_symmetrically():
    assert resolve_cnv_thresholds("chr1", "XY") == (0.3, -0.3)
    assert resolve_cnv_thresholds("chr1", "XY", 0.35) == (0.35, -0.35)
    # A single reviewer-chosen number cannot carry per-contig expectations, so it
    # applies to every contig rather than only the autosomes.
    assert resolve_cnv_thresholds("chrX", "XY", 0.35) == (0.35, -0.35)
    assert resolve_cnv_thresholds("chrY", "Female", 0.35) == (0.35, -0.35)
    # Absent or meaningless overrides fall back to the configured thresholds.
    for empty in (None, 0, 0.0, "", "nonsense"):
        assert resolve_cnv_thresholds("chr1", "XY", empty) == (0.3, -0.3)


@pytest.mark.parametrize(
    "level, cutoff, expect_called",
    [
        (0.32, None, True),    # past the 0.3 default
        (0.32, 0.35, False),   # a stricter cut-off drops it
        (0.22, 0.2, True),     # a looser cut-off picks it up
    ],
)
def test_cnv_load_counts_against_the_cutoff_in_force(level, cutoff, expect_called):
    load = compute_cnv_load(_flat_track(level), BIN_WIDTH, "XY", cutoff_override=cutoff)
    assert (load["total_percent"] > 99.0) is expect_called
    expected_label = (
        f"±{cutoff:g}, non-default" if cutoff else "±0.3, calling default"
    )
    assert load["cutoff_label"] == expected_label


@pytest.mark.parametrize(
    "level, cutoff, expected",
    [
        (0.32, None, "GAIN"),
        (0.32, 0.35, CNV_NO_EVENT_LABEL),
        (-0.32, 0.35, CNV_NO_EVENT_LABEL),
    ],
)
def test_gene_states_follow_the_cutoff_in_force(level, cutoff, expected):
    gene_frame = pd.DataFrame(
        [{"chrom": "chr1", "start_pos": 10_000_000, "end_pos": 14_000_000,
          "gene": "TESTGENE"}]
    )
    rows = compute_target_gene_cnv_states(
        _flat_track(level), BIN_WIDTH, ["TESTGENE"], "XY",
        gene_frames=[gene_frame], cutoff_override=cutoff,
    )
    assert rows[0]["state"] == expected


def test_whole_chromosome_events_follow_the_cutoff_in_force():
    cytobands = load_cytobands_bed()
    if cytobands.empty:
        pytest.skip("cytoband resource unavailable")
    track = _flat_track(0.32, bins=250)

    called = detect_cnv_events(
        cnv_data=track, bin_width=BIN_WIDTH, sex_estimate="XY",
        cytobands_df=cytobands, cutoff_override=None,
    )
    assert any(e.event_type.startswith("WHOLE_CHR") for e in called)

    stricter = detect_cnv_events(
        cnv_data=track, bin_width=BIN_WIDTH, sex_estimate="XY",
        cytobands_df=cytobands, cutoff_override=0.35,
    )
    assert not stricter


def test_regional_analysis_gate_follows_the_cutoff_in_force():
    """The adaptive regional rule is gated on the calling cut-off, so raising the
    cut-off must raise the gate rather than leaving it at the default."""
    cytobands = load_cytobands_bed()
    centromeres = load_centromere_bed()
    if cytobands.empty or centromeres.empty:
        pytest.skip("cytoband resources unavailable")

    values = np.full(250, 0.0)
    values[40:70] = 0.32  # past 0.3, inside 0.35
    empty_genes = pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "gene"])

    def called_states(cutoff):
        df = analyze_cytoband_cnv(
            {"chr1": values}, "chr1", {"bin_width": BIN_WIDTH}, cytobands,
            centromeres, empty_genes, "XY", cutoff_override=cutoff,
        )
        if df.empty:
            return set()
        return set(df["cnv_state"]) - {"NO_CHANGE", "NO_DATA"}

    assert called_states(None), "a 0.32 block should be called at the 0.3 default"
    assert not called_states(0.35), "and must not be called at 0.35"


def test_report_preferences_carry_the_live_sample_cutoff():
    """A report generated from the CNV panel must use the cut-off on screen.

    The panel's Cut-off control is per-sample session state; the report is built
    from the persisted admin preferences. If the live value is not folded in, a
    reviewer sets 0.35, generates the report, and silently gets one called at the
    0.3 default.
    """
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_cutoff,
        resolve_cnv_gui_cutoff,
    )

    from robin.gui_launcher import GUILauncher

    class _Launcher(GUILauncher):  # inherits the state -> preference mapping
        def __init__(self):
            self.plotting_preferences = PlottingPreferencesConfig(
                cnv_gui_cutoff="calling"
            )
            self._cnv_state = {}

    launcher = _Launcher()

    def cutoff_for(sample):
        prefs = launcher._plotting_preferences_for_sample(sample)
        return resolve_cnv_cutoff(resolve_cnv_gui_cutoff(prefs))

    # No live selection: the admin default stands.
    assert cutoff_for("/data/s1") is None

    launcher._cnv_state["/data/s1"] = {"cutoff": "0.35"}
    assert cutoff_for("/data/s1") == 0.35
    # Session state is per sample, so another sample is unaffected.
    assert cutoff_for("/data/s2") is None

    # An explicit "calling" selection means the calling thresholds, not 0.35.
    launcher._cnv_state["/data/s1"] = {"cutoff": "calling"}
    assert cutoff_for("/data/s1") is None


def test_cutoff_label_names_the_default_and_the_override():
    """Neither case may be left for the reader to infer.

    "±0.3" alone does not tell a reader whether that is the site's configured
    threshold or something a reviewer typed, and a report may be read by someone
    who does not know the configured value.
    """
    from robin.gui.plotting_preferences import (
        cnv_cutoff_heading_suffix,
        cnv_cutoff_label,
    )

    assert cnv_cutoff_label(None) == "±0.3, calling default"
    assert cnv_cutoff_label(0.3) == "±0.3, calling default"
    assert cnv_cutoff_label(0.35) == "±0.35, non-default"
    assert cnv_cutoff_label(0.2) == "±0.2, non-default"
    assert cnv_cutoff_heading_suffix(0.35) == " (cut-off ±0.35, non-default)"


def test_every_cutoff_driven_heading_carries_the_label():
    """Each output that moves with the cut-off must say which one it used."""
    from robin.analysis.cnv_regional import cnv_load_summary_text, compute_cnv_load

    load = compute_cnv_load(_flat_track(0.5), BIN_WIDTH, "XY", cutoff_override=0.35)
    assert "±0.35, non-default" in load["cutoff_label"]
    assert "cut-off ±0.35, non-default" in cnv_load_summary_text(load)


def test_live_panel_settings_reach_the_report():
    """Every CNV panel control must reach a report generated from that panel.

    The Cut-off was plumbed through first; the Y-ranges and label settings had
    the same gap, so the report silently used the saved admin defaults while the
    reviewer was looking at something else on screen.
    """
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_cutoff,
        resolve_cnv_gene_label_points,
        resolve_cnv_genome_axis_log2,
        resolve_cnv_gui_cutoff,
    )
    from robin.gui_launcher import GUILauncher

    class _Launcher(GUILauncher):  # inherits the state -> preference mapping
        def __init__(self):
            self.plotting_preferences = PlottingPreferencesConfig()
            self._cnv_state = {}

    launcher = _Launcher()

    def prefs_for(sample):
        return launcher._plotting_preferences_for_sample(sample)

    # Nothing selected: the admin defaults stand.
    admin_default = resolve_cnv_genome_axis_log2(PlottingPreferencesConfig())
    assert resolve_cnv_genome_axis_log2(prefs_for("/data/s1")) == admin_default

    # A value the admin default is not, so this proves the panel overrode it.
    launcher._cnv_state["/data/s1"] = {
        "genome_axis": "3",
        "cutoff": "0.35",
        "gene_label_size": "7",
    }
    live = prefs_for("/data/s1")
    assert resolve_cnv_genome_axis_log2(live) == 3.0
    assert resolve_cnv_genome_axis_log2(live) != admin_default
    assert resolve_cnv_cutoff(resolve_cnv_gui_cutoff(live)) == 0.35
    assert resolve_cnv_gene_label_points(live) == 7.0
    assert resolve_cnv_gene_label_points(live) != resolve_cnv_gene_label_points(
        PlottingPreferencesConfig()
    )

    # Per sample, and "auto" means auto rather than a leftover value.
    assert resolve_cnv_genome_axis_log2(prefs_for("/data/s2")) == admin_default
    launcher._cnv_state["/data/s1"] = {"genome_axis": "auto"}
    assert resolve_cnv_genome_axis_log2(prefs_for("/data/s1")) is None


def test_genome_figure_honours_a_fixed_y_range():
    """A fixed Y-range must set the axis, not merely be accepted and ignored."""
    from robin.reporting.plotting import build_CNV_genome_figure
    from robin.analysis.cnv_analysis import Result

    rng = np.random.default_rng(0)
    cnv = {f"chr{c}": rng.normal(0.0, 0.15, 400) for c in list(range(1, 23)) + ["X"]}
    cnv_dict = {"bin_width": 100_000}

    for requested in (1.0, 2.0, 4.0):
        fig = build_CNV_genome_figure(
            Result({k: 2 * 2 ** v for k, v in cnv.items()}),
            cnv_dict,
            normalized_cnv=cnv,
            use_normalized_difference=True,
            sex_estimate="XY",
            fixed_axis_log2=requested,
        )
        assert fig is not None
        low, high = fig.axes[0].get_ylim()
        assert low == pytest.approx(-requested)
        assert high == pytest.approx(requested)


# --- gene value rule ---------------------------------------------------------

def test_single_dropout_bin_cannot_carry_a_gene_call():
    """One deep bin among normal ones is a coverage dropout, not a deletion.

    A bin at a fraction of expected depth reads as a very negative log2 value.
    Taken alone it was enough to report a gene as deleted where the gene was
    in fact unchanged, so a single bin must not carry the call.
    """
    from robin.cnv_plot_style import robust_gene_value

    rb1 = np.full(30, 0.05)
    rb1[11] = -3.1                      # one dropout bin
    value = robust_gene_value(rb1, cutoff=0.3)
    assert abs(value) < 0.3, "a lone dropout must not produce a call"


def test_a_real_homozygous_deletion_is_still_reported_at_its_own_depth():
    """Runs of 3-7 bins backed every array-confirmed deletion; 2 is the floor."""
    from robin.cnv_plot_style import (
        CNV_GENE_FOCAL_RESCUE_MIN_BINS,
        robust_gene_value,
    )

    assert CNV_GENE_FOCAL_RESCUE_MIN_BINS == 2
    cdkn2a = np.array([0.05, 0.02, -5.1, -5.0, -4.8, -5.2, 0.03, 0.01])
    assert robust_gene_value(cdkn2a, cutoff=0.3) == pytest.approx(-5.2)

    # Two bins is enough; one is not.
    assert abs(robust_gene_value(np.array([0.0, -2.0, 0.0, 0.0]), cutoff=0.3)) < 0.9
    assert robust_gene_value(np.array([0.0, -2.0, -2.0, 0.0]), cutoff=0.3) == pytest.approx(-2.0)


def test_a_gap_in_coverage_does_not_join_two_bins_into_a_run():
    """Non-adjacent deep bins are two dropouts, not one event."""
    from robin.cnv_plot_style import robust_gene_value

    gapped = np.array([0.0, -3.0, np.nan, -3.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(robust_gene_value(gapped, cutoff=0.3)) < 3.0


def test_a_gene_narrower_than_a_bin_is_widened_before_judging():
    """A single bin has no noise protection at all.

    unique_genes.bed carries EGFR twice, once as a 205 bp stub. Judged on its
    one bin, that stub read as a clear loss while the gene as a whole was
    unchanged.
    """
    from robin.cnv_plot_style import CNV_GENE_MIN_EVAL_BINS, gene_bin_window

    assert CNV_GENE_MIN_EVAL_BINS >= 3
    assert gene_bin_window(10, 11, 100) == (9, 12)
    assert gene_bin_window(10, 40, 100) == (10, 40)   # already wide: untouched
    # Never runs off either end of the track.
    assert gene_bin_window(0, 1, 100) == (0, 3)
    assert gene_bin_window(99, 100, 100) == (97, 100)
    start, stop = gene_bin_window(0, 1, 2)
    assert (start, stop) == (0, 2)


def test_multi_name_bed_rows_resolve_each_gene_they_name():
    """`CDKN2A,CDKN2B,CDKN2B-AS1` must be findable as CDKN2A.

    Exact-string matching reported 20 of the 45 NGTD loci as "not in reference",
    including a CDKN2A homozygous deletion later confirmed by array.
    """
    import pandas as pd
    from robin.analysis.cnv_regional import compute_target_gene_cnv_states

    bed = pd.DataFrame([
        {"chrom": "chr9", "start_pos": 21_968_177, "end_pos": 22_009_071,
         "gene": "CDKN2A,CDKN2B,CDKN2B-AS1"},
    ])
    track = {"chr9": np.full(60, -3.0)}
    for name in ("CDKN2A", "CDKN2B", "CDKN2B-AS1", "CDKN2B/CDKN2B-AS1"):
        row = compute_target_gene_cnv_states(
            track, 1_000_000, [name], "XY", gene_frames=[bed]
        )[0]
        assert row["state"] == "LOSS", f"{name} did not resolve: {row['state']}"


# --- baseline centring -------------------------------------------------------

def test_baseline_shift_finds_the_modal_level():
    """Ploidy is scaled to the genome mean, which is not the diploid level.

    On a genome with net loss the mean sits below the diploid peak, so every
    unchanged region floats upward. In testing this ran consistently positive,
    and grew with how aneuploid the genome was.
    """
    from robin.analysis.cnv_analysis import estimate_cnv_baseline_shift

    rng = np.random.default_rng(0)

    def genome(offset, sd=0.15, n=3000):
        return {f"chr{c}": rng.normal(offset, sd, n) for c in range(1, 23)}

    assert estimate_cnv_baseline_shift(genome(0.08)) == pytest.approx(0.08, abs=0.02)
    assert estimate_cnv_baseline_shift(genome(-0.06)) == pytest.approx(-0.06, abs=0.02)
    assert estimate_cnv_baseline_shift(genome(0.0)) == pytest.approx(0.0, abs=0.02)
    # A noisier sample must still be corrected: peak height falls with noise, so
    # the guard cannot be a fixed fraction of the bins.
    assert estimate_cnv_baseline_shift(genome(0.05, sd=0.24)) == pytest.approx(
        0.05, abs=0.02
    )


def test_baseline_shift_refuses_rather_than_guesses():
    """An uncertain estimate must leave the track alone, never shift it blindly."""
    from robin.analysis.cnv_analysis import (
        CNV_MAX_BASELINE_SHIFT,
        estimate_cnv_baseline_shift,
    )

    rng = np.random.default_rng(1)
    # Too few bins to estimate anything.
    assert estimate_cnv_baseline_shift({"chr1": np.zeros(10)}) == 0.0
    # No dominant level: a flat histogram must not drive a correction.
    flat = {f"chr{c}": rng.uniform(-1.4, 1.4, 3000) for c in range(1, 23)}
    assert estimate_cnv_baseline_shift(flat) == 0.0
    # Beyond the cap the genome is more likely badly aneuploid than off-centre,
    # and shifting it could subtract a real event.
    huge = {f"chr{c}": rng.normal(CNV_MAX_BASELINE_SHIFT + 0.3, 0.1, 3000)
            for c in range(1, 23)}
    assert estimate_cnv_baseline_shift(huge) == 0.0
    # Sex chromosomes must not drag the estimate.
    assert estimate_cnv_baseline_shift({"chrX": rng.normal(-1.0, 0.1, 5000)}) == 0.0


def test_centring_is_applied_and_can_be_turned_off():
    from robin.analysis.cnv_analysis import compute_cnv_log2_from_ploidy

    rng = np.random.default_rng(2)
    # Ploidy centred on 2.11 -> log2 about +0.077 before centring.
    cnv = {f"chr{c}": rng.normal(2.11, 0.2, 3000) for c in range(1, 23)}

    raw = compute_cnv_log2_from_ploidy(cnv, "XY", centre_baseline=False)
    centred = compute_cnv_log2_from_ploidy(cnv, "XY")

    raw_med = float(np.median(np.concatenate([v for v in raw.values()])))
    cen_med = float(np.median(np.concatenate([v for v in centred.values()])))
    assert raw_med > 0.05, "test genome should start off-centre"
    assert abs(cen_med) < 0.02, "centred track should sit on zero"
    # Differences between regions must be preserved -- it is a shift, not a rescale.
    shift = raw_med - cen_med
    for chrom in raw:
        assert np.allclose(raw[chrom] - centred[chrom], shift, atol=1e-9)


# --- genome-wide segment line ------------------------------------------------

def test_arm_mean_spans_split_at_the_centromere():
    from robin.cnv_plot_style import cnv_arm_mean_spans

    x = np.arange(1000, dtype=float)
    y = np.concatenate([np.zeros(400), np.full(600, 0.55)])

    # Without a split the two arms average into one misleading level.
    (only,) = cnv_arm_mean_spans(x, y)
    assert only[2] == pytest.approx(0.33, abs=0.01)

    p, q = cnv_arm_mean_spans(x, y, split_at=[400.0])
    assert p[2] == pytest.approx(0.0, abs=1e-9)
    assert q[2] == pytest.approx(0.55, abs=1e-9)

    # Gaps are excluded from the mean, not treated as zero.
    gapped = y.copy()
    gapped[10:20] = np.nan
    p2, _q2 = cnv_arm_mean_spans(x, gapped, split_at=[400.0])
    assert p2[2] == pytest.approx(0.0, abs=1e-9)


def test_genome_line_uses_the_value_the_arm_table_called_on():
    """The figure and the arm table must not show different levels for one arm.

    Recomputing the arm mean from the plotted points moved 13 of 561 arms across
    the cut-off; splitting the calling track at the centromere rather than the
    cytoband arm boundary still moved 2. The level is therefore taken from
    analyze_chromosome_arms itself.
    """
    from robin.analysis.cnv_analysis import prepare_cnv_calling_track
    from robin.analysis.cnv_classification import analyze_chromosome_arms
    from robin.analysis.cnv_regional import load_cytobands_bed
    from robin.reporting.plotting import _arm_levels_from_calling_track

    cytobands = load_cytobands_bed()
    if cytobands.empty:
        pytest.skip("cytoband resource unavailable")

    rng = np.random.default_rng(3)
    ploidy = {f"chr{c}": rng.normal(2.0, 0.2, 260) for c in range(1, 23)}
    ploidy["chr7"] = rng.normal(3.0, 0.2, 260)          # a whole-chromosome gain
    bin_width = 1_000_000

    drawn = _arm_levels_from_calling_track(ploidy, bin_width, "XY")
    calling, calling_bin = prepare_cnv_calling_track(ploidy, bin_width, "XY")

    checked = 0
    for contig, arms in drawn.items():
        p_mean, q_mean, *_ = analyze_chromosome_arms(
            calling, contig, calling_bin, "XY", cytobands
        )
        for arm, table in (("p", p_mean), ("q", q_mean)):
            if arm not in arms or table is None:
                continue
            assert arms[arm] == pytest.approx(float(table), abs=1e-12)
            checked += 1
    assert checked > 20, "expected to compare many arms"
    # The gained chromosome still reads past the cut-off on the line.
    assert abs(drawn["chr7"]["q"]) >= 0.3


def test_a_segment_never_spans_a_gap_in_coverage():
    """Bins far apart on the genome are not neighbours, however adjacent in the array.

    The chromosome pages drop missing bins before plotting, so the segmenter sees
    a gap-free array. On chr1 that joined two clusters of bins either side of the
    1q12 heterochromatin -- 18 Mb apart -- into one segment, and painted a single
    level across the whole span.
    """
    from robin.cnv_plot_style import cnv_segment_spans

    # Two clusters 18 Mb apart at different levels, gaps already stripped.
    left_x = np.arange(124.78, 125.18, 0.012)
    right_x = np.arange(143.18, 143.49, 0.012)
    xs = np.concatenate([left_x, right_x])
    ys = np.concatenate([np.full(left_x.size, -1.11), np.full(right_x.size, -1.66)])

    spans = cnv_segment_spans(xs, ys)
    assert len(spans) == 2, "the two clusters must not be joined"
    assert not any(end - start > 5 for start, end, _ in spans)
    (l_start, l_end, l_level), (r_start, r_end, r_level) = spans
    assert l_level == pytest.approx(-1.11, abs=1e-6)
    assert r_level == pytest.approx(-1.66, abs=1e-6)
    assert l_end < 126 and r_start > 143

    # Evenly spaced data is untouched -- the gap rule must not fire on normal input.
    even_x = np.arange(0.0, 100.0, 0.05)
    even_y = np.zeros(even_x.size)
    assert len(cnv_segment_spans(even_x, even_y)) == 1


def test_every_drawn_level_equals_the_data_beneath_it():
    """A level painted over bins it was not computed from is simply wrong."""
    from robin.cnv_plot_style import cnv_segment_spans

    rng = np.random.default_rng(4)
    xs = np.arange(0.0, 200.0, 0.05)
    ys = np.where(xs < 80, rng.normal(0.0, 0.1, xs.size), rng.normal(-0.5, 0.1, xs.size))
    for start, end, level in cnv_segment_spans(xs, ys):
        under = ys[(xs >= start) & (xs <= end)]
        under = under[np.isfinite(under)]
        if under.size:
            assert level == pytest.approx(float(under.mean()), abs=1e-9)
