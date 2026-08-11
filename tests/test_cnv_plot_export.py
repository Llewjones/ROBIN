"""CNV plot PDF export and the GUI chart wiring that feeds the same conventions."""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest

from robin.reporting.cnv_export import (
    CnvExportUnavailable,
    build_cnv_pdf,
    load_cnv_export_context,
)


def _write_sample(tmp_path, *, bin_width: int = 1_000_000):
    """A minimal sample directory: a gain, a loss and a focal deletion."""
    rng = np.random.default_rng(7)
    cnv = {
        "chr1": np.abs(rng.normal(2.0, 0.2, 60)),
        "chr7": np.abs(rng.normal(3.1, 0.2, 60)),
        "chr10": np.abs(rng.normal(1.0, 0.2, 60)),
    }
    cnv["chr9"] = np.abs(rng.normal(2.0, 0.2, 60))
    cnv["chr9"][20:24] = 0.1

    sample_dir = tmp_path / "sample_TEST"
    sample_dir.mkdir()
    np.save(sample_dir / "CNV.npy", cnv, allow_pickle=True)
    np.save(
        sample_dir / "CNV_dict.npy",
        {"bin_width": bin_width, "variance": 0.03},
        allow_pickle=True,
    )
    with (sample_dir / "XYestimate.pkl").open("wb") as handle:
        pickle.dump("XY", handle)
    return sample_dir


def _pdf_page_count(payload: bytes) -> int:
    import PyPDF2

    return len(PyPDF2.PdfReader(io.BytesIO(payload)).pages)


def test_export_context_reads_the_sample_directory(tmp_path) -> None:
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True)

    assert context["sex_estimate"] == "XY"
    assert context["cnv_dict"]["bin_width"] == 1_000_000
    assert set(context["chromosomes"]) == {"chr1", "chr7", "chr9", "chr10"}
    # log2 track derived from ploidy: a 3-copy autosome is log2(3/2).
    assert context["normalized_cnv"]["chr7"].mean() == pytest.approx(0.585, abs=0.1)


def test_export_raises_when_the_sample_has_no_cnv_data(tmp_path) -> None:
    empty = tmp_path / "sample_EMPTY"
    empty.mkdir()
    with pytest.raises(CnvExportUnavailable):
        load_cnv_export_context(empty, use_log2=True)


@pytest.mark.parametrize("use_log2", [True, False])
def test_genome_pdf_is_a_single_vector_page(tmp_path, use_log2: bool) -> None:
    sample_dir = _write_sample(tmp_path)
    payload, filename = build_cnv_pdf(
        sample_dir, kind="genome", use_log2=use_log2
    )

    assert payload[:5] == b"%PDF-"
    assert _pdf_page_count(payload) == 1
    assert filename.startswith("sample_TEST_CNV_genome_")
    assert filename.endswith(".pdf")
    assert ("log2" in filename) is use_log2


def test_chromosome_pdf_has_one_page_per_chromosome(tmp_path) -> None:
    sample_dir = _write_sample(tmp_path)
    payload, filename = build_cnv_pdf(
        sample_dir, kind="chromosomes", use_log2=True
    )

    assert payload[:5] == b"%PDF-"
    assert _pdf_page_count(payload) == 4
    assert filename == "sample_TEST_CNV_chromosomes_log2.pdf"


def test_chromosome_pdf_can_be_restricted_to_a_selection(tmp_path) -> None:
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True)
    from robin.reporting.cnv_export import export_cnv_chromosome_pdf

    payload = export_cnv_chromosome_pdf(context, chromosomes=["chr7", "chr9"])
    assert _pdf_page_count(payload) == 2


def test_unknown_pdf_kind_is_rejected(tmp_path) -> None:
    sample_dir = _write_sample(tmp_path)
    with pytest.raises(ValueError):
        build_cnv_pdf(sample_dir, kind="something-else", use_log2=True)


def test_export_context_carries_the_live_display_settings(tmp_path) -> None:
    """A downloaded plot must match what the user has set on screen."""
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(
        sample_dir,
        use_log2=True,
        cutoff_override=0.15,
        outliers_only=True,
        show_trend=False,
    )
    assert context["cutoff_override"] == pytest.approx(0.15)
    assert context["outliers_only"] is True
    assert context["show_trend"] is False


def test_export_defaults_match_the_plot_defaults(tmp_path) -> None:
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True)
    assert context["cutoff_override"] is None
    assert context["outliers_only"] is False
    assert context["show_trend"] is True


@pytest.mark.parametrize(
    "settings",
    [
        {"cutoff_override": 0.15},
        {"outliers_only": True},
        {"show_trend": False},
        {"cutoff_override": 0.5, "outliers_only": True, "show_trend": False},
    ],
)
def test_pdf_builds_under_every_display_setting(tmp_path, settings) -> None:
    sample_dir = _write_sample(tmp_path)
    payload, _name = build_cnv_pdf(
        sample_dir, kind="genome", use_log2=True, **settings
    )
    assert payload[:5] == b"%PDF-"
    assert _pdf_page_count(payload) == 1


def test_cutoff_setting_moves_the_drawn_lines(tmp_path) -> None:
    """The exported figure draws the cut-off the user selected, not the default."""
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import build_CNV_genome_figure

    sample_dir = _write_sample(tmp_path)

    def drawn_cutoffs(cutoff):
        context = load_cnv_export_context(
            sample_dir, use_log2=True, cutoff_override=cutoff
        )
        fig = build_CNV_genome_figure(
            context["result"],
            context["cnv_dict"],
            context["normalized_cnv"],
            use_normalized_difference=True,
            sex_estimate=context["sex_estimate"],
            panel_genes_df=context["panel_genes_df"],
            target_coverage_df=context["target_coverage_df"],
            significant_regions=context["significant_regions"],
            cutoff_override=context["cutoff_override"],
        )
        try:
            ax = fig.axes[0]
            return {
                round(float(line.get_ydata()[0]), 4)
                for line in ax.get_lines()
                if len(set(line.get_ydata())) == 1
                and line.get_linestyle() not in ("-", "solid")
            }
        finally:
            plt.close(fig)

    assert {0.3, -0.3} <= drawn_cutoffs(None)
    assert {0.15, -0.15} <= drawn_cutoffs(0.15)
    assert {0.5, -0.5} <= drawn_cutoffs(0.5)


def test_gene_label_size_reaches_the_rendered_figure(tmp_path) -> None:
    """The label size chosen on screen must be the size drawn in the export."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from robin.reporting.plotting import build_CNV_genome_figure

    sample_dir = _write_sample(tmp_path)
    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 5_000_000, "end_pos": 5_600_000, "gene": "EGFR"}]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr7"],
            "startpos": [5_000_000],
            "endpos": [5_600_000],
            "name": ["EGFR"],
            "coverage": [80.0],
        }
    )

    def drawn_sizes(size):
        context = load_cnv_export_context(
            sample_dir, use_log2=True, gene_label_size=size
        )
        fig = build_CNV_genome_figure(
            context["result"],
            context["cnv_dict"],
            context["normalized_cnv"],
            use_normalized_difference=True,
            sex_estimate=context["sex_estimate"],
            panel_genes_df=panel,
            target_coverage_df=coverage,
            significant_regions={},
            configured_genes=("EGFR",),
            gene_label_size=context["gene_label_size"],
        )
        try:
            return {
                text.get_fontsize()
                for text in fig.axes[0].texts
                if text.get_rotation() == 90
            }
        finally:
            plt.close(fig)

    assert drawn_sizes(3.5) == {3.5}
    assert drawn_sizes(8.0) == {8.0}


def test_export_context_defaults_gene_label_size_to_none(tmp_path) -> None:
    """No explicit size means the figure builder keeps its own default."""
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True)
    assert context["gene_label_size"] is None


def test_label_orientation_reaches_the_rendered_figure(tmp_path) -> None:
    """The Label style toggle must change what the export actually draws."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from robin.reporting.plotting import build_CNV_genome_figure

    sample_dir = _write_sample(tmp_path)
    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 5_000_000, "end_pos": 5_600_000, "gene": "EGFR"}]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr7"],
            "startpos": [5_000_000],
            "endpos": [5_600_000],
            "name": ["EGFR"],
            "coverage": [80.0],
        }
    )

    def rotations(rotated):
        context = load_cnv_export_context(
            sample_dir, use_log2=True, gene_labels_rotated=rotated
        )
        fig = build_CNV_genome_figure(
            context["result"],
            context["cnv_dict"],
            context["normalized_cnv"],
            use_normalized_difference=True,
            sex_estimate=context["sex_estimate"],
            panel_genes_df=panel,
            target_coverage_df=coverage,
            significant_regions={},
            configured_genes=("EGFR",),
            gene_labels_rotated=context["gene_labels_rotated"],
        )
        try:
            return {
                text.get_rotation()
                for text in fig.axes[0].texts
                if text.get_text() == "EGFR"
            }
        finally:
            plt.close(fig)

    assert rotations(True) == {90.0}
    assert rotations(False) == {0.0}


def test_export_context_defaults_orientation_to_none(tmp_path) -> None:
    """No explicit orientation means the figure builder keeps its own default."""
    sample_dir = _write_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True)
    assert context["gene_labels_rotated"] is None


def _chromosome_label_rotations(sample_dir, **settings):
    """Label rotations drawn on the per-chromosome figures."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 5_000_000, "end_pos": 5_600_000, "gene": "EGFR"}]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr7"],
            "startpos": [5_000_000],
            "endpos": [5_600_000],
            "name": ["EGFR"],
            "coverage": [80.0],
        }
    )
    context = load_cnv_export_context(
        sample_dir, use_log2=True, configured_genes=("EGFR",), **settings
    )
    rotations = set()
    sizes = set()
    for _contig, fig in iter_CNV_chromosome_figures(
        context["result"],
        context["cnv_dict"],
        normalized_cnv=context["normalized_cnv"],
        use_log2_ratio=True,
        sex_estimate=context["sex_estimate"],
        panel_genes_df=panel,
        target_coverage_df=coverage,
        chromosomes=["chr7"],
        configured_genes=("EGFR",),
        gene_labels_rotated=context["gene_labels_rotated"],
        **(
            {"gene_label_size": context["gene_label_size"]}
            if context.get("gene_label_size")
            else {}
        ),
    ):
        for text in fig.axes[0].texts:
            if text.get_text() == "EGFR":
                rotations.add(text.get_rotation())
                sizes.add(text.get_fontsize())
        plt.close(fig)
    return rotations, sizes


def test_per_chromosome_pages_honour_the_label_orientation(tmp_path) -> None:
    """Regression: the chromosome wrapper accepted the setting but dropped it.

    The genome-wide figure honoured the toggle while every per-chromosome page
    stayed rotated, so an exported PDF disagreed with the live view.
    """
    sample_dir = _write_sample(tmp_path)
    assert _chromosome_label_rotations(sample_dir, gene_labels_rotated=True)[0] == {90.0}
    assert _chromosome_label_rotations(sample_dir, gene_labels_rotated=False)[0] == {0.0}


def test_per_chromosome_pages_honour_the_label_size(tmp_path) -> None:
    """The same wrapper carries the label size, so guard it the same way."""
    sample_dir = _write_sample(tmp_path)
    assert _chromosome_label_rotations(sample_dir, gene_label_size=3.5)[1] == {3.5}
    assert _chromosome_label_rotations(sample_dir, gene_label_size=8.0)[1] == {8.0}


def _deep_deletion_sample(tmp_path):
    """A sample whose chr9 carries a deletion far beyond any fixed window."""
    rng = np.random.default_rng(5)
    cnv = {
        "chr1": np.abs(rng.normal(2.0, 0.2, 150)),
        "chr7": np.abs(rng.normal(2.9, 0.2, 120)),
        "chr9": np.abs(rng.normal(2.0, 0.2, 130)),
    }
    cnv["chr9"][20:24] = 0.02          # about log2 -6.6
    sample_dir = tmp_path / "sample_DEEP"
    sample_dir.mkdir()
    np.save(sample_dir / "CNV.npy", cnv, allow_pickle=True)
    np.save(
        sample_dir / "CNV_dict.npy",
        {"bin_width": 1_000_000, "variance": 0.03},
        allow_pickle=True,
    )
    with (sample_dir / "XYestimate.pkl").open("wb") as handle:
        pickle.dump("XY", handle)
    return sample_dir


def _chromosome_pages(sample_dir, **kwargs):
    import matplotlib.pyplot as plt

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    context = load_cnv_export_context(sample_dir, use_log2=True, fixed_axis_log2=2.0)
    pages = []
    for contig, fig in iter_CNV_chromosome_figures(
        context["result"],
        context["cnv_dict"],
        normalized_cnv=context["normalized_cnv"],
        use_log2_ratio=True,
        sex_estimate="XY",
        fixed_axis_log2=2.0,
        **kwargs,
    ):
        pages.append(
            (contig, fig._suptitle.get_text(), tuple(fig.axes[0].get_ylim()))
        )
        plt.close(fig)
    return pages


def test_only_chromosomes_with_offscale_data_get_a_second_page(tmp_path) -> None:
    sample_dir = _deep_deletion_sample(tmp_path)
    pages = _chromosome_pages(sample_dir, full_range_pages=True)

    contigs = [contig for contig, _title, _ylim in pages]
    assert contigs.count("chr9") == 2          # deep deletion, needs both views
    assert contigs.count("chr1") == 1          # quiet, one page is enough
    assert contigs.count("chr7") == 1          # gain fits the window


def test_full_range_page_shows_the_deepest_bin(tmp_path) -> None:
    """The whole point: an extreme event at its true value, not an edge marker."""
    sample_dir = _deep_deletion_sample(tmp_path)
    pages = _chromosome_pages(sample_dir, full_range_pages=True)

    deepest = float(np.log2(0.02 / 2.0))
    fixed = next(y for c, t, y in pages if c == "chr9" and "full range" not in t)
    full = next(y for c, t, y in pages if c == "chr9" and "full range" in t)

    assert fixed == (-2.0, 2.0)                # the selected granularity
    assert full[0] <= deepest                  # and the true depth
    assert full[0] < fixed[0]


def test_full_range_pages_are_off_by_default(tmp_path) -> None:
    """Callers that key results by chromosome must not get duplicates."""
    sample_dir = _deep_deletion_sample(tmp_path)
    pages = _chromosome_pages(sample_dir)
    contigs = [contig for contig, _title, _ylim in pages]
    assert len(contigs) == len(set(contigs)) == 3


def test_chromosome_pdf_download_includes_the_full_range_pages(tmp_path) -> None:
    from robin.reporting.cnv_export import export_cnv_chromosome_pdf

    sample_dir = _deep_deletion_sample(tmp_path)
    context = load_cnv_export_context(sample_dir, use_log2=True, fixed_axis_log2=2.0)

    with_pages = export_cnv_chromosome_pdf(context)
    without = export_cnv_chromosome_pdf(context, full_range_pages=False)

    assert _pdf_page_count(with_pages) == 4    # three chromosomes, one doubled
    assert _pdf_page_count(without) == 3


def test_full_range_limits_never_clip_the_data() -> None:
    from robin.reporting.plotting import _full_range_y_limits

    values = np.array([-6.64, -0.1, 0.0, 0.2, 1.9])
    lo, hi = _full_range_y_limits(values, use_log=True)
    assert lo <= values.min()
    assert hi >= values.max()
    # The no-change baseline stays on the panel.
    assert lo < 0.0 < hi


def test_full_range_limits_stay_positive_on_the_ploidy_scale() -> None:
    from robin.reporting.plotting import _full_range_y_limits

    lo, hi = _full_range_y_limits(np.array([0.05, 2.0, 3.1]), use_log=False)
    assert lo >= 0.0
    assert hi >= 3.1


def test_clustered_gene_labels_avoid_the_cutoff_line(tmp_path) -> None:
    """Regression: MTAP landed on the +0.3 cut-off line and was unreadable.

    CDKN2A, CDKN2B and MTAP sit within ~300 kb at 9p21, so they stack into
    lanes; on a +/-1.5 window the first lane fell exactly on the cut-off.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    rng = np.random.default_rng(7)
    values = np.abs(rng.normal(2.0, 0.18, 138))
    values[21:23] = 2.0 * 2**0.15          # just above baseline, as in the report
    cnv = {"chr9": values}
    sample_dir = tmp_path / "sample_9P21"
    sample_dir.mkdir()
    np.save(sample_dir / "CNV.npy", cnv, allow_pickle=True)
    np.save(
        sample_dir / "CNV_dict.npy",
        {"bin_width": 1_000_000, "variance": 0.03},
        allow_pickle=True,
    )
    with (sample_dir / "XYestimate.pkl").open("wb") as handle:
        pickle.dump("XY", handle)

    genes = ("MTAP", "CDKN2A", "CDKN2B")
    panel = pd.DataFrame(
        [
            {"chrom": "chr9", "start_pos": 21_800_000, "end_pos": 21_900_000, "gene": "MTAP"},
            {"chrom": "chr9", "start_pos": 21_960_000, "end_pos": 22_010_000, "gene": "CDKN2A"},
            {"chrom": "chr9", "start_pos": 22_100_000, "end_pos": 22_130_000, "gene": "CDKN2B"},
        ]
    )
    coverage = pd.DataFrame(
        {
            "chrom": panel["chrom"],
            "startpos": panel["start_pos"],
            "endpos": panel["end_pos"],
            "name": panel["gene"],
            "coverage": [45.0, 50.0, 48.0],
        }
    )
    context = load_cnv_export_context(sample_dir, use_log2=True, configured_genes=genes)

    drawn = {}
    for _contig, fig in iter_CNV_chromosome_figures(
        context["result"],
        context["cnv_dict"],
        normalized_cnv=context["normalized_cnv"],
        use_log2_ratio=True,
        sex_estimate="XY",
        panel_genes_df=panel,
        target_coverage_df=coverage,
        configured_genes=genes,
        gene_labels_rotated=False,
        fixed_axis_log2=1.5,
    ):
        for text in fig.axes[0].texts:
            if text.get_text() in genes:
                drawn[text.get_text()] = text.get_position()[1]
        plt.close(fig)

    assert set(drawn) == set(genes)
    # None of them may sit on the baseline or either calling cut-off.
    for gene, y in drawn.items():
        for level in (0.0, 0.3, -0.3):
            assert abs(y - level) > 0.06, f"{gene} sits on the {level:+g} line"


def test_offscale_gene_marker_earns_a_full_range_page(tmp_path) -> None:
    """An amplified panel gene beyond the fixed window needs the true-value view."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from robin.reporting.plotting import iter_CNV_chromosome_figures

    rng = np.random.default_rng(3)
    values = np.abs(rng.normal(3.4, 0.15, 159))
    values[55] = 5.4                        # EGFR bin, log2 ~1.43, beyond +/-1.2
    cnv = {"chr7": values}
    sample_dir = tmp_path / "sample_AMP"
    sample_dir.mkdir()
    np.save(sample_dir / "CNV.npy", cnv, allow_pickle=True)
    np.save(
        sample_dir / "CNV_dict.npy",
        {"bin_width": 1_000_000, "variance": 0.03},
        allow_pickle=True,
    )
    with (sample_dir / "XYestimate.pkl").open("wb") as handle:
        pickle.dump("XY", handle)

    panel = pd.DataFrame(
        [{"chrom": "chr7", "start_pos": 55_000_000, "end_pos": 55_400_000, "gene": "EGFR"}]
    )
    coverage = pd.DataFrame(
        {
            "chrom": ["chr7"],
            "startpos": [55_000_000],
            "endpos": [55_400_000],
            "name": ["EGFR"],
            "coverage": [120.0],
        }
    )
    context = load_cnv_export_context(sample_dir, use_log2=True, configured_genes=("EGFR",))

    titles = []
    for _contig, fig in iter_CNV_chromosome_figures(
        context["result"],
        context["cnv_dict"],
        normalized_cnv=context["normalized_cnv"],
        use_log2_ratio=True,
        sex_estimate="XY",
        panel_genes_df=panel,
        target_coverage_df=coverage,
        configured_genes=("EGFR",),
        fixed_axis_log2=1.2,
        full_range_pages=True,
    ):
        titles.append(fig._suptitle.get_text())
        plt.close(fig)

    assert len(titles) == 2
    assert any("full range" in title for title in titles)


# --- downloaded PDFs must agree with the panel and the admin defaults --------


def _export_launcher(state=None):
    """A launcher carrying only what the export settings resolver touches."""
    from robin.gui.plotting_preferences import PlottingPreferencesConfig
    from robin.gui_launcher import GUILauncher

    class _Launcher(GUILauncher):  # inherits the state -> preference overlay
        def __init__(self):
            self.plotting_preferences = PlottingPreferencesConfig()
            self._cnv_state = dict(state or {})

    return _Launcher()


def test_export_with_no_panel_state_uses_the_admin_defaults():
    """A sample whose CNV section was never opened must not export on literals.

    The download handler used to read the in-memory panel state alone, with
    hardcoded fallbacks, so an unopened sample exported on linear / outliers /
    calling-default / auto-fit however the admin defaults were set.
    """
    from robin.gui.components.cnv import cnv_export_settings
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_chromosome_axis_log2,
        resolve_cnv_cutoff,
        resolve_cnv_genome_axis_log2,
        resolve_cnv_gui_cutoff,
        resolve_cnv_gui_y_scale,
    )

    admin = PlottingPreferencesConfig()
    launcher = _export_launcher()          # no state for this sample at all
    use_log2, kwargs = cnv_export_settings(launcher, "/data/never-opened")

    assert use_log2 == (resolve_cnv_gui_y_scale(admin) == "log")
    assert kwargs["cutoff_override"] == resolve_cnv_cutoff(
        resolve_cnv_gui_cutoff(admin)
    )
    assert kwargs["fixed_axis_log2"] == resolve_cnv_chromosome_axis_log2(admin)
    assert kwargs["genome_axis_log2"] == resolve_cnv_genome_axis_log2(admin)


def test_the_live_panel_still_overrides_the_admin_defaults_on_export():
    """Whatever is on screen is what the download must render."""
    from robin.gui.components.cnv import cnv_export_settings

    sample = "/data/s1"
    launcher = _export_launcher(
        {
            sample: {
                "y_scale": "linear",
                "cutoff": "0.2",
                "chrom_axis": "3",
                "genome_axis": "auto",
                "gene_coverage_filter": "outliers",
                "show_trend": False,
            }
        }
    )
    use_log2, kwargs = cnv_export_settings(launcher, sample)

    assert use_log2 is False
    assert kwargs["cutoff_override"] == 0.2
    assert kwargs["fixed_axis_log2"] == 3.0
    assert kwargs["genome_axis_log2"] is None      # "auto" means auto
    assert kwargs["outliers_only"] is True
    assert kwargs["show_trend"] is False

    # A different sample is unaffected by that panel.
    _, other = cnv_export_settings(launcher, "/data/s2")
    assert other["cutoff_override"] != 0.2
