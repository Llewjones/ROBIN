from __future__ import annotations

from pathlib import Path

import pytest

from robin.gui.plotting_preferences import (
    CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
    CNV_REPORT_SCALE_PLOIDY,
    PLOTTING_PREFERENCES_KEY,
    PlottingPreferencesConfig,
    cnv_report_plot_caption,
    resolve_cnv_report_scale,
    resolve_cnv_summary_normalized,
)
from robin.reference_contigs import DEFAULT_REFERENCE_CONTIG_SCOPE
from robin.security import SecurityStore


def test_plotting_preferences_defaults() -> None:
    config = PlottingPreferencesConfig()
    assert config.cnv_report_scale == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
    assert config.reference_contig_scope == DEFAULT_REFERENCE_CONTIG_SCOPE


def test_plotting_preferences_legacy_log2_alias() -> None:
    config = PlottingPreferencesConfig.from_dict(
        {"cnv_report_scale": "log2_ratio"}
    )
    assert config.cnv_report_scale == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE


def test_cnv_report_plot_caption_normalized() -> None:
    caption = cnv_report_plot_caption(CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE)
    assert "log" in caption.lower()


def test_security_store_plotting_preferences_roundtrip(tmp_path: Path) -> None:
    store = SecurityStore(db_path=tmp_path / "security.db")
    payload = PlottingPreferencesConfig(
        cnv_report_scale=CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
    ).to_dict()
    store.set_gui_setting(PLOTTING_PREFERENCES_KEY, payload, updated_by_user_id=None)
    loaded = store.get_gui_setting(PLOTTING_PREFERENCES_KEY)
    assert loaded is not None
    restored = PlottingPreferencesConfig.from_dict(loaded)
    assert resolve_cnv_report_scale(restored.cnv_report_scale) == (
        CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
    )


def test_resolve_cnv_summary_normalized_uses_admin_preference(tmp_path: Path) -> None:
    store = SecurityStore(db_path=tmp_path / "security.db")
    store.set_gui_setting(
        PLOTTING_PREFERENCES_KEY,
        PlottingPreferencesConfig(
            cnv_report_scale=CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
        ).to_dict(),
        updated_by_user_id=None,
    )
    prefs = PlottingPreferencesConfig.from_dict(
        store.get_gui_setting(PLOTTING_PREFERENCES_KEY)
    )
    assert resolve_cnv_summary_normalized(None, plotting_preferences=prefs) is True
    assert resolve_cnv_summary_normalized(False, plotting_preferences=prefs) is False
    assert resolve_cnv_summary_normalized(True, plotting_preferences=prefs) is True


def test_resolve_cnv_summary_normalized_defaults_to_log2_ratio() -> None:
    """CNV plots default to the log2 ratio scale, matching the array convention."""
    assert resolve_cnv_summary_normalized(None, plotting_preferences=PlottingPreferencesConfig()) is True

    ploidy = PlottingPreferencesConfig(cnv_report_scale=CNV_REPORT_SCALE_PLOIDY)
    assert resolve_cnv_summary_normalized(None, plotting_preferences=ploidy) is False


def test_robin_report_loads_admin_plotting_preferences(tmp_path: Path, monkeypatch) -> None:
    """When plotting_preferences is omitted, RobinReport must load the store default."""
    from robin.reporting.report import RobinReport

    prefs = PlottingPreferencesConfig(
        cnv_report_scale=CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
    )
    monkeypatch.setattr(
        "robin.gui.plotting_preferences.load_plotting_preferences",
        lambda store=None: prefs,
    )

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "master.csv").write_text(
        "read_id,channel,mux,start_time,duration,passed_filter\n"
    )

    report = RobinReport(
        filename=str(tmp_path / "test_report.pdf"),
        output=str(sample_dir),
        center="test",
        plotting_preferences=None,
    )
    assert report.cnv_summary_normalized is True
    assert (
        report.plotting_preferences.cnv_report_scale
        == CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE
    )


def test_robin_report_respects_explicit_ploidy_override(tmp_path: Path, monkeypatch) -> None:
    from robin.reporting.report import RobinReport

    prefs = PlottingPreferencesConfig(
        cnv_report_scale=CNV_REPORT_SCALE_NORMALIZED_DIFFERENCE,
    )
    monkeypatch.setattr(
        "robin.gui.plotting_preferences.load_plotting_preferences",
        lambda store=None: prefs,
    )

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "master.csv").write_text(
        "read_id,channel,mux,start_time,duration,passed_filter\n"
    )

    report = RobinReport(
        filename=str(tmp_path / "test_report.pdf"),
        output=str(sample_dir),
        center="test",
        cnv_summary_normalized=False,
        plotting_preferences=None,
    )
    assert report.cnv_summary_normalized is False


def test_robin_report_empty_config_does_not_load_store(tmp_path: Path, monkeypatch) -> None:
    """Explicit empty PlottingPreferencesConfig must not be replaced by store load."""
    from robin.reporting.report import RobinReport

    called = {"count": 0}

    def _boom(store=None):
        called["count"] += 1
        raise AssertionError("load_plotting_preferences should not be called")

    monkeypatch.setattr(
        "robin.gui.plotting_preferences.load_plotting_preferences",
        _boom,
    )

    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "master.csv").write_text(
        "read_id,channel,mux,start_time,duration,passed_filter\n"
    )

    report = RobinReport(
        filename=str(tmp_path / "test_report.pdf"),
        output=str(sample_dir),
        center="test",
        plotting_preferences=PlottingPreferencesConfig(),
    )
    # Tracks the default rather than pinning a scale: the point of this test is
    # that the store was never consulted.
    assert report.cnv_summary_normalized is True
    assert called["count"] == 0


def test_cnv_cutoff_setting_normalises_values() -> None:
    from robin.gui.plotting_preferences import (
        CNV_CUTOFF_CALLING,
        cnv_cutoff_setting,
        resolve_cnv_cutoff,
    )

    assert cnv_cutoff_setting(None) == CNV_CUTOFF_CALLING
    assert cnv_cutoff_setting("calling") == CNV_CUTOFF_CALLING
    assert cnv_cutoff_setting("junk") == CNV_CUTOFF_CALLING
    assert cnv_cutoff_setting(0.15) == "0.15"
    # Stored magnitude is always positive; callers apply it symmetrically.
    assert cnv_cutoff_setting("-0.3") == "0.3"

    assert resolve_cnv_cutoff(CNV_CUTOFF_CALLING) is None
    assert resolve_cnv_cutoff("0.2") == 0.2
    assert resolve_cnv_cutoff(0) is None


def test_cnv_cutoff_options_lead_with_the_calling_default() -> None:
    from robin.gui.plotting_preferences import CNV_CUTOFF_CALLING, cnv_cutoff_options

    options = cnv_cutoff_options()
    assert next(iter(options)) == CNV_CUTOFF_CALLING
    assert "0.3" in options


def test_cutoff_preference_round_trips() -> None:
    from robin.gui.plotting_preferences import PlottingPreferencesConfig

    config = PlottingPreferencesConfig().with_updates(cnv_gui_cutoff="0.15")
    assert config.cnv_gui_cutoff == "0.15"
    assert PlottingPreferencesConfig.from_dict(config.to_dict()).cnv_gui_cutoff == "0.15"


def test_chrom_axis_setting_normalises_values() -> None:
    from robin.gui.plotting_preferences import (
        CNV_CHROM_AXIS_AUTO,
        cnv_chrom_axis_setting,
        resolve_cnv_chrom_axis,
    )

    assert cnv_chrom_axis_setting(None) == CNV_CHROM_AXIS_AUTO
    assert cnv_chrom_axis_setting("auto") == CNV_CHROM_AXIS_AUTO
    assert cnv_chrom_axis_setting("junk") == CNV_CHROM_AXIS_AUTO
    assert cnv_chrom_axis_setting(2) == "2"
    assert cnv_chrom_axis_setting("-1.2") == "1.2"

    assert resolve_cnv_chrom_axis(CNV_CHROM_AXIS_AUTO) is None
    assert resolve_cnv_chrom_axis("2") == 2.0
    assert resolve_cnv_chrom_axis(0) is None


def test_chrom_axis_defaults_to_plus_minus_one_point_five() -> None:
    """The agreed default range for the per-chromosome pages."""
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_chromosome_axis_log2,
    )

    config = PlottingPreferencesConfig()
    assert config.cnv_gui_chrom_axis == "1.5"
    assert resolve_cnv_chromosome_axis_log2(config) == 1.5


def test_chrom_axis_options_lead_with_auto() -> None:
    from robin.gui.plotting_preferences import (
        CNV_CHROM_AXIS_AUTO,
        cnv_chrom_axis_options,
    )

    options = cnv_chrom_axis_options()
    assert next(iter(options)) == CNV_CHROM_AXIS_AUTO
    assert "2" in options


def test_chrom_axis_preference_round_trips() -> None:
    from robin.gui.plotting_preferences import PlottingPreferencesConfig

    config = PlottingPreferencesConfig().with_updates(cnv_gui_chrom_axis="1.2")
    assert config.cnv_gui_chrom_axis == "1.2"
    restored = PlottingPreferencesConfig.from_dict(config.to_dict())
    assert restored.cnv_gui_chrom_axis == "1.2"

    auto = config.with_updates(cnv_gui_chrom_axis="auto")
    assert PlottingPreferencesConfig.from_dict(auto.to_dict()).cnv_gui_chrom_axis == "auto"


def test_gene_label_size_defaults_smaller_than_the_old_fixed_size() -> None:
    """The report labels were 6 pt and too crowded on a full panel."""
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_gene_label_points,
    )

    config = PlottingPreferencesConfig()
    assert resolve_cnv_gene_label_points(config) < 6.0


def test_gene_label_size_normalises_and_clamps() -> None:
    from robin.gui.plotting_preferences import (
        CNV_GENE_LABEL_SIZE_PRESETS,
        cnv_gene_label_size_setting,
        resolve_cnv_gene_label_size,
    )

    assert cnv_gene_label_size_setting(5) == "5"
    assert cnv_gene_label_size_setting("junk") == "4.5"
    assert cnv_gene_label_size_setting(None) == "4.5"
    # A stray value cannot make labels unreadable or swamp the panel.
    assert resolve_cnv_gene_label_size(99) == max(CNV_GENE_LABEL_SIZE_PRESETS)
    assert resolve_cnv_gene_label_size(0.01) == min(CNV_GENE_LABEL_SIZE_PRESETS)


def test_gene_label_size_options_are_labelled_in_points() -> None:
    from robin.gui.plotting_preferences import cnv_gene_label_size_options

    options = cnv_gene_label_size_options()
    assert "4.5" in options
    assert "pt" in options["4.5"]


def test_gui_label_size_scales_with_the_report_setting() -> None:
    """One setting has to read the same in both renderers."""
    from robin.gui.plotting_preferences import cnv_gene_label_size_px

    small = cnv_gene_label_size_px("3.5")
    large = cnv_gene_label_size_px("8")
    assert small < large
    assert small > 0


def test_gene_label_size_preference_round_trips() -> None:
    from robin.gui.plotting_preferences import PlottingPreferencesConfig

    config = PlottingPreferencesConfig().with_updates(cnv_gui_gene_label_size="6")
    assert config.cnv_gui_gene_label_size == "6"
    restored = PlottingPreferencesConfig.from_dict(config.to_dict())
    assert restored.cnv_gui_gene_label_size == "6"


def test_label_orientation_defaults_to_horizontal() -> None:
    """Gene names read left to right unless a crowded panel calls for Portrait."""
    from robin.gui.plotting_preferences import (
        CNV_LABEL_ORIENTATION_HORIZONTAL,
        PlottingPreferencesConfig,
        cnv_label_is_rotated,
    )

    config = PlottingPreferencesConfig()
    assert config.cnv_gui_label_orientation == CNV_LABEL_ORIENTATION_HORIZONTAL
    assert cnv_label_is_rotated(config.cnv_gui_label_orientation) is False

    # Portrait is still reachable and still rotates.
    rotated = config.with_updates(cnv_gui_label_orientation="portrait")
    assert cnv_label_is_rotated(rotated.cnv_gui_label_orientation) is True


def test_label_orientation_accepts_the_terms_users_use() -> None:
    from robin.gui.plotting_preferences import (
        DEFAULT_CNV_LABEL_ORIENTATION,
        resolve_cnv_label_orientation,
    )

    for value in ("portrait", "vertical", "rotated"):
        assert resolve_cnv_label_orientation(value) == "rotated"
    for value in ("horizontal", "flat"):
        assert resolve_cnv_label_orientation(value) == "horizontal"
    # Anything unrecognised falls back to the default rather than erroring.
    assert resolve_cnv_label_orientation("junk") == DEFAULT_CNV_LABEL_ORIENTATION
    assert resolve_cnv_label_orientation(None) == DEFAULT_CNV_LABEL_ORIENTATION


def test_label_orientation_round_trips() -> None:
    from robin.gui.plotting_preferences import PlottingPreferencesConfig

    config = PlottingPreferencesConfig().with_updates(
        cnv_gui_label_orientation="horizontal"
    )
    assert config.cnv_gui_label_orientation == "horizontal"
    restored = PlottingPreferencesConfig.from_dict(config.to_dict())
    assert restored.cnv_gui_label_orientation == "horizontal"


def test_genome_y_range_is_fixed_by_default_and_independent_of_chromosomes() -> None:
    """The two Y-range menus default differently and never track each other."""
    from robin.gui.plotting_preferences import (
        DEFAULT_CNV_GENOME_AXIS,
        PlottingPreferencesConfig,
        resolve_cnv_chromosome_axis_log2,
        resolve_cnv_genome_axis_log2,
    )

    config = PlottingPreferencesConfig()
    assert DEFAULT_CNV_GENOME_AXIS == "2"
    # The defaults differ, which is itself the independence check.
    assert resolve_cnv_genome_axis_log2(config) == pytest.approx(2.0)
    assert resolve_cnv_chromosome_axis_log2(config) == pytest.approx(1.5)

    updated = config.with_updates(cnv_gui_genome_axis="1.5")
    assert resolve_cnv_genome_axis_log2(updated) == pytest.approx(1.5)
    assert resolve_cnv_chromosome_axis_log2(updated) == pytest.approx(1.5)

    # And the reverse: changing the chromosome range leaves the genome alone.
    updated = config.with_updates(cnv_gui_chrom_axis="3")
    assert resolve_cnv_chromosome_axis_log2(updated) == pytest.approx(3.0)
    assert resolve_cnv_genome_axis_log2(updated) == pytest.approx(2.0)

    # "auto" is still reachable and still means fit-to-data.
    updated = config.with_updates(cnv_gui_genome_axis="auto")
    assert resolve_cnv_genome_axis_log2(updated) is None


def test_genome_y_range_survives_a_round_trip_and_older_stored_settings() -> None:
    from robin.gui.plotting_preferences import (
        DEFAULT_CNV_GENOME_AXIS,
        PlottingPreferencesConfig,
    )

    config = PlottingPreferencesConfig().with_updates(cnv_gui_genome_axis="2.5")
    assert (
        PlottingPreferencesConfig.from_dict(config.to_dict()).cnv_gui_genome_axis
        == "2.5"
    )
    # Preferences saved before this setting existed take the current default
    # rather than a stored value, so an upgrade adopts the agreed window.
    legacy = PlottingPreferencesConfig.from_dict({"schema_version": 8})
    assert legacy.cnv_gui_genome_axis == DEFAULT_CNV_GENOME_AXIS


def test_cnv_cutoff_presets_are_sorted_and_offered_in_order() -> None:
    """The menu is read as a ladder, so a new preset must land in sequence."""
    from robin.gui.plotting_preferences import (
        CNV_CUTOFF_CALLING,
        CNV_CUTOFF_PRESETS,
        cnv_cutoff_options,
        resolve_cnv_cutoff,
    )

    assert 0.35 in CNV_CUTOFF_PRESETS
    assert list(CNV_CUTOFF_PRESETS) == sorted(CNV_CUTOFF_PRESETS)
    assert len(set(CNV_CUTOFF_PRESETS)) == len(CNV_CUTOFF_PRESETS)

    options = cnv_cutoff_options()
    # Calling default stays first; presets follow in ladder order.
    assert list(options)[0] == CNV_CUTOFF_CALLING
    assert options["0.35"] == "±0.35"
    assert list(options)[1:] == [f"{value:g}" for value in CNV_CUTOFF_PRESETS]

    # And it round-trips through the stored form.
    assert resolve_cnv_cutoff("0.35") == pytest.approx(0.35)


def test_report_caption_follows_the_cutoff_in_force():
    """The caption must state the cut-off the figure was actually drawn at.

    The admin Cut-off control is labelled "GUI + report", so a report drawn at
    +/-0.35 that still prints "within +/-0.3" is describing a different figure
    than the one above it.
    """
    from robin.gui.plotting_preferences import (
        cnv_calling_cutoff_magnitude,
        cnv_report_plot_caption,
    )

    calling = cnv_calling_cutoff_magnitude()

    default = cnv_report_plot_caption("normalized_difference")
    assert f"within ±{calling:g}" in default
    assert "calling cut-offs" in default

    overridden = cnv_report_plot_caption(
        "normalized_difference", cutoff_override=0.35
    )
    assert "within ±0.35)" in overridden
    assert f"within ±{calling:g})" not in overridden
    # Everything is called at the override, so the caption names it as the
    # cut-off the report was called at and says what it replaced.
    assert "called at" in overridden
    assert f"in place of the ±{calling:g} default" in overridden

    # An override equal to the calling threshold is not an override.
    assert cnv_report_plot_caption(
        "normalized_difference", cutoff_override=calling
    ) == default


def test_report_resolves_the_admin_cutoff_for_its_figures():
    """The value the report hands to its plots comes from the admin preference."""
    from robin.gui.plotting_preferences import (
        PlottingPreferencesConfig,
        resolve_cnv_cutoff,
        resolve_cnv_gui_cutoff,
    )

    def report_override(setting):
        prefs = PlottingPreferencesConfig(cnv_gui_cutoff=setting)
        return resolve_cnv_cutoff(resolve_cnv_gui_cutoff(prefs))

    assert report_override("calling") is None
    assert report_override("0.35") == 0.35
    assert report_override("0.2") == 0.2
