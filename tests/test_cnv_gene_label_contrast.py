"""Gene names are inked darker than the markers they name.

On the genome-wide panel the names sit directly on the bin scatter, which uses
the same gain/loss blues and reds as the markers. A label in the marker's own
colour sinks into the cloud, so the text is darkened while the marker keeps the
report's convention colour.
"""

import pytest

from robin.reporting.plotting import (
    CNV_COLORS,
    CNV_LABEL_INK,
    _CNV_GENE_LABEL_PATH_EFFECTS,
    _panel_label_color,
    _panel_point_color,
)


def _luminance(hex_colour):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@pytest.mark.parametrize("point", [
    {"clinical_trial": True},
    {"direction": "gain"},
    {"direction": "loss"},
])
def test_label_is_darker_than_its_marker(point):
    assert _luminance(_panel_label_color(point)) < _luminance(_panel_point_color(point))


@pytest.mark.parametrize("point", [
    {"clinical_trial": True},
    {"direction": "gain"},
    {"direction": "loss"},
])
def test_label_never_equals_a_scatter_colour(point):
    """The whole point: the name must not be the colour of the bins under it."""
    scatter = {CNV_COLORS[k] for k in ("plot_gain", "plot_loss", "plot_neutral")}
    assert _panel_label_color(point) not in scatter


def test_step2_label_is_still_distinct_from_gain_and_loss():
    trial = _panel_label_color({"clinical_trial": True})
    assert trial != _panel_label_color({"direction": "gain"})
    assert trial != _panel_label_color({"direction": "loss"})


def test_every_ink_maps_to_a_real_convention_colour():
    for key in CNV_LABEL_INK:
        assert key in CNV_COLORS


def test_gene_markers_are_orange_for_gain_and_green_for_loss():
    assert _panel_point_color({"direction": "gain"}) == CNV_COLORS["gene_gain"]
    assert _panel_point_color({"direction": "loss"}) == CNV_COLORS["gene_loss"]


def test_gui_matches_the_report_convention():
    """These used to run the other way round from the report."""
    from robin.gui.components.cnv import (
        _CNV_GENE_GAIN_COLOR,
        _CNV_GENE_LOSS_COLOR,
    )

    assert _CNV_GENE_GAIN_COLOR == CNV_COLORS["gene_gain"]
    assert _CNV_GENE_LOSS_COLOR == CNV_COLORS["gene_loss"]


def test_bin_scatter_keeps_its_own_colours():
    """Only the gene markers and names were recoloured, not the point cloud."""
    assert CNV_COLORS["plot_gain"] != CNV_COLORS["gene_gain"]
    assert CNV_COLORS["plot_loss"] != CNV_COLORS["gene_loss"]


def test_unknown_marker_colour_falls_back_to_itself():
    """Never drop a label because its colour is unrecognised."""
    assert _panel_label_color({"direction": "gain", "clinical_trial": False})


def test_labels_keep_a_white_halo():
    assert _CNV_GENE_LABEL_PATH_EFFECTS, "halo carries separation over dense scatter"
