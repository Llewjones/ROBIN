"""Select events must yield the option key, not the visible label.

The CNV cut-off, axis and gene-label selects are ``{key: label}`` maps, e.g.
``{"3.5": "Tiny (3.5 pt)"}``. The handlers read the label out of the event, put
it in the sample state, and the downstream ``float()`` then raised and fell back
to the default - so the dropdown appeared to do nothing at all.

These tests exercise the resolution logic against the real option maps.
"""

import pytest

from robin.gui.plotting_preferences import (
    cnv_chrom_axis_options,
    cnv_cutoff_options,
    cnv_gene_label_size_options,
    cnv_genome_axis_options,
)

OPTION_MAPS = [
    cnv_cutoff_options(),
    cnv_gene_label_size_options(),
    cnv_chrom_axis_options(),
    cnv_genome_axis_options(),
]


class _Widget:
    def __init__(self, options, value=None):
        self.options = options
        self.value = value


class _Event:
    def __init__(self, args=None, value=None):
        if args is not None:
            self.args = args
        if value is not None:
            self.value = value


def _resolve(ev, widget, default=None):
    """Mirror of the component's _select_key, which is defined inside a closure."""
    options = getattr(widget, "options", None) or {}
    keys = {str(key) for key in options}
    raw = getattr(ev, "args", None)
    if raw is None:
        raw = getattr(ev, "value", None)
    if isinstance(raw, list) and len(raw) >= 2 and isinstance(raw[1], dict):
        raw = raw[1].get("label")
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("label"))
    if raw is not None:
        text = str(raw)
        if text in keys:
            return text
        for key, label in options.items():
            if str(label) == text:
                return str(key)
    current = getattr(widget, "value", None)
    if current is not None and str(current) in keys:
        return str(current)
    return default


@pytest.mark.parametrize("options", OPTION_MAPS)
def test_these_selects_really_do_have_labels_unlike_their_keys(options):
    """If this ever stops being true the bug class disappears with it."""
    assert any(str(key) != str(label) for key, label in options.items())


@pytest.mark.parametrize("options", OPTION_MAPS)
def test_a_label_payload_resolves_back_to_its_key(options):
    for key, label in options.items():
        widget = _Widget(options, value=None)
        assert _resolve(_Event(args=str(label)), widget) == str(key)


@pytest.mark.parametrize("options", OPTION_MAPS)
def test_a_key_payload_is_passed_through(options):
    for key in options:
        assert _resolve(_Event(args=str(key)), _Widget(options)) == str(key)


@pytest.mark.parametrize("options", OPTION_MAPS)
def test_quasar_style_payload_resolves(options):
    key, label = next(iter(options.items()))
    ev = _Event(args=[0, {"value": str(key), "label": str(label)}])
    assert _resolve(ev, _Widget(options)) == str(key)


def test_stale_widget_value_never_overrides_the_event():
    """The old code read widget.value back over the event and lost the change."""
    options = cnv_gene_label_size_options()
    widget = _Widget(options, value="3.5")  # stale: the user just picked 6
    assert _resolve(_Event(args=options["6"]), widget) == "6"


def test_unresolvable_payload_falls_back_to_the_default():
    options = cnv_cutoff_options()
    assert _resolve(_Event(args="nonsense"), _Widget(options), "calling") == "calling"


@pytest.mark.parametrize("options", OPTION_MAPS)
def test_resolved_keys_are_usable_downstream(options):
    """Every key must be either a float or a known sentinel, never a label."""
    for key in options:
        text = str(key)
        if text in {"calling", "auto"}:
            continue
        float(text)
