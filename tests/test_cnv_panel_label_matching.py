"""Matching configured ``[cnv].genes`` symbols to panel target labels.

Panel BED names are composite where targets overlap: rCNS2 writes
``CDKN2A,CDKN2B,CDKN2B-AS1`` and ``CASC11,MYC``, and the packaged NGTD list
uses slashes instead (``CDKN2B/CDKN2B-AS1``). Matching the label as one string
silently dropped 15 of a real site's 48 configured genes from the PDF report -
CDKN2A/B, NF1, ERBB2, MYCN, MLH1 among them - while the GUI, which already
split on commas, showed them. Each component is now matched separately.
"""

import os

import pytest

from robin import resources
from robin.reporting.plotting import (
    _panel_label_components,
    _panel_label_matches_configured,
)

PANEL = "rCNS2_panel_name_uniq.bed"


def _panel_labels():
    path = os.path.join(
        os.path.dirname(os.path.abspath(resources.__file__)), PANEL
    )
    if not os.path.exists(path):
        pytest.skip(f"{PANEL} not packaged")
    labels = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                labels.append(parts[3].strip())
    return labels


def test_components_split_on_comma_and_slash():
    assert _panel_label_components("CDKN2A,CDKN2B,CDKN2B-AS1") == [
        "cdkn2a", "cdkn2b", "cdkn2b-as1",
    ]
    assert _panel_label_components("CDKN2B/CDKN2B-AS1") == ["cdkn2b", "cdkn2b-as1"]
    assert _panel_label_components("MYC") == ["myc"]
    assert _panel_label_components("  ,  ") == []


@pytest.mark.parametrize("gene,label", [
    ("MYC", "CASC11,MYC"),
    ("CDKN2A", "CDKN2A,CDKN2B,CDKN2B-AS1"),
    ("CDKN2B", "CDKN2A,CDKN2B,CDKN2B-AS1"),
    ("CDKN2B-AS1", "CDKN2A,CDKN2B,CDKN2B-AS1"),
    ("NF1", "MIR4733HG,NF1"),
    ("ERBB2", "ERBB2,MIR4728"),
    ("MYCN", "MYCN,MYCNOS"),
    ("MLH1", "LRRFIP2,MLH1"),
    ("MSH2", "KCNK12,MSH2"),
    ("MSH6", "FBXO11,MSH6"),
    ("CDK4", "CDK4,MIR6759,TSPAN31"),
    ("BRCA1", "BRCA1,RPL21P4"),
    ("DDX3X", "DDX3X,RN7SL15P"),
    ("MYB", "MYB,MYB-AS1"),
])
def test_composite_labels_match_their_components(gene, label):
    """Every gene a real site configured that the old matcher dropped."""
    assert _panel_label_matches_configured(label, [gene])


def test_ngtd_slash_convention_matches_the_bed_comma_convention():
    assert _panel_label_matches_configured("CDKN2A,CDKN2B,CDKN2B-AS1",
                                           ["CDKN2B/CDKN2B-AS1"]) is False
    # ...but either side's component form matches the other's.
    assert _panel_label_matches_configured("CDKN2B/CDKN2B-AS1", ["CDKN2B"])


def test_unrelated_genes_do_not_match():
    assert not _panel_label_matches_configured("CASC11,MYC", ["EGFR"])
    assert not _panel_label_matches_configured("MYCN,MYCNOS", ["MYC"])
    assert not _panel_label_matches_configured("", ["MYC"])
    assert not _panel_label_matches_configured("MYC", [])


def test_no_match_invents_a_gene_the_label_does_not_contain():
    """Guard against the split making matching too generous."""
    genes = ["MYC", "EGFR", "NF1", "TP53", "RB1", "PTEN", "CDKN2A", "MYB"]
    for label in _panel_labels():
        for gene in genes:
            if _panel_label_matches_configured(label, [gene]):
                assert gene.casefold() in label.casefold(), (gene, label)


def test_suffixed_symbols_still_match_by_prefix():
    """Pre-existing behaviour: a configured symbol matches its -AS1 partner."""
    assert _panel_label_matches_configured("MYB-AS1", ["MYB"])
    assert _panel_label_matches_configured("EGFR_1", ["EGFR"])


def test_case_and_whitespace_insensitive():
    assert _panel_label_matches_configured(" casc11 , myc ", ["MYC"])
    assert _panel_label_matches_configured("CASC11,MYC", ["  myc  "])


def _row(name):
    import pandas as pd

    return pd.Series({"gene": name})


@pytest.mark.parametrize("label,configured,shown", [
    ("CASC11,MYC", ["MYC"], "MYC"),
    ("MIR4733HG,NF1", ["NF1"], "NF1"),
    ("LRRFIP2,MLH1", ["MLH1"], "MLH1"),
    ("KCNK12,MSH2", ["MSH2"], "MSH2"),
    ("FBXO11,MSH6", ["MSH6"], "MSH6"),
    ("MYB,MYB-AS1", ["MYB-AS1"], "MYB-AS1"),
])
def test_configured_gene_wins_the_display_name(label, configured, shown):
    """Otherwise the marker is named for a bystander gene nobody asked for."""
    from robin.reporting.plotting import _panel_target_label

    assert _panel_target_label(_row(label), configured) == shown


def test_unconfigured_targets_keep_the_first_component():
    from robin.reporting.plotting import _panel_target_label

    assert _panel_target_label(_row("EPS8L3,GSTM5"), ["MYC"]) == "EPS8L3"
    assert _panel_target_label(_row("EPS8L3,GSTM5")) == "EPS8L3"


def test_configured_order_breaks_ties():
    """One interval, several requested genes: the first listed one is shown."""
    from robin.reporting.plotting import _panel_target_label

    label = "CDKN2A,CDKN2B,CDKN2B-AS1"
    assert _panel_target_label(_row(label), ["CDKN2A", "CDKN2B"]) == "CDKN2A"
    assert _panel_target_label(_row(label), ["CDKN2B", "CDKN2A"]) == "CDKN2B"


def test_display_name_is_always_a_real_component():
    """Never invent a name the panel does not carry."""
    from robin.reporting.plotting import _panel_target_label

    for label in _panel_labels():
        shown = _panel_target_label(_row(label), ["MYC", "NF1", "CDKN2B", "EGFR"])
        assert shown in [p.strip() for p in label.replace("/", ",").split(",")]


def test_slash_labels_are_handled_too():
    from robin.reporting.plotting import _panel_target_label

    assert _panel_target_label(_row("CDKN2B/CDKN2B-AS1"), ["CDKN2B-AS1"]) == "CDKN2B-AS1"


def test_missing_or_blank_names_are_tolerated():
    import pandas as pd

    from robin.reporting.plotting import _panel_target_label

    assert _panel_target_label(pd.Series({"gene": ""}), ["MYC"]) == ""
    assert _panel_target_label(pd.Series({"gene": "nan"}), ["MYC"]) == ""
    assert _panel_target_label(pd.Series({"other": "MYC"}), ["MYC"]) == ""
