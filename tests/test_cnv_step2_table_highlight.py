"""Step 2 targets are highlighted in the NGTD / log2 table.

The table still lists every configured target - that requirement is unchanged.
Step 2 genes are marked purple, the same convention the plots use for their
gene markers, so the table and the figures read alike.
"""

import pytest

from robin.reporting.plotting import _panel_label_matches_configured
from robin.reporting.sections.cnv import CNV_REPORT_TRIAL_LEGEND_COLOR
from robin.workflow_config import get_cnv_clinical_trial_genes


def test_purple_matches_the_plot_convention():
    from robin.reporting.plotting import CNV_COLORS

    assert CNV_REPORT_TRIAL_LEGEND_COLOR == CNV_COLORS["plot_trial"]


@pytest.mark.parametrize("gene", ["ERBB2", "MTAP", "FGFR1", "BRCA1"])
def test_step2_genes_are_recognised(gene):
    assert _panel_label_matches_configured(gene, tuple(get_cnv_clinical_trial_genes()))


@pytest.mark.parametrize("gene", ["TP53", "PTEN", "ATRX"])
def test_non_step2_genes_are_not(gene):
    assert not _panel_label_matches_configured(
        gene, tuple(get_cnv_clinical_trial_genes())
    )
