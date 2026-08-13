"""The report body uses the zoomed chromosome axis; the PDF carries both views.

Fitting the report's per-chromosome panels to the full data range flattens
ordinary gains and losses into the middle of the panel to accommodate a handful
of extreme bins. The report therefore draws the fixed window, and the
downloadable per-chromosome PDF adds a full-range page whenever bins fall
outside it, so nothing is lost.
"""

import inspect

from robin.cnv_plot_style import CNV_CHROMOSOME_AXIS_LOG2


def test_report_requests_the_fixed_window():
    import robin.reporting.sections.cnv as section

    source = inspect.getsource(section)
    assert "fixed_axis_log2=CNV_CHROMOSOME_AXIS_LOG2" in source
    assert "full_range_axis=False" in source
    assert "fixed_axis_log2=None" not in source


def test_pdf_export_keeps_full_range_pages():
    from robin.reporting.cnv_export import export_cnv_chromosome_pdf

    parameters = inspect.signature(export_cnv_chromosome_pdf).parameters
    assert parameters["full_range_pages"].default is True


def test_plot_default_axis_is_the_fixed_window():
    from robin.reporting.plotting import iter_CNV_chromosome_figures

    parameters = inspect.signature(iter_CNV_chromosome_figures).parameters
    assert parameters["fixed_axis_log2"].default == CNV_CHROMOSOME_AXIS_LOG2
    assert parameters["full_range_axis"].default is False
