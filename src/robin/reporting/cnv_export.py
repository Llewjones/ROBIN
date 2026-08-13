"""Standalone PDF export of the CNV figures for a finished or in-progress sample.

The live GUI shows CNV as interactive ECharts, but the reporting scientists want
the same figures as vector PDFs they can drop into a case record — the way they
already export the whole-genome methylation (WGM) CNV plots. Rather than
screenshotting the browser, this rebuilds the report's matplotlib figures straight
from the sample directory, so a downloaded plot and the plot in the PDF report are
the same picture.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import natsort
import numpy as np

logger = logging.getLogger(__name__)

# Files written by the CNV analysis stage that the export reads back.
CNV_PLOIDY_NPY = "CNV.npy"
CNV_DICT_NPY = "CNV_dict.npy"
XY_ESTIMATE_PKL = "XYestimate.pkl"


class CnvExportUnavailable(RuntimeError):
    """Raised when a sample has no CNV data to export yet."""


def _load_npy_dict(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:
        logger.debug("Could not load %s", path, exc_info=True)
        return None


def _default_clinical_trial_genes() -> Sequence[str]:
    """Step 2 target genes from workflow config, empty when unavailable."""
    try:
        from robin.workflow_config import get_cnv_clinical_trial_genes

        return get_cnv_clinical_trial_genes()
    except Exception:
        logger.debug("Could not resolve clinical trial genes", exc_info=True)
        return ()


def _load_sex_estimate(sample_dir: Path) -> str:
    path = sample_dir / XY_ESTIMATE_PKL
    if not path.exists():
        return "Unknown"
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    except Exception:
        logger.debug("Could not load %s", path, exc_info=True)
        return "Unknown"
    text = str(value).strip()
    return text or "Unknown"


def load_cnv_export_context(
    sample_dir: Path | str,
    *,
    use_log2: bool,
    configured_genes: Sequence[str] = (),
    reference_contig_scope: Optional[str] = None,
    plot_bin_width: Optional[int] = None,
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    show_trend: bool = True,
    fixed_axis_log2: Optional[float] = None,
    genome_axis_log2: Optional[float] = None,
    gene_label_size: Optional[float] = None,
    gene_labels_rotated: Optional[bool] = None,
    clinical_trial_genes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble everything the CNV figure builders need from a sample directory.

    Raises:
        CnvExportUnavailable: when the sample has no CNV track yet.
    """
    from robin.analysis.cnv_analysis import Result, compute_cnv_log2_from_ploidy
    from robin.analysis.cnv_classification import detect_cnv_events
    from robin.analysis.cnv_regional import (
        analyze_cytoband_cnv,
        build_significant_regions,
        format_chromosome_cnv_status,
        is_reportable_chromosome,
        load_panel_gene_bed,
        load_target_coverage_df,
    )

    sample_dir = Path(sample_dir)
    cnv_map = _load_npy_dict(sample_dir / CNV_PLOIDY_NPY)
    cnv_dict = _load_npy_dict(sample_dir / CNV_DICT_NPY)
    if not cnv_map or not cnv_dict or not cnv_dict.get("bin_width"):
        raise CnvExportUnavailable(
            "No CNV data available for this sample yet — run CNV analysis first."
        )
    if isinstance(cnv_map, dict) and "cnv" in cnv_map:
        cnv_map = cnv_map["cnv"]

    bin_width = int(cnv_dict["bin_width"])
    sex_estimate = _load_sex_estimate(sample_dir)
    log2_cnv = compute_cnv_log2_from_ploidy(cnv_map, sex_estimate)

    # CNV load is a property of the sample, so it is computed once here and
    # carried into whichever export the user asked for.
    from robin.analysis.cnv_regional import compute_cnv_load, cnv_load_summary_text

    cnv_load = compute_cnv_load(
        log2_cnv, bin_width, sex_estimate, cutoff_override=cutoff_override
    )

    chromosomes = [
        contig
        for contig in natsort.natsorted(cnv_map.keys())
        if is_reportable_chromosome(contig)
    ]

    panel_name, panel_genes_df = load_panel_gene_bed(str(sample_dir))
    target_coverage_df = load_target_coverage_df(str(sample_dir))

    # Cytoband-level calls drive both the shaded regions and the per-chromosome
    # captions, exactly as in the PDF report.
    significant_regions: Dict[str, List[dict]] = {}
    chromosome_status: Dict[str, str] = {}
    from robin.analysis.cnv_regional import (
        load_centromere_bed,
        load_cytobands_bed,
        load_gene_bed,
    )

    cytobands_bed = load_cytobands_bed()
    centromere_bed = load_centromere_bed()
    gene_bed = load_gene_bed()

    if not cytobands_bed.empty and not centromere_bed.empty:
        try:
            events = detect_cnv_events(
                cnv_data=log2_cnv,
                bin_width=bin_width,
                sex_estimate=sex_estimate,
                cytobands_df=cytobands_bed,
                gene_df=gene_bed,
                cutoff_override=cutoff_override,
            )
        except Exception:
            logger.debug("CNV event detection failed for export", exc_info=True)
            events = []
        for chrom in chromosomes:
            try:
                cytoband_analysis = analyze_cytoband_cnv(
                    log2_cnv,
                    chrom,
                    cnv_dict,
                    cytobands_bed,
                    centromere_bed,
                    gene_bed,
                    sex_estimate,
                    cutoff_override=cutoff_override,
                )
            except Exception:
                logger.debug("Cytoband analysis failed for %s", chrom, exc_info=True)
                continue
            if cytoband_analysis is None or cytoband_analysis.empty:
                continue
            regions = build_significant_regions(cytoband_analysis)
            if regions:
                significant_regions[chrom] = regions
            chromosome_status[chrom] = format_chromosome_cnv_status(
                chrom, events, cytoband_analysis
            )

    return {
        "result": Result(cnv_map),
        "cnv_dict": cnv_dict,
        "normalized_cnv": log2_cnv,
        "use_log2": bool(use_log2),
        "sex_estimate": sex_estimate,
        "panel_name": panel_name,
        "panel_genes_df": panel_genes_df,
        "target_coverage_df": target_coverage_df,
        "cnv_load": cnv_load,
        "cnv_load_text": cnv_load_summary_text(cnv_load),
        "significant_regions": significant_regions,
        "chromosome_status": chromosome_status,
        "chromosomes": chromosomes,
        "configured_genes": tuple(configured_genes or ()),
        "reference_contig_scope": reference_contig_scope,
        "plot_bin_width": plot_bin_width,
        # Display settings carried over from the live view so a downloaded plot
        # matches what the user is looking at.
        "cutoff_override": cutoff_override,
        "outliers_only": bool(outliers_only),
        "show_trend": bool(show_trend),
        "fixed_axis_log2": fixed_axis_log2,
        "genome_axis_log2": genome_axis_log2,
        "gene_label_size": gene_label_size,
        "gene_labels_rotated": gene_labels_rotated,
        "clinical_trial_genes": tuple(
            clinical_trial_genes
            if clinical_trial_genes is not None
            else _default_clinical_trial_genes()
        ),
    }


def export_cnv_genome_pdf(context: Dict[str, Any]) -> bytes:
    """Genome-wide CNV summary as a one-page vector PDF."""
    from robin.reporting.plotting import create_CNV_genome_pdf

    return create_CNV_genome_pdf(
        context["result"],
        context["cnv_dict"],
        normalized_cnv=context["normalized_cnv"],
        use_normalized_difference=context["use_log2"],
        plot_bin_width=context.get("plot_bin_width"),
        sex_estimate=context["sex_estimate"],
        panel_genes_df=context["panel_genes_df"],
        target_coverage_df=context["target_coverage_df"],
        significant_regions=context["significant_regions"],
        reference_contig_scope=context.get("reference_contig_scope"),
        configured_genes=context["configured_genes"],
        clinical_trial_genes=context.get("clinical_trial_genes", ()),
        cutoff_override=context.get("cutoff_override"),
        outliers_only=context.get("outliers_only", False),
        show_trend=context.get("show_trend", True),
        fixed_axis_log2=context.get("genome_axis_log2"),
        header_text=context.get("cnv_load_text"),
        **(
            {"gene_label_size": context["gene_label_size"]}
            if context.get("gene_label_size")
            else {}
        ),
        **(
            {"gene_labels_rotated": bool(context["gene_labels_rotated"])}
            if context.get("gene_labels_rotated") is not None
            else {}
        ),
    )


def export_cnv_chromosome_pdf(
    context: Dict[str, Any],
    *,
    chromosomes: Optional[Sequence[str]] = None,
    fig_height: float = 3.0,
    fig_width: float = 9.0,
    full_range_pages: bool = True,
) -> bytes:
    """Per-chromosome CNV plots as a multi-page vector PDF.

    Pages are larger than the four-per-page report layout because a downloaded
    plot is read on its own rather than stacked on an A4 page.

    A chromosome carrying bins outside the fixed Y window gets a **second page**
    fitted to the data, so the selected granularity and the true depth of an
    extreme event are both available without compromising either.
    """
    from robin.reporting.plotting import create_CNV_chromosome_pdf

    selected = list(chromosomes) if chromosomes else context["chromosomes"]
    return create_CNV_chromosome_pdf(
        context["result"],
        context["cnv_dict"],
        significant_regions=context["significant_regions"],
        chromosomes=selected,
        panel_genes_df=context["panel_genes_df"],
        chromosome_status=context["chromosome_status"],
        normalized_cnv=context["normalized_cnv"],
        target_coverage_df=context["target_coverage_df"],
        use_log2_ratio=context["use_log2"],
        fixed_axis_log2=context.get("fixed_axis_log2"),
        plot_bin_width=context.get("plot_bin_width"),
        sex_estimate=context["sex_estimate"],
        fig_height=fig_height,
        fig_width=fig_width,
        reference_contig_scope=context.get("reference_contig_scope"),
        configured_genes=context["configured_genes"],
        clinical_trial_genes=context.get("clinical_trial_genes", ()),
        cutoff_override=context.get("cutoff_override"),
        outliers_only=context.get("outliers_only", False),
        show_trend=context.get("show_trend", True),
        full_range_pages=full_range_pages,
        header_text=context.get("cnv_load_text"),
        **(
            {"gene_label_size": context["gene_label_size"]}
            if context.get("gene_label_size")
            else {}
        ),
        **(
            {"gene_labels_rotated": bool(context["gene_labels_rotated"])}
            if context.get("gene_labels_rotated") is not None
            else {}
        ),
    )


def build_cnv_pdf(
    sample_dir: Path | str,
    *,
    kind: str,
    use_log2: bool,
    configured_genes: Sequence[str] = (),
    reference_contig_scope: Optional[str] = None,
    plot_bin_width: Optional[int] = None,
    chromosomes: Optional[Sequence[str]] = None,
    cutoff_override: Optional[float] = None,
    outliers_only: bool = False,
    show_trend: bool = True,
    fixed_axis_log2: Optional[float] = None,
    genome_axis_log2: Optional[float] = None,
    gene_label_size: Optional[float] = None,
    gene_labels_rotated: Optional[bool] = None,
) -> Tuple[bytes, str]:
    """Build a CNV PDF for a sample. Returns ``(pdf_bytes, suggested_filename)``.

    ``kind`` is ``genome`` for the whole-genome summary or ``chromosomes`` for the
    per-chromosome pages. Safe to call off the UI thread.
    """
    sample_dir = Path(sample_dir)
    context = load_cnv_export_context(
        sample_dir,
        use_log2=use_log2,
        configured_genes=configured_genes,
        reference_contig_scope=reference_contig_scope,
        plot_bin_width=plot_bin_width,
        cutoff_override=cutoff_override,
        outliers_only=outliers_only,
        show_trend=show_trend,
        fixed_axis_log2=fixed_axis_log2,
        genome_axis_log2=genome_axis_log2,
        gene_label_size=gene_label_size,
        gene_labels_rotated=gene_labels_rotated,
    )
    scale = "log2" if use_log2 else "ploidy"
    sample_name = sample_dir.name or "sample"
    if kind == "genome":
        return (
            export_cnv_genome_pdf(context),
            f"{sample_name}_CNV_genome_{scale}.pdf",
        )
    if kind == "chromosomes":
        return (
            export_cnv_chromosome_pdf(context, chromosomes=chromosomes),
            f"{sample_name}_CNV_chromosomes_{scale}.pdf",
        )
    raise ValueError(f"Unknown CNV PDF kind: {kind!r}")
