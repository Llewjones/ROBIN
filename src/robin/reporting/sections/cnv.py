"""
CNV Analysis Section for ROBIN Reports.

This module handles the Copy Number Variation (CNV) analysis section of the report.
"""

import os
import re
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import natsort
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    NextPageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from ..sections.base import ReportSection
from robin.cnv_plot_style import CNV_CHROMOSOME_AXIS_LOG2
from ..plotting import (
    create_CNV_plot,
    create_CNV_plot_per_chromosome,
    cnv_chromosome_fig_height_for_page,
    cnv_chromosome_fig_width_for_page,
    CNV_CHROMOSOME_PLOT_SPACER_PT,
    CNV_CHROMOSOME_PLOTS_PER_PAGE,
    CNV_REPORT_FRAME_PADDING_PT,
)

# from robin.subpages.CNVObjectClass import (
#    CNVAnalysis
# )

#: Width:height of the genome-wide CNV summary on its A4 landscape page.
#: A deep panel makes a genome profile look stretched, since the data sits in a
#: narrow band with empty axis above and below it.
GENOME_SUMMARY_ASPECT = 3.2

#: Purple used for the Step 2 legend, matching the marker colour on
#: the figures so the note and the genes it describes read as one thing.
CNV_REPORT_TRIAL_LEGEND_COLOR = "#7E22CE"


def _image_aspect(img_buf, fallback_ratio: float) -> float:
    """Height/width of a rendered figure, so it can be placed without stretching."""
    try:
        from reportlab.lib.utils import ImageReader

        img_buf.seek(0)
        width_px, height_px = ImageReader(img_buf).getSize()
        img_buf.seek(0)
        if width_px:
            return float(height_px) / float(width_px)
    except Exception:
        logger.debug("Could not measure figure aspect", exc_info=True)
    return 1.0 / float(fallback_ratio)


def _clinical_trial_genes(report) -> tuple:
    """Step 2 target genes for this run, from workflow config."""
    try:
        from robin.workflow_config import get_cnv_clinical_trial_genes

        return tuple(get_cnv_clinical_trial_genes())
    except Exception:
        logger.debug("Could not resolve clinical trial genes", exc_info=True)
        return ()


def _report_cutoff_override(report) -> "float | None":
    """Review cut-off set in the admin panel, or None to use the calling thresholds.

    The same value drives the drawn lines, the point colouring, the CNV load and
    the figure caption, so a report cannot show a cut-off it did not draw at.
    """
    return getattr(report, "cnv_cutoff", None)


def _cutoff_text(report) -> str:
    """Bare cut-off label, for prose rather than a heading."""
    from robin.gui.plotting_preferences import cnv_cutoff_label

    return cnv_cutoff_label(_report_cutoff_override(report))


def _cutoff_suffix(report) -> str:
    """Cut-off note for a table heading whose contents move with the cut-off."""
    from robin.gui.plotting_preferences import cnv_cutoff_heading_suffix

    return cnv_cutoff_heading_suffix(_report_cutoff_override(report))


def _gene_label_size_kwargs(report) -> dict:
    """Gene label styling from admin plotting preferences, when configured."""
    kwargs = {}
    size = getattr(report, "cnv_gene_label_size", None)
    if size:
        kwargs["gene_label_size"] = size
    rotated = getattr(report, "cnv_gene_labels_rotated", None)
    if rotated is not None:
        kwargs["gene_labels_rotated"] = bool(rotated)
    return kwargs


from robin.analysis.cnv_analysis import (
    Result,
    estimate_cnv_baseline_shift,
    moving_average,
    CNV_Difference,
    compute_cnv_log2_from_ploidy,
    prepare_cnv_calling_track,
    resolve_cnv_calling_bin_width,
)
from robin.analysis.cnv_classification import detect_cnv_events, get_cnv_summary, CNVEvent
from robin.analysis.cnv_regional import (
    CNV_GENE_NO_DATA_LABEL,
    CNV_GENE_UNKNOWN_LABEL,
    CNV_GENE_NOT_LOCATED_LABEL,
    CNV_LOAD_TITLE,
    CNV_NO_EVENT_LABEL,
    compute_target_gene_cnv_states,
    SIGNIFICANT_CNV_STATES,
    analyze_cytoband_cnv,
    compute_cnv_load,
    build_regional_cnv_events,
    build_significant_regions,
    format_chromosome_cnv_status,
    format_panel_genes_for_table,
    is_reportable_chromosome,
    load_panel_gene_bed,
    load_target_coverage_df,
    panel_genes_in_region,
)
from robin.reference_contigs import is_visible_contig
from robin.classification_config import (
    is_resolution_sufficient,
    resolve_cnv_thresholds,
)
from robin.workflow_config import get_cnv_genes, load_workflow_toml

from robin import resources

logger = logging.getLogger(__name__)


def _resolve_configured_cnv_genes(output_dir: str) -> tuple[str, ...]:
    """Resolve ``[cnv].genes`` for report overlays from env/TOML or sample pointer."""
    genes = get_cnv_genes()
    if genes:
        return genes
    pointer = Path(output_dir) / ".robin_workflow_toml"
    if not pointer.is_file():
        return ()
    try:
        toml_path = Path(pointer.read_text(encoding="utf-8").strip())
        if toml_path.is_file():
            return get_cnv_genes(load_workflow_toml(toml_path))
    except Exception:
        logger.debug(
            "Could not resolve [cnv].genes from sample workflow pointer",
            exc_info=True,
        )
    return ()


def calculate_chromosome_stats(result, ref_result, XYestimate):
    """Calculate chromosome-wide statistics and baselines.

    Args:
        result: CNV result object for sample
        ref_result: CNV result object for reference

    Returns:
        Dictionary of chromosome statistics including means, baselines, and thresholds
    """
    stats = {}
    autosome_means = []

    # Calculate normalized values and stats for each chromosome
    for chrom in result.cnv.keys():
        if chrom != "chrM" and chrom in ref_result:
            # Calculate normalized CNV values
            sample_avg = moving_average(result.cnv[chrom])
            ref_avg = moving_average(ref_result[chrom])

            # Pad arrays if needed
            max_len = max(len(sample_avg), len(ref_avg))
            if len(sample_avg) < max_len:
                sample_avg = np.pad(sample_avg, (0, max_len - len(sample_avg)))
            if len(ref_avg) < max_len:
                ref_avg = np.pad(ref_avg, (0, max_len - len(ref_avg)))

            # Calculate normalized CNV
            normalized_cnv = sample_avg - ref_avg

            # Calculate basic statistics
            chr_mean = np.mean(normalized_cnv)
            chr_std = np.std(normalized_cnv)

            # Store autosome means for global statistics
            if chrom.startswith("chr") and chrom[3:].isdigit():
                autosome_means.append(chr_mean)

            # Set baseline and thresholds based on chromosome and sex
            if chrom == "chrX":
                if XYestimate == "XX":  # Female
                    baseline = 1.0  # Expected +1 relative to male control
                else:  # Male
                    baseline = 0.0  # Expected same as male control
            elif chrom == "chrY":
                if XYestimate == "XY":  # Male
                    baseline = 0.0
                else:  # Female
                    baseline = -1.0  # Expected absence
            else:  # Autosomes
                baseline = 0.0

            stats[chrom] = {
                "mean": chr_mean,
                "std": chr_std,
                "baseline": baseline,
                "normalized_cnv": normalized_cnv,
            }

    # Calculate global autosome statistics
    global_mean = np.mean(autosome_means)
    global_std = np.std(autosome_means)

    # Store global stats
    stats["global"] = {"mean": global_mean, "std": global_std}

    # self.chromosome_stats = stats
    return stats


class CNVSection(ReportSection):
    """Section containing the CNV analysis."""

    FULL_PLOT_WIDTH = inch * 7.5
    CHROMOSOME_PLOTS_PER_PAGE = CNV_CHROMOSOME_PLOTS_PER_PAGE
    CHROMOSOME_PLOT_SPACER = CNV_CHROMOSOME_PLOT_SPACER_PT

    def _chromosome_pdf_plot_height(self) -> float:
        """Height in inches for one per-chromosome plot (four per PDF page)."""
        return cnv_chromosome_fig_height_for_page(
            self.report.doc.height / inch,
            plots_per_page=self.CHROMOSOME_PLOTS_PER_PAGE,
            spacer_pt=self.CHROMOSOME_PLOT_SPACER,
            frame_padding_pt=CNV_REPORT_FRAME_PADDING_PT,
        )

    def _chromosome_pdf_plot_width(self) -> float:
        """Width in inches for one per-chromosome plot within the PDF frame."""
        return cnv_chromosome_fig_width_for_page(
            self.report.doc.width / inch,
            frame_padding_pt=CNV_REPORT_FRAME_PADDING_PT,
        )

    NGTD_TABLE_TITLE = "NGTD and Step 2 Targets CNVs"

    def _append_ngtd_target_table(
        self, log2_cnv, bin_width: int, sex_estimate: str, panel_genes_df
    ) -> None:
        """Every NGTD / Step 2 target and its CNV state, event or not.

        Targets with nothing on them are listed too, so the table reads as
        "these were looked at" rather than leaving the reader to infer it from
        an absence.
        """
        try:
            from robin.workflow_config import get_cnv_ngtd_genes

            genes = tuple(get_cnv_ngtd_genes())
        except Exception:
            logger.debug("Could not resolve NGTD target genes", exc_info=True)
            return
        if not genes:
            return
        try:
            # Panel first (what was actually targeted), then the genome-wide
            # reference so an off-panel target can still be placed and named as
            # off-panel rather than as unlocatable.
            from robin.analysis.cnv_regional import load_gene_bed

            frames = [
                f
                for f in (panel_genes_df, load_gene_bed())
                if f is not None and not f.empty
            ]
            rows = compute_target_gene_cnv_states(
                log2_cnv,
                int(bin_width),
                genes,
                sex_estimate,
                gene_frames=frames or None,
                cutoff_override=_report_cutoff_override(self.report),
            )
        except Exception:
            logger.debug("Could not resolve NGTD target CNV states", exc_info=True)
            return
        if not rows:
            return

        title_style = ParagraphStyle(
            "CNVTableTitle",
            parent=self.styles.styles["Normal"],
            fontSize=9,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        )
        self.elements.append(
            Paragraph(
                self.NGTD_TABLE_TITLE + _cutoff_suffix(self.report), title_style
            )
        )
        self.elements.append(Spacer(1, 2))
        # Purple marks a Step 2 target, matching the gene markers on the plots so
        # the table and the figures read the same way.
        try:
            from robin.reporting.plotting import _panel_label_matches_configured
            from robin.workflow_config import get_cnv_clinical_trial_genes

            step2_genes = tuple(get_cnv_clinical_trial_genes())
        except Exception:
            logger.debug("Could not resolve Step 2 target genes", exc_info=True)
            step2_genes = ()

        table_rows = [["Gene", "Chr", "Log2 ratio", "Result"]]
        any_step2 = False
        for row in rows:
            gene = str(row["gene"])
            if step2_genes and _panel_label_matches_configured(gene, step2_genes):
                any_step2 = True
                gene = f'<font color="{CNV_REPORT_TRIAL_LEGEND_COLOR}"><b>{gene}</b></font>'
            table_rows.append(
                [
                    gene,
                    row["chrom"] or "--",
                    f"{row['value']:+.2f}" if row["value"] is not None else "--",
                    row["state"],
                ]
            )
        self.elements.append(
            self.create_table(table_rows, auto_col_width=True, compact=True)
        )
        self.elements.append(Spacer(1, 4))
        self.elements.append(
            Paragraph(
                f"Every configured target is listed. &quot;{CNV_NO_EVENT_LABEL}&quot; means the "
                f"gene was assessed and no gain or loss crossed the calling cut-off. "
                f"&quot;{CNV_GENE_NOT_LOCATED_LABEL}&quot; means it is not a target on this "
                f"sample's panel and so was not assessed \u2014 that is not the same as no change. "
                f"The value shown is the most extreme bin overlapping the gene, matching the "
                f"gene markers on the plots."
                + (
                    " Genes in purple are current Step 2 targets."
                    if any_step2
                    else ""
                ),
                ParagraphStyle(
                    "NGTDNote",
                    parent=self.styles.styles["Normal"],
                    fontSize=7,
                    textColor=colors.HexColor("#475569"),
                    spaceAfter=6,
                ),
            )
        )

    def _append_cnv_load_block(self) -> None:
        """CNV load as its own titled block in the detailed analysis section.

        Counted at the cut-off in force, so it agrees with the lines on the
        figures and with the event tables beneath it, and labelled with that
        cut-off the same way they are.
        """
        load = getattr(self, "cnv_load", None)
        if not load or not load.get("assessed_mb"):
            return
        title_style = ParagraphStyle(
            "CNVTableTitle",
            parent=self.styles.styles["Normal"],
            fontSize=9,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        )
        heading = CNV_LOAD_TITLE + _cutoff_suffix(self.report)
        self.elements.append(Paragraph(heading, title_style))
        self.elements.append(Spacer(1, 2))
        rows = [["", "Genome affected", "Span"]]
        for label, pct_key, mb_key in (
            ("Total", "total_percent", "total_mb"),
            ("Gain", "gain_percent", "gain_mb"),
            ("Loss", "loss_percent", "loss_mb"),
        ):
            rows.append(
                [label, f"{load[pct_key]:.1f}%", f"{load[mb_key]:,.0f} Mb"]
            )
        rows.append(["Assessed", "", f"{load['assessed_mb']:,.0f} Mb"])
        self.elements.append(self.create_table(rows, auto_col_width=True, compact=True))
        self.elements.append(Spacer(1, 4))
        self.elements.append(
            Paragraph(
                "CNV load is the proportion of the assessed genome whose log2 ratio "
                "crosses the configured calling thresholds — the same cut-off drawn "
                "on the plots. Bins without data are excluded from both the "
                "proportion and the assessed span.",
                ParagraphStyle(
                    "CNVLoadNote",
                    parent=self.styles.styles["Normal"],
                    fontSize=7,
                    textColor=colors.HexColor("#475569"),
                    spaceAfter=6,
                ),
            )
        )

    def add_content(self):
        """Add the CNV analysis content to the report."""
        logger.debug("Starting CNV section processing")

        # Load CNV data and XYestimate
        XYestimate = "Unknown"  # Default value
        cnv_file = os.path.join(self.report.output, "CNV.npy")

        # Check for required files
        if not os.path.exists(cnv_file):
            logger.error("No CNV.npy file found in output directory")
            return

        # Load CNV data
        logger.debug("Loading CNV data from %s", cnv_file)
        CNVresult = np.load(cnv_file, allow_pickle="TRUE").item()
        CNVresult = Result(CNVresult)
        logger.debug("CNV data loaded with keys: %s", list(CNVresult.cnv.keys())[:5])

        cnv_dict = np.load(
            os.path.join(self.report.output, "CNV_dict.npy"), allow_pickle=True
        ).item()
        logger.debug("CNV dict loaded with keys: %s", list(cnv_dict.keys()))

        # Store cnv_dict in report for use by other methods
        self.report.cnv_dict = cnv_dict

        # Load XY estimate if available
        if os.path.exists(os.path.join(self.report.output, "XYestimate.pkl")):
            with open(os.path.join(self.report.output, "XYestimate.pkl"), "rb") as file:
                XYestimate = pickle.load(file)
                logger.debug("Loaded XY estimate: %s", XYestimate)

        # Add CNV section header
        logger.debug("Adding CNV section header")

        # Start detailed analysis section
        self.elements.append(PageBreak())
        self.elements.append(
            Paragraph(
                "Copy Number Variation Detailed Analysis",
                self.styles.styles["Heading2"],
            )
        )

        try:
            # Initialize CNVAnalysis object with the same settings as UI
            # cnv_analyzer = CNVAnalysis(target_panel="rCNS2")
            # cnv_analyzer.XYestimate = XYestimate

            # Load required resource files
            gene_bed_file = os.path.join(
                os.path.dirname(os.path.abspath(resources.__file__)), "unique_genes.bed"
            )
            cytoband_file = os.path.join(
                os.path.dirname(os.path.abspath(resources.__file__)), "cytoBand.txt"
            )
            logger.debug(
                "Resource files: gene_bed=%s, cytoband=%s", gene_bed_file, cytoband_file
            )

            # Load gene and cytoband data
            gene_bed = None
            cytobands_bed = None
            if os.path.exists(gene_bed_file):
                gene_bed = pd.read_csv(
                    gene_bed_file,
                    sep="\t",
                    names=["chrom", "start_pos", "end_pos", "gene"],
                )
                logger.debug("Loaded gene bed file with shape: %s", gene_bed.shape)
            if os.path.exists(cytoband_file):
                cytobands_bed = pd.read_csv(
                    cytoband_file,
                    sep="\t",
                    names=["chrom", "start_pos", "end_pos", "name", "stain"],
                )
                logger.debug("Loaded cytoband file with shape: %s", cytobands_bed.shape)

            # Set up CNVAnalysis object with loaded data
            # cnv_analyzer.gene_bed = gene_bed
            # cnv_analyzer.cytobands_bed = cytobands_bed
            # cnv_analyzer.cnv_dict = cnv_dict

            # Get reference CNV data with matching bin width
            logger.debug(
                "Getting reference CNV data with bin width %s", cnv_dict["bin_width"]
            )

            r2_cnv = Result(
                np.load(
                    os.path.join(self.report.output, "CNV2.npy"), allow_pickle="TRUE"
                ).item()
            ).cnv

            use_normalized_summary = getattr(
                self.report, "cnv_summary_normalized", False
            )
            trial_genes = _clinical_trial_genes(self.report)
            # Always computed, whatever scale the figures are drawn on. The
            # regional table, CNV load, the NGTD table and the gene calls all
            # work on this track, so tying it to the plotting preference left
            # them silently absent whenever the report was set to ploidy.
            log2_cnv = compute_cnv_log2_from_ploidy(CNVresult.cnv, XYestimate)
            # Recorded so the figure can state it: a baseline correction that
            # moves every value is not something a report should apply silently.
            try:
                baseline_shift = estimate_cnv_baseline_shift(
                    compute_cnv_log2_from_ploidy(
                        CNVresult.cnv, XYestimate, centre_baseline=False
                    )
                )
            except Exception:
                logger.debug("Could not resolve the CNV baseline shift", exc_info=True)
                baseline_shift = 0.0

            # Initialize CNV_Difference object for normalized values
            result3 = CNV_Difference()

            # Calculate normalized CNV values
            logger.debug("Calculating normalized CNV values")
            for key in CNVresult.cnv.keys():
                if key != "chrM" and re.match(r"^chr(\d+|X|Y)$", key):
                    if key in r2_cnv:
                        moving_avg_data1 = moving_average(CNVresult.cnv[key])
                        moving_avg_data2 = moving_average(r2_cnv[key])
                        # Pad arrays if needed
                        if len(moving_avg_data1) != len(moving_avg_data2):
                            max_len = max(len(moving_avg_data1), len(moving_avg_data2))
                            if len(moving_avg_data1) < max_len:
                                moving_avg_data1 = np.pad(
                                    moving_avg_data1,
                                    (0, max_len - len(moving_avg_data1)),
                                )
                            if len(moving_avg_data2) < max_len:
                                moving_avg_data2 = np.pad(
                                    moving_avg_data2,
                                    (0, max_len - len(moving_avg_data2)),
                                )
                        # Calculate difference
                        result3.cnv[key] = moving_avg_data1 - moving_avg_data2

            # Set the result3 in the analyzer
            # cnv_analyzer.result3 = result3

            # Calculate chromosome statistics using CNVAnalysis logic
            chromosome_stats = calculate_chromosome_stats(CNVresult, r2_cnv, XYestimate)
            # cnv_analyzer.chromosome_stats = chromosome_stats

            # Add gain/loss thresholds to chromosome stats using centralized rules
            for chrom, stats in chromosome_stats.items():
                if chrom != "global":
                    gain_threshold, loss_threshold = resolve_cnv_thresholds(
                        chrom, XYestimate, _report_cutoff_override(self.report)
                    )
                    stats["gain_threshold"] = gain_threshold
                    stats["loss_threshold"] = loss_threshold

            # Add Summary Card
            logger.debug("Adding CNV summary card")
            # Create summary card table data
            summary_data = []

            # Add genetic sex row (simplified)
            summary_data.append(
                [
                    Paragraph("Genetic Sex:", self.styles.styles["Normal"]),
                    Paragraph(XYestimate, self.styles.styles["Normal"]),
                ]
            )

            # Add analysis metrics
            summary_data.append(
                [
                    Paragraph("Bin Width:", self.styles.styles["Normal"]),
                    Paragraph(
                        f"{cnv_dict['bin_width']:,}", self.styles.styles["Normal"]
                    ),
                ]
            )
            summary_data.append(
                [
                    Paragraph("Variance:", self.styles.styles["Normal"]),
                    Paragraph(
                        f"{cnv_dict.get('variance', 0):.2f}",
                        self.styles.styles["Normal"],
                    ),
                ]
            )
            centromeres_file = os.path.join(
                os.path.dirname(os.path.abspath(resources.__file__)),
                "cenSatRegions.bed",
            )

            centromere_bed = pd.read_csv(
                centromeres_file,
                usecols=[0, 1, 2, 3],
                names=["chrom", "start_pos", "end_pos", "name"],
                header=None,
                sep=r"\s+",
            )

            panel_name, panel_genes_df = load_panel_gene_bed(self.report.output)
            target_coverage_df = load_target_coverage_df(self.report.output)
            configured_genes = _resolve_configured_cnv_genes(self.report.output)
            scope = getattr(self.report, "reference_contig_scope", None)
            reportable_chromosomes = [
                chrom
                for chrom in natsort.natsorted(result3.cnv.keys())
                if is_visible_contig(chrom, scope)
            ]
            cytoband_analysis_by_chrom: dict[str, pd.DataFrame] = {}
            regional_cnv_events: list[dict] = []
            for chrom in reportable_chromosomes:
                cytoband_analysis = analyze_cytoband_cnv(
                    log2_cnv,
                    chrom,
                    cnv_dict,
                    cytobands_bed,
                    centromere_bed,
                    gene_bed if gene_bed is not None else pd.DataFrame(
                        columns=["chrom", "start_pos", "end_pos", "gene"]
                    ),
                    XYestimate,
                    cutoff_override=_report_cutoff_override(self.report),
                )
                cytoband_analysis_by_chrom[chrom] = cytoband_analysis
                regional_cnv_events.extend(
                    build_regional_cnv_events(cytoband_analysis, panel_genes_df)
                )

            # Calculate gene counts
            total_gained_genes = set()
            total_lost_genes = set()
            for chrom in natsort.natsorted(log2_cnv.keys()):
                if chrom != "chrM" and re.match(r"^chr(\d+|X|Y)$", chrom):
                    analysis = cytoband_analysis_by_chrom.get(chrom)
                    if analysis is None:
                        analysis = analyze_cytoband_cnv(
                            log2_cnv,
                            chrom,
                            cnv_dict,
                            cytobands_bed,
                            centromere_bed,
                            gene_bed if gene_bed is not None else pd.DataFrame(
                                columns=["chrom", "start_pos", "end_pos", "gene"]
                            ),
                            XYestimate,
                            cutoff_override=_report_cutoff_override(self.report),
                        )
                    if not analysis.empty:
                        # Get genes in gained regions (including HIGH_GAIN)
                        gained = analysis[
                            analysis["cnv_state"].isin(["GAIN", "HIGH_GAIN"])
                        ]
                        for _, row in gained.iterrows():
                            if row["genes"]:
                                total_gained_genes.update(row["genes"])

                        # Get genes in lost regions (including DEEP_LOSS)
                        lost = analysis[
                            analysis["cnv_state"].isin(["LOSS", "DEEP_LOSS"])
                        ]
                        for _, row in lost.iterrows():
                            if row["genes"]:
                                total_lost_genes.update(row["genes"])

            # CNV load gets its own block alongside the detailed events below.
            try:
                # Same cut-off the figures are drawn at, so the load agrees with
                # the lines above it. compute_cnv_load reports which threshold it
                # used, and that label is printed with the block.
                self.cnv_load = compute_cnv_load(
                    log2_cnv,
                    int(cnv_dict["bin_width"]),
                    str(XYestimate),
                    cutoff_override=_report_cutoff_override(self.report),
                )
            except Exception:
                logger.debug("Could not compute CNV load", exc_info=True)
                self.cnv_load = None

            # Add gene counts to summary
            summary_data.append(
                [
                    Paragraph("Genes in Gained Regions:", self.styles.styles["Normal"]),
                    Paragraph(
                        str(len(total_gained_genes)), self.styles.styles["Normal"]
                    ),
                ]
            )
            summary_data.append(
                [
                    Paragraph("Genes in Lost Regions:", self.styles.styles["Normal"]),
                    Paragraph(str(len(total_lost_genes)), self.styles.styles["Normal"]),
                ]
            )

            # Create summary table with styling
            if summary_data:
                formatted_summary_data = []
                for row in summary_data:
                    formatted_row = [
                        row[0].text if hasattr(row[0], "text") else str(row[0]),
                        row[1].text if hasattr(row[1], "text") else str(row[1]),
                    ]
                    formatted_summary_data.append(formatted_row)

                summary_table = self.create_table(
                    formatted_summary_data,
                    auto_col_width=True,
                    compact=True,
                    font_size=9,
                )
                # Add specific styling while preserving modern table style
                summary_table.setStyle(
                    TableStyle(
                        [
                            *self.MODERN_TABLE_STYLE._cmds,
                            (
                                "ALIGN",
                                (1, 0),
                                (1, -1),
                                "RIGHT",
                            ),  # Right-align the count column
                            (
                                "FONTNAME",
                                (0, 0),
                                (-1, -1),
                                "Helvetica-Bold",
                            ),  # Bold font for all cells
                            (
                                "FONTSIZE",
                                (0, 0),
                                (-1, -1),
                                9,
                            ),  # Unified font size
                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                4,
                            ),
                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                4,
                            ),
                        ]
                    )
                )
                self.elements.append(summary_table)

            # Detect CNV events using centralized classification rules
            logger.info("Detecting CNV events using centralized rules")
            events = []
            
            # Check if resolution is sufficient
            analysis_binw = int(cnv_dict.get("bin_width", 1000000))
            calling_binw = resolve_cnv_calling_bin_width(analysis_binw)
            if not is_resolution_sufficient(calling_binw):
                logger.warning("Resolution insufficient for CNV calling")
                summary_whole_chr_events = []
                summary_arm_events = []
            else:
                # Arm/whole-chromosome events: log2(ploidy / expected), ≥1 Mb bins
                analysis_binw = int(cnv_dict.get("bin_width", 1000000))
                calling_cnv, calling_binw = prepare_cnv_calling_track(
                    CNVresult.cnv, analysis_binw, XYestimate
                )
                events = detect_cnv_events(
                    cnv_data=calling_cnv,
                    bin_width=calling_binw,
                    sex_estimate=XYestimate,
                    cytobands_df=cytobands_bed,
                    gene_df=gene_bed,
                    cutoff_override=_report_cutoff_override(self.report),
                )
                
                # Convert events to summary format
                summary_whole_chr_events = []
                summary_arm_events = []
                
                for event in events:
                    if event.event_type.startswith("WHOLE_CHR_"):
                        event_type = event.event_type.replace("WHOLE_CHR_", "")
                        summary_whole_chr_events.append(
                            f"Chromosome {event.chromosome[3:]}: {event_type} (mean={event.mean_cnv:.2f})"
                        )
                        logger.info(f"Detected whole chromosome {event_type} for {event.chromosome}")
                    else:
                        arm_label = f"{event.arm}-arm" if event.arm else "arm"
                        summary_arm_events.append(
                            f"Chromosome {event.chromosome[3:]} {arm_label}: {event.event_type} (mean={event.mean_cnv:.2f}, {event.proportion_affected:.0%} of arm)"
                        )
                        logger.info(f"Detected arm event: {event.chromosome} {arm_label} {event.event_type}")
                
                # Log the final counts
                logger.info(f"Found {len(summary_arm_events)} arm events")
                logger.info(f"Found {len(summary_whole_chr_events)} whole chromosome events")

            self.summary_elements.append(
                Paragraph(
                    "Copy Number Variation Summary",
                    ParagraphStyle(
                        "SummaryHeader",
                        parent=self.styles.styles["Heading3"],
                        fontSize=12,
                        fontName="Helvetica-Bold",
                        textColor=self.styles.COLORS["primary"],
                        spaceAfter=12,
                    ),
                )
            )

            # Add whole chromosome events to summary
            if summary_whole_chr_events:
                self.summary_elements.append(
                    Paragraph(
                        f"Whole Chromosome Events{_cutoff_suffix(self.report)}:<br/> "
                        + " <br/> ".join(summary_whole_chr_events),
                        ParagraphStyle(
                            "SummaryText",
                            parent=self.styles.styles["Normal"],
                            fontSize=10,
                            fontName="Helvetica",
                            textColor=self.styles.COLORS["text"],
                            leading=14,
                            spaceAfter=12,
                        ),
                    )
                )

            # Add arm events to summary
            if summary_arm_events:
                logger.debug(f"Found {len(summary_arm_events)} arm events to report")
                self.summary_elements.append(
                    Paragraph(
                        f"Chromosome Arm Events{_cutoff_suffix(self.report)} "
                        "(requires visual inspection):<br/> "
                        + " <br/> ".join(summary_arm_events),
                        ParagraphStyle(
                            "SummaryText",
                            parent=self.styles.styles["Normal"],
                            fontSize=10,
                            fontName="Helvetica",
                            textColor=self.styles.COLORS["text"],
                            leading=14,
                            spaceAfter=12,
                        ),
                    )
                )

            # Generate genome-wide CNV plot
            logger.debug("Generating genome-wide CNV plot")
            from robin.gui.plotting_preferences import cnv_report_plot_caption

            significant_regions: dict[str, list[dict]] = {}
            for chrom in reportable_chromosomes:
                cytoband_analysis = cytoband_analysis_by_chrom.get(chrom)
                if cytoband_analysis is not None and not cytoband_analysis.empty:
                    region_list = build_significant_regions(cytoband_analysis)
                    if region_list:
                        significant_regions[chrom] = region_list

            use_normalized_summary = getattr(
                self.report, "cnv_summary_normalized", False
            )
            trial_genes = _clinical_trial_genes(self.report)
            # The genome-wide profile is a 4:1 figure with dozens of gene
            # labels; on the portrait text column it has to be scaled to about
            # half size, which is what made it look cluttered. Give it a
            # landscape page and render it at roughly the size it is placed at.
            landscape_width = getattr(
                self.report.doc, "landscape_width", inch * 10.4
            )
            # Aspect chosen so the figure plus its caption fill the landscape
            # frame: a taller panel also gives stacked gene labels more room.
            genome_fig_width_inch = landscape_width / inch
            genome_fig_height_inch = genome_fig_width_inch / GENOME_SUMMARY_ASPECT
            img_buf = create_CNV_plot(
                CNVresult,
                cnv_dict,
                normalized_cnv=log2_cnv,
                use_normalized_difference=use_normalized_summary,
                sex_estimate=str(XYestimate),
                panel_genes_df=panel_genes_df,
                target_coverage_df=target_coverage_df,
                significant_regions=significant_regions,
                reference_contig_scope=scope,
                configured_genes=configured_genes,
                clinical_trial_genes=trial_genes,
                fig_width=genome_fig_width_inch,
                fig_height=genome_fig_height_inch,
                fixed_axis_log2=getattr(self.report, "cnv_genome_axis_log2", None),
                cutoff_override=_report_cutoff_override(self.report),
                **_gene_label_size_kwargs(self.report),
            )
            summary_caption_scale = (
                "normalized_difference" if use_normalized_summary else "ploidy"
            )
            self.summary_elements.append(
                NextPageTemplate(self.report.LANDSCAPE_TEMPLATE)
            )
            self.summary_elements.append(PageBreak())
            # Place at the image's own aspect so it fills the width without
            # being stretched (the previous fixed 7.5x2 in placement distorted
            # a 4:1 figure by about 6%).
            self.summary_elements.append(
                Image(
                    img_buf,
                    width=landscape_width,
                    height=landscape_width * _image_aspect(img_buf, GENOME_SUMMARY_ASPECT),
                )
            )
            self.summary_elements.append(
                Paragraph(
                    cnv_report_plot_caption(
                        summary_caption_scale,
                        has_clinical_trial_genes=bool(trial_genes),
                        cutoff_override=_report_cutoff_override(self.report),
                        baseline_shift=baseline_shift,
                    ),
                    ParagraphStyle(
                        "PlotCaption",
                        parent=self.styles.styles["Caption"],
                        fontSize=9,
                        fontName="Helvetica",
                        textColor=self.styles.COLORS["text"],
                        alignment=1,  # Center alignment
                        spaceBefore=6,
                        spaceAfter=12,
                    ),
                )
            )
            # Everything after the genome-wide figure returns to portrait.
            self.summary_elements.append(
                NextPageTemplate(self.report.PORTRAIT_TEMPLATE)
            )
            self.summary_elements.append(PageBreak())

            # Create summary of CNV events using centralized detection
            logger.debug("Creating CNV summary using centralized events")

            # Use the events detected above
            whole_chr_events = []
            arm_events = []

            for event in events:
                if event.event_type.startswith("WHOLE_CHR_"):
                    event_type = event.event_type.replace("WHOLE_CHR_", "")
                    whole_chr_events.append([
                        event.chromosome.replace("chr", ""),
                        event_type,
                        f"{event.mean_cnv:.2f}",
                    ])
                else:
                    arm_label = f"{event.arm}-arm" if event.arm else "arm"
                    arm_events.append([
                        event.chromosome.replace("chr", ""),
                        arm_label,
                        event.event_type,
                        f"{event.mean_cnv:.2f}",
                        f"{event.proportion_affected:.1%}",
                    ])

            # CNV load, immediately before the event tables it contextualises.
            self._append_cnv_load_block()
            self._append_ngtd_target_table(
                log2_cnv, cnv_dict["bin_width"], str(XYestimate), panel_genes_df
            )

            # Add whole chromosome events summary if any exist
            if whole_chr_events:
                self.elements.append(
                    Paragraph(
                        "Whole Chromosome Events" + _cutoff_suffix(self.report),
                        ParagraphStyle(
                            "CNVTableTitle",
                            parent=self.styles.styles["Normal"],
                            fontSize=9,
                            fontName="Helvetica-Bold",
                            spaceAfter=4,
                        ),
                    )
                )
                self.elements.append(Spacer(1, 2))
                whole_chr_data = [["Chr", "State", "Mean CNV"]]
                whole_chr_data.extend(whole_chr_events)
                whole_chr_table = self.create_table(
                    whole_chr_data,
                    repeat_rows=1,
                    auto_col_width=False,
                    col_widths=self.col_widths_for_page([1.0, 2.0, 2.0]),
                    compact=True,
                )
                whole_chr_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                            ("ALIGN", (1, 1), (1, -1), "CENTER"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                self.elements.append(whole_chr_table)
                self.elements.append(Spacer(1, 4))

            # Build arm and regional event tables. Always stack vertically (never nested
            # side-by-side) so each table can split across pages. Nested tables
            # cannot split and cause LayoutError when content exceeds frame height.
            arm_col_weights = [1.0, 1.55, 1.5, 1.55, 2.2]
            regional_col_weights = [1.0, 3.0, 1.55, 1.55, 1.55, 1.55, 1.5, 3.9]

            arm_header = None
            arm_table = None
            regional_header = None
            regional_table = None

            if regional_cnv_events:
                regional_data = [[
                    "Chr",
                    "Region",
                    "Start (Mb)",
                    "End (Mb)",
                    "Length (Mb)",
                    "Mean CNV",
                    "State",
                    "Panel genes",
                ]]
                for event in regional_cnv_events:
                    panel_gene_text = format_panel_genes_for_table(event["panel_genes"])
                    regional_data.append([
                        event["chrom"],
                        event["region"],
                        f"{event['start_mb']:.2f}",
                        f"{event['end_mb']:.2f}",
                        f"{event['length_mb']:.2f}",
                        f"{event['mean_cnv']:.2f}",
                        event["state"],
                        panel_gene_text,
                    ])
                regional_table = self.create_table(
                    regional_data,
                    repeat_rows=1,
                    auto_col_width=False,
                    col_widths=self.col_widths_for_page(regional_col_weights),
                    compact=True,
                )
                regional_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (2, 1), (5, -1), "RIGHT"),
                            ("ALIGN", (6, 1), (6, -1), "CENTER"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                regional_title = "Regional CNV Events"
                if panel_name:
                    regional_title += f" ({panel_name} panel genes)"
                regional_title += _cutoff_suffix(self.report)
                regional_header = Paragraph(
                    regional_title,
                    ParagraphStyle(
                        "CNVTableTitle",
                        parent=self.styles.styles["Normal"],
                        fontSize=9,
                        fontName="Helvetica-Bold",
                        spaceAfter=4,
                    ),
                )

            if arm_events:
                arm_data = [["Chr", "Arm", "State", "Mean CNV", "Proportion Affected"]]
                arm_data.extend(arm_events)
                arm_table = self.create_table(
                    arm_data,
                    repeat_rows=1,
                    auto_col_width=False,
                    col_widths=self.col_widths_for_page(arm_col_weights),
                    compact=True,
                )
                arm_table.setStyle(
                    TableStyle(
                        [
                            ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                            ("ALIGN", (2, 1), (2, -1), "CENTER"),
                            ("ALIGN", (4, 1), (4, -1), "RIGHT"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                arm_header = Paragraph(
                    "Arm Events (visual inspection)" + _cutoff_suffix(self.report),
                    ParagraphStyle(
                        "CNVTableTitle",
                        parent=self.styles.styles["Normal"],
                        fontSize=9,
                        fontName="Helvetica-Bold",
                        spaceAfter=4,
                    ),
                )

            # Stack vertically as separate flowables so each table can split across pages
            if regional_header is not None:
                self.elements.append(regional_header)
                self.elements.append(regional_table)
            if arm_header is not None:
                if regional_header is not None:
                    self.elements.append(Spacer(1, 8))
                self.elements.append(arm_header)
                self.elements.append(arm_table)

            # Add note about detailed view
            self.elements.append(
                Paragraph(
                    (
                        "Note: Regional events are derived from merged cytoband analysis. "
                        "Panel genes are targets from the sample analysis panel overlapping "
                        "each called region. Arm-level events require visual inspection."
                    ),
                    ParagraphStyle(
                        "Note",
                        parent=self.styles.styles["Normal"],
                        fontSize=9,
                        textColor=self.styles.COLORS["text"],
                        spaceBefore=3,
                        spaceAfter=3,
                        italics=True,
                    ),
                )
            )

            # Add individual chromosome plots at full page width
            try:
                chromosome_status = {}

                for chrom in reportable_chromosomes:
                    cytoband_analysis = cytoband_analysis_by_chrom[chrom]
                    chromosome_status[chrom] = format_chromosome_cnv_status(
                        chrom,
                        events,
                        cytoband_analysis,
                    )

                if panel_name and not panel_genes_df.empty:
                    if configured_genes:
                        gene_selection_blurb = (
                            "Coverage markers show configured "
                            f"[cnv].genes targets ({len(configured_genes)}): "
                            + ", ".join(configured_genes)
                            + ". "
                        )
                    else:
                        gene_selection_blurb = (
                            "Coverage markers show panel targets >3 SD from the "
                            "chromosome mean. "
                        )
                    panel_plot_blurb = (
                        "Scatter points show bin-level log2(ploidy / expected copy number); "
                        "the dark trace is a rolling median. Gene markers show sequencing "
                        "coverage normalised onto the shared CNV axis "
                        "(log2(cov / mean cov) in log mode)."
                        if use_normalized_summary
                        else (
                            "Scatter points show bin-level copy number; the dark "
                            "trace is a rolling median. Gene markers show sequencing "
                            "coverage normalised onto the shared CNV axis "
                            "(mean_cnv × cov / mean cov)."
                        )
                    )
                    self.elements.append(
                        Paragraph(
                            (
                                f"Individual chromosome plots include coverage-scaled markers for "
                                f"genes in the <b>{panel_name}</b> target panel "
                                f"({len(panel_genes_df)} genes). "
                                f"{gene_selection_blurb}"
                                f"{panel_plot_blurb} "
                                "Coverage is normalised to the genome-wide mean "
                                "target coverage so gene markers match the summary plot."
                            ),
                            ParagraphStyle(
                                "PanelGeneLegend",
                                parent=self.styles.styles["Normal"],
                                fontSize=9,
                                textColor=self.styles.COLORS["text"],
                                spaceBefore=3,
                                spaceAfter=6,
                            ),
                        )
                    )

                # Generate all chromosome plots at once
                logger.debug(
                    "Generating individual chromosome plots for %d chromosomes",
                    len(reportable_chromosomes),
                )
                chromosome_plot_height_inch = self._chromosome_pdf_plot_height()
                chromosome_plot_width_inch = self._chromosome_pdf_plot_width()
                chromosome_plots = create_CNV_plot_per_chromosome(
                    CNVresult,
                    cnv_dict,
                    significant_regions=significant_regions,
                    chromosomes=reportable_chromosomes,
                    panel_genes_df=panel_genes_df,
                    chromosome_status=chromosome_status,
                    normalized_cnv=log2_cnv,
                    target_coverage_df=target_coverage_df,
                    use_log2_ratio=use_normalized_summary,
                    sex_estimate=str(XYestimate),
                    fig_height=chromosome_plot_height_inch,
                    fig_width=chromosome_plot_width_inch,
                    configured_genes=configured_genes,
                    clinical_trial_genes=trial_genes,
                    # The report shows each chromosome on its own full data
                    # range: on a fixed window a deep deletion is clamped to the
                    # axis edge, and a percentile fit clips it just the same. The
                    # report is read one chromosome at a time, so true depth
                    # matters more than comparability between them. The fixed
                    # window remains on the live view and the Chromosome PDFs.
                    # Zoomed to the fixed window in the report body: the
                    # full-range view flattens ordinary gains and losses to
                    # accommodate a handful of extreme bins. Both views are
                    # still available in the downloadable per-chromosome PDF,
                    # which adds a full-range page whenever bins fall outside
                    # this window.
                    fixed_axis_log2=CNV_CHROMOSOME_AXIS_LOG2,
                    full_range_axis=False,
                    # Four plots to a page, so the purple legend is stated once
                    # in the body text below instead of on every figure, where
                    # it collided with the chromosome titles.
                    figure_legend=False,
                    cutoff_override=_report_cutoff_override(self.report),
                    **_gene_label_size_kwargs(self.report),
                )
                plot_lookup = dict(chromosome_plots)
                plotted_chromosomes = [
                    chrom for chrom in reportable_chromosomes if chrom in plot_lookup
                ]
                chromosome_plot_height = inch * chromosome_plot_height_inch
                chromosome_plot_width = inch * chromosome_plot_width_inch

                for plot_idx, chrom in enumerate(plotted_chromosomes):
                    img_buf = plot_lookup[chrom]
                    self.elements.append(
                        Image(
                            img_buf,
                            width=chromosome_plot_width,
                            height=chromosome_plot_height,
                        )
                    )
                    last_on_page = (
                        (plot_idx + 1) % self.CHROMOSOME_PLOTS_PER_PAGE == 0
                    )
                    if (
                        plot_idx < len(plotted_chromosomes) - 1
                        and not last_on_page
                    ):
                        self.elements.append(Spacer(1, self.CHROMOSOME_PLOT_SPACER))

                # The chromosome figures carry the same cut-off lines as the
                # genome-wide panel, stated once under the block rather than on
                # each of the four figures per page.
                self.elements.append(Spacer(1, 4))
                self.elements.append(
                    Paragraph(
                        f"Dashed amber lines = gain / loss cut-off "
                        f"{_cutoff_text(self.report)}.",
                        ParagraphStyle(
                            "CNVChromCutoff",
                            parent=self.styles.styles["Normal"],
                            fontSize=7,
                            textColor=colors.HexColor("#475569"),
                            spaceAfter=2,
                        ),
                    )
                )

                # The purple convention, stated once under the block rather
                # than on each of the four figures per page.
                if trial_genes:
                    from robin.gui.plotting_preferences import (
                        CNV_STEP2_LEGEND,
                    )

                    self.elements.append(Spacer(1, 4))
                    self.elements.append(
                        Paragraph(
                            CNV_STEP2_LEGEND,
                            ParagraphStyle(
                                "CNVTrialLegend",
                                parent=self.styles.styles["Normal"],
                                fontSize=7,
                                textColor=colors.HexColor(
                                    CNV_REPORT_TRIAL_LEGEND_COLOR
                                ),
                                spaceAfter=4,
                            ),
                        )
                    )

                # Combined event list for CSV export only (tables above already
                # show whole-chromosome, regional, and arm events separately).
                all_cnv_events = []
                for event in regional_cnv_events:
                    all_cnv_events.append([
                        event["chrom"],
                        event["region"],
                        f"{event['start_mb']:.2f}",
                        f"{event['end_mb']:.2f}",
                        f"{event['length_mb']:.2f}",
                        f"{event['mean_cnv']:.2f}",
                        event["state"],
                        format_panel_genes_for_table(event["panel_genes"]),
                    ])
                for event in events:
                    region_name = (
                        f"{event.chromosome} {event.arm}-arm"
                        if event.arm
                        else f"{event.chromosome} whole chromosome"
                    )
                    panel_genes = panel_genes_in_region(
                        panel_genes_df,
                        event.chromosome,
                        event.start_pos,
                        event.end_pos,
                    )
                    all_cnv_events.append([
                        event.chromosome.replace("chr", ""),
                        region_name.replace(f"{event.chromosome} ", ""),
                        f"{event.start_pos/1e6:.2f}",
                        f"{event.end_pos/1e6:.2f}",
                        f"{event.length/1e6:.2f}",
                        f"{event.mean_cnv:.2f}",
                        event.event_type.replace("WHOLE_CHR_", ""),
                        format_panel_genes_for_table(panel_genes),
                    ])

                try:
                    # Whole chromosome events
                    if summary_whole_chr_events or whole_chr_events:
                        # whole_chr_events is already a list of lists
                        df_whole = pd.DataFrame(
                            whole_chr_events, columns=["Chr", "State", "Mean CNV"]
                        )
                        if not df_whole.empty:
                            df_whole["Mean CNV"] = pd.to_numeric(
                                df_whole["Mean CNV"], errors="coerce"
                            )
                        self.export_frames["cnv_whole_chromosome_events"] = df_whole

                    # Chromosome arm events
                    if arm_events:
                        df_arm = pd.DataFrame(
                            arm_events,
                            columns=[
                                "Chr",
                                "Arm",
                                "State",
                                "Mean CNV",
                                "Proportion Affected",
                            ],
                        )
                        if not df_arm.empty:
                            df_arm["Mean CNV"] = pd.to_numeric(
                                df_arm["Mean CNV"], errors="coerce"
                            )
                            # Convert percentage strings like "70%" to 0.7
                            df_arm["Proportion Affected"] = (
                                df_arm["Proportion Affected"]
                                .astype(str)
                                .str.rstrip("%")
                            )
                            df_arm["Proportion Affected"] = (
                                pd.to_numeric(
                                    df_arm["Proportion Affected"], errors="coerce"
                                )
                                / 100.0
                            )
                        self.export_frames["cnv_arm_events"] = df_arm

                    # Regional CNV events
                    if regional_cnv_events:
                        df_regional = pd.DataFrame(
                            [
                                {
                                    "Chr": event["chrom"],
                                    "Region": event["region"],
                                    "Start (Mb)": event["start_mb"],
                                    "End (Mb)": event["end_mb"],
                                    "Length (Mb)": event["length_mb"],
                                    "Mean CNV": event["mean_cnv"],
                                    "State": event["state"],
                                    "Panel genes": format_panel_genes_for_table(
                                        event["panel_genes"]
                                    ),
                                }
                                for event in regional_cnv_events
                            ]
                        )
                        self.export_frames["cnv_regional_events"] = df_regional

                    # Combined detailed export (same rows as the separate tables above)
                    if all_cnv_events:
                        df_detail = pd.DataFrame(
                            all_cnv_events,
                            columns=[
                                "Chr",
                                "Region",
                                "Start (Mb)",
                                "End (Mb)",
                                "Length (Mb)",
                                "Mean CNV",
                                "State",
                                "Panel genes",
                            ],
                        )
                        if not df_detail.empty:
                            for col in [
                                "Start (Mb)",
                                "End (Mb)",
                                "Length (Mb)",
                                "Mean CNV",
                            ]:
                                df_detail[col] = pd.to_numeric(
                                    df_detail[col], errors="coerce"
                                )
                        self.export_frames["cnv_detailed_events"] = df_detail
                except Exception as ex:
                    logger.error(
                        "Error building CNV export DataFrames: %s",
                        str(ex),
                        exc_info=True,
                    )

            except Exception as e:
                logger.error(
                    "Error processing detailed CNV analysis: %s", str(e), exc_info=True
                )
                self.elements.append(
                    Paragraph(
                        "Error processing detailed CNV analysis data",
                        self.styles.styles["Normal"],
                    )
                )

        except Exception as e:
            logger.error("Error processing CNV section: %s", str(e), exc_info=True)
            self.elements.append(
                Paragraph(
                    "Error processing CNV analysis data",
                    self.styles.styles["Normal"],
                )
            )
