"""
report.py

This module contains the main report class that coordinates the generation of the PDF report.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import inch
from .styling.styles import ReportStyles
from robin.gui import fonts
from robin.build_info import get_git_commit
from robin.utils.clinvar_manager import (
    format_clinvar_version_label,
    load_sample_clinvar_provenance,
)

logger = logging.getLogger(__name__)


class RobinReport:
    """Main class for generating ROBIN PDF reports."""

    def __init__(
        self,
        filename,
        output,
        center: str,
        progress_callback=None,
        workflow_steps=None,
        display_config=None,
        viewer_role=None,
        sample_identifiers=None,
        generated_by=None,
        generated_at=None,
        cnv_summary_normalized=None,
        plotting_preferences=None,
    ):
        """Initialize the report generator.

        Args:
            filename: Output PDF filename
            output: Directory containing analysis output files
            center: Center ID running the analysis
            progress_callback: Optional callback function for progress updates
            workflow_steps: Optional list of workflow steps to determine which sections to include
            display_config: Optional admin display configuration for section visibility
            viewer_role: Role key used to resolve display settings ('user' or 'admin')
            sample_identifiers: Optional dict with first_name, last_name, dob, nhs_number for inclusion in report
            generated_by: Optional username of the person who triggered report generation
            generated_at: Optional report generation timestamp (YYYY-MM-DD HH:MM:SS)
            cnv_summary_normalized: When True, genome-wide CNV summary uses
                log2(ploidy / expected copy number). When None (default), uses the admin
                Plotting preference.
            plotting_preferences: Optional pre-loaded plotting preferences (GUI).
                When None, loads persisted admin preferences from the security store.
        """
        self.filename = filename
        self.output = output
        self.center = center
        self.sample_id = os.path.basename(os.path.normpath(output))
        self.progress_callback = progress_callback
        self.workflow_steps = workflow_steps
        self.display_config = display_config
        from robin.gui.display_config import DEFAULT_VIEWER_ROLE

        self.viewer_role = viewer_role or DEFAULT_VIEWER_ROLE
        self.sample_identifiers = sample_identifiers
        self.generated_by = (str(generated_by).strip() if generated_by else None) or None
        self.generated_at = (str(generated_at).strip() if generated_at else None) or None
        from robin.gui.plotting_preferences import (
            load_plotting_preferences,
            resolve_cnv_chromosome_axis_log2,
            resolve_cnv_cutoff,
            resolve_cnv_genome_axis_log2,
            resolve_cnv_gene_label_points,
            resolve_cnv_gui_cutoff,
            resolve_cnv_gui_label_orientation,
            resolve_cnv_summary_normalized,
            resolve_plotting_reference_contig_scope,
        )

        # When omitted, load persisted admin defaults (not an empty in-memory config).
        self.plotting_preferences = (
            plotting_preferences
            if plotting_preferences is not None
            else load_plotting_preferences()
        )
        self.cnv_summary_normalized = resolve_cnv_summary_normalized(
            cnv_summary_normalized,
            plotting_preferences=self.plotting_preferences,
        )
        self.reference_contig_scope = resolve_plotting_reference_contig_scope(
            self.plotting_preferences
        )
        self.cnv_chromosome_axis_log2 = resolve_cnv_chromosome_axis_log2(
            self.plotting_preferences
        )
        self.cnv_genome_axis_log2 = resolve_cnv_genome_axis_log2(
            self.plotting_preferences
        )
        # Review cut-off from the admin panel, applied to the report's figures,
        # their caption and the CNV load. None means the calling thresholds.
        self.cnv_cutoff = resolve_cnv_cutoff(
            resolve_cnv_gui_cutoff(self.plotting_preferences)
        )
        self.cnv_gene_label_size = resolve_cnv_gene_label_points(
            self.plotting_preferences
        )
        self.cnv_gene_labels_rotated = (
            resolve_cnv_gui_label_orientation(self.plotting_preferences) == "rotated"
        )
        self.robin_commit = get_git_commit()
        self.clinvar_metadata = load_sample_clinvar_provenance(self.output)
        self.clinvar_version = format_clinvar_version_label(self.clinvar_metadata)

        # Handle filename with None prefix
        if filename.startswith("None"):
            final_folder = os.path.basename(os.path.normpath(output))
            self.filename = filename.replace("None", final_folder, 1)
            self.sample_id = final_folder

        # Initialize styling
        self.fonts_dir = os.path.join(os.path.dirname(os.path.abspath(fonts.__file__)))
        self.styles = ReportStyles(self.fonts_dir)

        # Initialize document elements
        self.elements_summary = []
        self.elements = []
        self.end_of_report_elements = []

        # Load master data
        self.masterdf = self._load_master_data()
        self.centreID = self._get_centre_id()

        # Create document
        self.doc = self._create_document()

        # Initialize sections
        self.sections = []
        self._initialize_sections()
    
    def _emit_progress(self, stage: str, message: str, progress: float = None):
        """Emit a progress update if callback is available."""
        if self.progress_callback:
            try:
                from robin.gui.report_progress import normalize_report_progress

                self.progress_callback({
                    'stage': stage,
                    'message': message,
                    'progress': normalize_report_progress(progress),
                })
            except Exception as e:
                logger.error(f"Error emitting progress: {e}")

    def _load_master_data(self):
        """Load the master data file if it exists."""
        master_path = os.path.join(self.output, "master.csv")
        if os.path.exists(master_path):
            return pd.read_csv(master_path)
        return None

    def _get_centre_id(self):
        """Get the centre ID from the provided center parameter."""
        return self.center

    #: Page template ids. Sections switch to the landscape one with
    #: ``NextPageTemplate`` for figures that need the full width of a page.
    PORTRAIT_TEMPLATE = "portrait"
    LANDSCAPE_TEMPLATE = "landscape"
    #: Narrower side margins on landscape pages, where the whole point is width.
    LANDSCAPE_SIDE_MARGIN = 0.6 * inch
    #: Paper for the landscape page. A4 keeps the whole report on one paper
    #: size — rotating it still gives the genome-wide CNV summary 10.5in of
    #: frame against the portrait column's 6.3in, which is the gain that
    #: matters. A3 reached 15.3in but made the report mixed-format, which is
    #: awkward to print.
    LANDSCAPE_PAGESIZE = A4

    def _create_document(self):
        """Create the PDF document with enhanced M3 margins and settings.

        Portrait A4 is the default. An **A3 landscape** template is registered
        alongside it so the genome-wide CNV summary can use a full page instead
        of being scaled down to the portrait text column.

        The page is A4 rotated, so the whole report stays one paper size while
        the figure still gets 10.5in of frame against the portrait column's
        6.3in. The standalone Genome PDF download is the route to a wider view.
        """
        doc = BaseDocTemplate(
            self.filename,
            pagesize=A4,
            rightMargin=1.0 * inch,  # Enhanced from 0.75 inch
            leftMargin=1.0 * inch,  # Enhanced from 0.75 inch
            topMargin=1.35 * inch,  # Keep existing for header compatibility
            bottomMargin=1.0 * inch,  # Enhanced from 0.75 inch
        )
        portrait_frame = Frame(
            doc.leftMargin,
            doc.bottomMargin,
            doc.width,
            doc.height,
            id=self.PORTRAIT_TEMPLATE,
        )
        landscape_size = landscape(self.LANDSCAPE_PAGESIZE)
        side = self.LANDSCAPE_SIDE_MARGIN
        landscape_frame = Frame(
            side,
            doc.bottomMargin,
            landscape_size[0] - 2 * side,
            landscape_size[1] - doc.topMargin - doc.bottomMargin,
            id=self.LANDSCAPE_TEMPLATE,
        )
        doc.addPageTemplates(
            [
                PageTemplate(
                    id=self.PORTRAIT_TEMPLATE, frames=[portrait_frame], pagesize=A4
                ),
                PageTemplate(
                    id=self.LANDSCAPE_TEMPLATE,
                    frames=[landscape_frame],
                    pagesize=landscape_size,
                ),
            ]
        )
        #: Usable width on a landscape page, for sections sizing figures.
        doc.landscape_width = landscape_size[0] - 2 * side
        return doc

    def _initialize_sections(self):
        """Initialize all report sections."""
        # Import sections here to avoid circular imports
        from .sections.classification import ClassificationSection
        from .sections.cnv import CNVSection
        from .sections.fusion import FusionSection
        from .sections.itd import ItdSection
        from .sections.coverage import CoverageSection
        from .sections.mgmt import MGMTSection
        from .sections.mnpflex import MNPFlexSection
        from .sections.run_data import RunDataSection
        from .sections.disclaimer import DisclaimerSection
        from .sections.variants import VariantsSection
        
        # Import section visibility helpers
        try:
            from robin.gui.config import any_classification_visible, is_section_visible
        except ImportError:
            def is_section_visible(section_id, **kwargs):  # type: ignore[misc]
                return True

            def any_classification_visible(**kwargs):  # type: ignore[misc]
                return True

        sections = []
        
        if any_classification_visible(
            self.workflow_steps,
            self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(ClassificationSection(self))
        
        if is_section_visible(
            "cnv",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(CNVSection(self))
        
        sections.append(VariantsSection(self))
        
        if is_section_visible(
            "fusion",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(FusionSection(self))

        if is_section_visible(
            "itd",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(ItdSection(self))
        
        if is_section_visible(
            "target",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(CoverageSection(self))
        
        if is_section_visible(
            "mgmt",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(MGMTSection(self))

        if is_section_visible(
            "mnpflex",
            workflow_steps=self.workflow_steps,
            display_config=self.display_config,
            surface="report",
            viewer_role=self.viewer_role,
        ):
            sections.append(MNPFlexSection(self))
        
        # Run data section (always included)
        sections.append(RunDataSection(self))
        
        # Disclaimer section (always included)
        sections.append(DisclaimerSection(self))

        self.sections = sections

    def generate_report(
        self,
        report_type="detailed",
        export_csv_dir=None,
        export_xlsx=False,
        export_zip=False,
    ):
        """Generate the complete PDF report.

        Args:
            report_type: Type of report to generate ('summary' or 'detailed')

        Returns:
            Path to the generated PDF file
        """
        try:
            logger.info("Starting report generation")
            self._emit_progress("initializing", "Initializing report generation...", 0.0)

            if not self.generated_at:
                self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Add summary section header (Heading 1 level)
            self.elements_summary.insert(
                0,
                Paragraph(
                    f"Summary - {self.sample_id}", self.styles.styles["Heading1"]
                ),
            )

            # Build summary card; add sample identifiers when available
            summary_lines = [
                "ROBIN Analysis Report<br/>",
                f"Sample ID: {self.sample_id}<br/>",
            ]
            si = getattr(self, "sample_identifiers", None)
            if si:
                if si.get("test_id"):
                    summary_lines.append(f"Test ID: {si['test_id']}<br/>")
                if si.get("first_name"):
                    summary_lines.append(f"First name: {si['first_name']}<br/>")
                if si.get("last_name"):
                    summary_lines.append(f"Last name: {si['last_name']}<br/>")
                if si.get("dob"):
                    summary_lines.append(f"Date of birth: {si['dob']}<br/>")
                if si.get("nhs_number"):
                    summary_lines.append(f"Hospital Number: {si['nhs_number']}<br/>")
                if si.get("notes"):
                    summary_lines.append(f"Notes: {si['notes']}<br/>")
            summary_lines.extend([
                f"Centre ID: {self.centreID if self.centreID else 'Not specified'}<br/>",
                f"Report Type: {report_type.title()}<br/>",
                f"Generated: {self.generated_at}<br/>",
            ])
            if self.generated_by:
                summary_lines.append(f"Generated by: {self.generated_by}")
            if self.robin_commit:
                summary_lines.append(f"ROBIN commit: {self.robin_commit}")
            if self.clinvar_metadata.get("file_date"):
                summary_lines.append(
                    f"ClinVar release: {self.clinvar_metadata['file_date']}"
                )
            summary_card_content = "\n            ".join(summary_lines)
            self.elements_summary.insert(
                1, Paragraph(summary_card_content, self.styles.styles["InfoCard"])
            )
            self.elements_summary.insert(2, Spacer(1, 8))

            # Process each section
            total_sections = len(self.sections)
            self._emit_progress("processing_sections", f"Processing {total_sections} report sections...", 0.2)
            
            for i, section in enumerate(self.sections):
                section_name = section.__class__.__name__.replace("Section", "")
                progress = 0.2 + (i / total_sections) * 0.5  # 20% to 70%
                
                try:
                    self._emit_progress("processing_sections", f"Loading {section_name} data...", progress)
                    section.add_content()
                    
                    summary_elements, main_elements = section.get_elements()

                    # Always include summary elements
                    self.elements_summary.extend(summary_elements)

                    # For detailed report or if it's the disclaimer section, include main elements
                    if (
                        report_type == "detailed"
                        or section.__class__.__name__ == "DisclaimerSection"
                    ):
                        self.elements.extend(main_elements)
                    
                    self._emit_progress("processing_sections", f"Completed {section_name} section", progress + 0.02)
                    
                except Exception as e:
                    logger.error(
                        f"Error processing section {section.__class__.__name__}: {e}",
                        exc_info=True,
                    )
                    # Add error card to report for better user feedback
                    error_content = f"""
                    <b>Section Processing Error</b><br/>
                    Section: {section.__class__.__name__}<br/>
                    Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}<br/>
                    <i>This section was skipped due to processing errors.</i>
                    """
                    self.elements_summary.append(
                        Paragraph(error_content, self.styles.styles["Error"])
                    )
                    self._emit_progress("processing_sections", f"Skipped {section_name} due to error", progress)

            # Add detailed analysis header and elements only for detailed reports
            if report_type == "detailed":
                self.elements.insert(0, PageBreak())
                self.elements.insert(
                    1,
                    Paragraph("Detailed Analysis", self.styles.styles["HeadlineLarge"]),
                )
                self.elements.insert(2, Spacer(1, 8))

            # Add page break before end of report elements
            if self.end_of_report_elements:
                self.end_of_report_elements.insert(0, PageBreak())

            # Combine all elements
            logger.info("Combining elements for final PDF")
            self._emit_progress("building_pdf", "Combining report sections...", 0.75)
            
            if report_type == "detailed":
                final_elements = (
                    self.elements_summary + self.elements + self.end_of_report_elements
                )
            else:
                final_elements = (
                    self.elements_summary + self.elements + self.end_of_report_elements
                )

            # Build the PDF
            self._emit_progress("building_pdf", "Rendering PDF document...", 0.82)
            from .header_footer import header_footer_canvas_factory

            self.doc.multiBuild(
                final_elements,
                canvasmaker=header_footer_canvas_factory(
                    self.sample_id,
                    self.centreID,
                    self.styles,
                    self.fonts_dir,
                    generated_by=self.generated_by,
                    generated_at=self.generated_at,
                    robin_commit=self.robin_commit,
                ),
            )

            logger.info(f"PDF created: {self.filename}")
            self._emit_progress("completed", "Report generation completed!", 1.0)

            try:
                metadata_path = os.path.join(
                    self.output, f"{self.sample_id}_report_metadata.json"
                )
                with open(metadata_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "sample_id": self.sample_id,
                            "centre_id": self.centreID,
                            "report_type": report_type,
                            "generated_at": self.generated_at,
                            "generated_by": self.generated_by,
                            "robin_commit": self.robin_commit or None,
                            "clinvar_release": self.clinvar_metadata.get("file_date") or None,
                            "clinvar_sha256": self.clinvar_metadata.get("sha256") or None,
                            "pdf_filename": os.path.basename(self.filename),
                        },
                        fh,
                        indent=2,
                    )
            except Exception as ex:
                logger.warning("Could not write report metadata JSON: %s", ex)

            # Add success message to report
            success_content = f"""
            <b>Report Generation Complete</b><br/>
            PDF file: {os.path.basename(self.filename)}<br/>
            Total sections processed: {len(self.sections)}<br/>
            Report type: {report_type.title()}
            """
            self.end_of_report_elements.append(
                Paragraph(success_content, self.styles.styles["Success"])
            )
            self.end_of_report_elements.append(Spacer(1, 6))

            # Optionally export CSV/XLSX/ZIP
            if export_csv_dir:
                try:
                    self._emit_progress("building_pdf", "Preparing CSV export...", 0.85)
                    os.makedirs(export_csv_dir, exist_ok=True)
                    manifest = {
                        "sample_id": self.sample_id,
                        "centre_id": self.centreID,
                        "report_type": report_type,
                        "generated_at": self.generated_at,
                        "generated_by": self.generated_by,
                        "robin_commit": self.robin_commit or None,
                        "clinvar_release": self.clinvar_metadata.get("file_date") or None,
                        "clinvar_sha256": self.clinvar_metadata.get("sha256") or None,
                        "files": [],
                    }

                    # Collect frames from sections
                    total_frames = 0
                    for section in self.sections:
                        frames = getattr(section, "get_export_frames", lambda: {})()
                        total_frames += len(frames)
                    
                    frame_count = 0
                    for section in self.sections:
                        frames = getattr(section, "get_export_frames", lambda: {})()
                        for name, df in frames.items():
                            frame_count += 1
                            progress = 0.85 + (frame_count / max(total_frames, 1)) * 0.08
                            self._emit_progress("building_pdf", f"Exporting {name} data...", progress)
                            
                            safe_name = name.replace(" ", "_")
                            csv_path = os.path.join(
                                export_csv_dir,
                                f"{self.sample_id}_{safe_name}.csv",
                            )
                            try:
                                df.to_csv(csv_path, index=False)
                                manifest["files"].append(
                                    {
                                        "name": name,
                                        "path": csv_path,
                                        "rows": int(df.shape[0]),
                                        "cols": int(df.shape[1]) if df.shape else 0,
                                    }
                                )
                            except Exception as ex:
                                logger.error(
                                    f"Error writing CSV for frame {name}: {str(ex)}",
                                    exc_info=True,
                                )

                    # Write manifest
                    self._emit_progress("building_pdf", "Writing manifest...", 0.93)
                    manifest_path = os.path.join(
                        export_csv_dir, f"{self.sample_id}_manifest.json"
                    )
                    with open(manifest_path, "w", encoding="utf-8") as fh:
                        json.dump(manifest, fh, indent=2)

                    # Optional XLSX workbook
                    if export_xlsx:
                        try:
                            xlsx_path = os.path.join(
                                export_csv_dir, f"{self.sample_id}_report_data.xlsx"
                            )
                            with pd.ExcelWriter(xlsx_path) as writer:
                                for f in manifest["files"]:
                                    # Reload to avoid potential dtype issues
                                    try:
                                        df = (
                                            pd.read_csv(f["path"])
                                            if f["path"].endswith(".csv")
                                            else None
                                        )
                                    except Exception:
                                        df = None
                                    if df is not None:
                                        sheet_name = (
                                            os.path.basename(f["path"])
                                            .replace(f"{self.sample_id}_", "")
                                            .replace(".csv", "")[:31]
                                        )
                                        df.to_excel(
                                            writer, index=False, sheet_name=sheet_name
                                        )
                            logger.info(f"XLSX written: {xlsx_path}")
                        except Exception as ex:
                            logger.error(
                                "Error writing XLSX: %s", str(ex), exc_info=True
                            )

                    # Optional ZIP archive
                    if export_zip:
                        try:
                            import zipfile

                            zip_path = os.path.join(
                                export_csv_dir, f"{self.sample_id}_report_data.zip"
                            )
                            with zipfile.ZipFile(
                                zip_path, "w", zipfile.ZIP_DEFLATED
                            ) as zf:
                                for f in manifest["files"]:
                                    if os.path.exists(f["path"]):
                                        zf.write(
                                            f["path"],
                                            arcname=os.path.basename(f["path"]),
                                        )
                                if os.path.exists(manifest_path):
                                    zf.write(
                                        manifest_path,
                                        arcname=os.path.basename(manifest_path),
                                    )
                            logger.info(f"ZIP written: {zip_path}")
                        except Exception as ex:
                            logger.error(
                                "Error writing ZIP: %s", str(ex), exc_info=True
                            )
                except Exception as ex:
                    logger.error(
                        "Error exporting CSV/XLSX/ZIP artifacts: %s",
                        str(ex),
                        exc_info=True,
                    )

            return self.filename
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            raise


def create_pdf(
    filename,
    output,
    center: str,
    report_type="detailed",
    export_csv_dir=None,
    export_xlsx=False,
    export_zip=False,
    progress_callback=None,
    workflow_steps=None,
    display_config=None,
    viewer_role=None,
    sample_identifiers=None,
    generated_by=None,
    generated_at=None,
    cnv_summary_normalized=None,
    plotting_preferences=None,
):
    """Create a PDF report from ROBIN analysis results.

    Args:
        filename: Output PDF filename
        output: Directory containing analysis output files
        center: Center ID running the analysis
        report_type: Type of report to generate ('summary' or 'detailed')
        export_csv_dir: Directory to export CSV files
        export_xlsx: Whether to export XLSX files
        export_zip: Whether to create ZIP archive
        progress_callback: Optional callback function for progress updates
        workflow_steps: Optional list of workflow steps to determine which sections to include
        display_config: Optional admin display configuration for section visibility
        viewer_role: Role key used to resolve display settings ('user' or 'admin')
        sample_identifiers: Optional dict with first_name, last_name, dob, nhs_number for report
        generated_by: Optional username of the person who triggered report generation
        generated_at: Optional report generation timestamp (YYYY-MM-DD HH:MM:SS)
        cnv_summary_normalized: When True, genome-wide CNV summary uses
            log2(ploidy / expected copy number). When None (default), uses the admin
            Plotting preference.
        plotting_preferences: Optional pre-loaded plotting preferences (GUI).
            When None, loads persisted admin preferences from the security store.

    Returns:
        Path to the generated PDF file
    """
    report = RobinReport(
        filename,
        output,
        center,
        progress_callback,
        workflow_steps=workflow_steps,
        display_config=display_config,
        viewer_role=viewer_role,
        sample_identifiers=sample_identifiers,
        generated_by=generated_by,
        generated_at=generated_at,
        cnv_summary_normalized=cnv_summary_normalized,
        plotting_preferences=plotting_preferences,
    )
    return report.generate_report(
        report_type=report_type,
        export_csv_dir=export_csv_dir,
        export_xlsx=export_xlsx,
        export_zip=export_zip,
    )
