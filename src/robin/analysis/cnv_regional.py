"""Shared regional CNV analysis helpers for reports and GUI."""

from __future__ import annotations

import functools
import logging
import os
import re

from typing import Sequence

import numpy as np
import pandas as pd

from robin import resources
from robin.analysis.cnv_classification import CNVEvent
from robin.utils.sequencing_files import panel_bed_filename

logger = logging.getLogger(__name__)

from robin.reference_contigs import CANONICAL_CONTIG_RE, is_canonical_contig

REPORTABLE_CHROMOSOME_RE = CANONICAL_CONTIG_RE
SIGNIFICANT_CNV_STATES = {"GAIN", "LOSS", "HIGH_GAIN", "DEEP_LOSS"}

#: Cytoband stains with no uniquely mappable sequence: centromeres (``acen``),
#: heterochromatin (``gvar``) and acrocentric stalks (``stalk``). Read depth
#: there reflects mappability, not copy number — on the log2 track these bands
#: sit at −3 to −5 — so they are reported as not assessed rather than as losses.
UNMAPPABLE_CYTOBAND_STAINS = frozenset({"acen", "gvar", "stalk"})


def is_reportable_chromosome(chromosome: str) -> bool:
    """Return True for standard autosomes/sex chromosomes used in CNV reporting."""
    return is_canonical_contig(chromosome)


def _resource_path(filename: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(resources.__file__)), filename)


def load_cytobands_bed() -> pd.DataFrame:
    """Packaged GRCh38 cytoband table, empty when the resource is missing."""
    columns = ["chrom", "start_pos", "end_pos", "name", "stain"]
    path = _resource_path("cytoBand.txt")
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path, sep="\t", names=columns)
    except Exception:
        logger.debug("Could not load cytoband resource", exc_info=True)
        return pd.DataFrame(columns=columns)


def load_centromere_bed() -> pd.DataFrame:
    """Packaged centromere/satellite regions, empty when the resource is missing."""
    columns = ["chrom", "start_pos", "end_pos", "name"]
    path = _resource_path("cenSatRegions.bed")
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(
            path,
            usecols=[0, 1, 2, 3],
            names=columns,
            header=None,
            sep=r"\s+",
        )
    except Exception:
        logger.debug("Could not load centromere resource", exc_info=True)
        return pd.DataFrame(columns=columns)


#: Control profile ROBIN normalises against. Bins with no coverage here are the
#: divisor-is-zero bins whose log2 ratio explodes into the "rain" below the
#: profile, so they are measured evidence of unmappability rather than an
#: annotation guess. ``cenSatRegions.bed`` is far too wide for this (it spans
#: PDGFRA, KIT, KDR, NF1 and SUZ12) and the cytoband stains are offset from
#: where mappability actually fails.
CONTROL_MAPPABILITY_RESOURCE = "HG01280_control_new.pkl"

#: A plot bin is hidden when at least this fraction of its control bins are empty.
UNMAPPABLE_CONTROL_FRACTION = 0.5


@functools.lru_cache(maxsize=1)
def _chromosome_lengths_from_cytobands() -> dict[str, int]:
    bands = load_cytobands_bed()
    if bands.empty:
        return {}
    try:
        return bands.groupby("chrom")["end_pos"].max().astype(int).to_dict()
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def load_control_mappability() -> dict[str, tuple[np.ndarray, int]]:
    """Per contig: cumulative count of empty control bins, plus the bin size.

    The cumulative form lets a plot bin of any width be scored with two lookups
    instead of a slice. Returns an empty mapping when the control is missing, so
    a missing resource shows everything rather than hiding it blindly.
    """
    path = _resource_path(CONTROL_MAPPABILITY_RESOURCE)
    if not os.path.exists(path):
        return {}
    try:
        import pickle

        with open(path, "rb") as handle:
            profile = pickle.load(handle)
    except Exception:
        logger.debug("Could not load control mappability profile", exc_info=True)
        return {}

    lengths = _chromosome_lengths_from_cytobands()
    out: dict[str, tuple[np.ndarray, int]] = {}
    for contig, counts in (profile or {}).items():
        try:
            arr = np.asarray(counts)
            if arr.ndim != 1 or arr.size == 0:
                continue
            chrom_length = int(lengths.get(str(contig), 0))
            if chrom_length <= 0:
                continue
            bin_size = int(round(chrom_length / arr.size))
            if bin_size <= 0:
                continue
            empty = np.concatenate(
                ([0], np.cumsum((arr == 0).astype(np.int64)))
            )
            out[str(contig)] = (empty, bin_size)
        except Exception:
            continue
    return out


@functools.lru_cache(maxsize=1)
def load_protected_target_intervals() -> dict[str, tuple[tuple[int, int], ...]]:
    """Merged intervals of every packaged panel target.

    A bin overlapping one of these is never hidden, whatever the control says.
    This is what makes target safety structural: it does not depend on the
    threshold being lenient enough. TBL1X, for instance, has zero control
    coverage across its whole length and no threshold would spare it.
    """
    import glob

    directory = os.path.dirname(os.path.abspath(resources.__file__))
    by_chrom: dict[str, list[list[int]]] = {}
    for bed in sorted(glob.glob(os.path.join(directory, "*_panel_name_uniq.bed"))):
        try:
            with open(bed) as handle:
                for line in handle:
                    if line.startswith(("#", "track", "browser")):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        start, end = int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if end > start:
                        by_chrom.setdefault(parts[0], []).append([start, end])
        except Exception:
            logger.debug("Could not read panel bed %s", bed, exc_info=True)

    merged: dict[str, tuple[tuple[int, int], ...]] = {}
    for chrom, spans in by_chrom.items():
        spans.sort()
        acc: list[list[int]] = []
        for start, end in spans:
            if acc and start <= acc[-1][1]:
                acc[-1][1] = max(acc[-1][1], end)
            else:
                acc.append([start, end])
        merged[chrom] = tuple((s, e) for s, e in acc)
    return merged


def unmappable_bin_mask(
    contig: str,
    positions_bp,
    bin_width_bp: int,
    *,
    threshold: float = UNMAPPABLE_CONTROL_FRACTION,
    protect_targets: bool = True,
) -> np.ndarray:
    """True where a plot bin should be hidden as unmappable.

    ``positions_bp`` are the plotted x of each bin and ``bin_width_bp`` its
    span. ``downsample_cnv_for_plot`` returns bin centres, so the window scored
    here is centred on the position rather than starting at it. A bin is hidden
    when at least ``threshold`` of the control bins it covers are empty, unless
    it overlaps a panel target.

    Display only: the values stay in the track, in segmentation and in the
    reported calls. Fails open - no control profile means nothing is hidden.
    """
    pos = np.asarray(positions_bp, dtype=float)
    hide = np.zeros(pos.shape, dtype=bool)
    entry = load_control_mappability().get(str(contig))
    if entry is None or pos.size == 0 or bin_width_bp <= 0:
        return hide

    empty_cumsum, control_bin = entry
    n_control = empty_cumsum.size - 1
    half = bin_width_bp / 2.0
    lo = np.clip(((pos - half) / control_bin).astype(np.int64), 0, n_control)
    hi = np.clip(((pos + half) / control_bin).astype(np.int64), 0, n_control)
    span = hi - lo
    covered = span > 0
    if not covered.any():
        return hide
    empty = empty_cumsum[hi[covered]] - empty_cumsum[lo[covered]]
    hide[covered] = (empty / span[covered]) >= threshold

    if protect_targets and hide.any():
        for start, end in load_protected_target_intervals().get(str(contig), ()):
            hide &= ~((pos - half < end) & (start < pos + half))
    return hide


@functools.lru_cache(maxsize=1)
def load_centromere_boundaries() -> dict[str, int]:
    """Genomic position separating the p and q arms of each chromosome.

    Taken as the start of the first q band in the packaged cytoband table, which
    is the canonical p/q boundary. Falls back to the midpoint of the centromere
    region when the cytoband resource is unavailable.
    """
    boundaries: dict[str, int] = {}
    cytobands = load_cytobands_bed()
    if not cytobands.empty:
        for chrom, bands in cytobands.groupby("chrom"):
            q_bands = bands[bands["name"].astype(str).str.startswith("q")]
            if not q_bands.empty:
                boundaries[str(chrom)] = int(q_bands["start_pos"].min())
                continue
            p_bands = bands[bands["name"].astype(str).str.startswith("p")]
            if not p_bands.empty:
                boundaries[str(chrom)] = int(p_bands["end_pos"].max())
    if boundaries:
        return boundaries

    centromeres = load_centromere_bed()
    for chrom, regions in centromeres.groupby("chrom"):
        boundaries[str(chrom)] = int(
            (regions["start_pos"].min() + regions["end_pos"].max()) / 2
        )
    return boundaries


def compute_cnv_load(
    log2_track: dict[str, "np.ndarray"],
    bin_width: int,
    sex_estimate: str = "Unknown",
    cutoff_override: float | None = None,
) -> dict[str, float]:
    """Fraction of the assessed genome carrying a called gain or loss.

    Counts bins past the cut-off currently in force, so the load always agrees
    with the lines drawn on the figures. With no override that is the configured
    **calling threshold** from ``classification_config``; ``cutoff_override``
    applies a symmetric log2 cut-off instead, matching the review Cut-off menu.

    The cut-off used is returned as ``cutoff`` and ``cutoff_label`` so that every
    place the load is shown can say which threshold produced it — a load figure
    is not comparable between samples reviewed at different cut-offs.

    Bins with no data are excluded from both the numerator and the denominator,
    so an incompletely covered genome reports load over what was actually
    assessed rather than diluting it toward zero.

    Returns Mb and percentage for gain, loss and the two combined, plus the
    assessed span the percentages are taken over.
    """
    from robin.classification_config import resolve_cnv_thresholds

    override = None
    if cutoff_override is not None:
        try:
            override = abs(float(cutoff_override))
        except (TypeError, ValueError):
            override = None
        if override is not None and not np.isfinite(override):
            override = None

    bin_mb = float(bin_width) / 1_000_000.0
    assessed = gained = lost = 0
    for chrom, values in (log2_track or {}).items():
        if not is_reportable_chromosome(chrom):
            continue
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        gain_threshold, loss_threshold = resolve_cnv_thresholds(
            chrom, sex_estimate, override
        )
        assessed += int(finite.size)
        gained += int(np.count_nonzero(finite >= gain_threshold))
        lost += int(np.count_nonzero(finite <= loss_threshold))

    assessed_mb = assessed * bin_mb

    def _percent(count: int) -> float:
        return (count / assessed * 100.0) if assessed else 0.0

    # One shared formatter, so a heading in the GUI, the report and the CSV all
    # name the cut-off the same way.
    try:
        from robin.gui.plotting_preferences import cnv_cutoff_label

        cutoff_label = cnv_cutoff_label(override)
    except Exception:  # pragma: no cover - GUI package unavailable
        cutoff_label = f"\u00b1{override:g}" if override else "calling"

    return {
        "cutoff": override,
        "cutoff_label": cutoff_label,
        "assessed_mb": assessed_mb,
        "gain_mb": gained * bin_mb,
        "loss_mb": lost * bin_mb,
        "total_mb": (gained + lost) * bin_mb,
        "gain_percent": _percent(gained),
        "loss_percent": _percent(lost),
        "total_percent": _percent(gained + lost),
    }


#: Label used when a listed target carries no called event, so the table always
#: says "looked at, nothing found" rather than leaving a blank cell.
CNV_NO_EVENT_LABEL = "No CNVs Detected"
#: Label for a listed gene that is not on the sample's target panel. Off-panel
#: sequence is background coverage only, so the gene was NOT assessed — which is
#: a different statement from having looked and found nothing, and must never be
#: reported as "No CNVs Detected".
CNV_GENE_NOT_LOCATED_LABEL = "Not on panel"
#: Label for a target that IS on the panel but has no CNV data covering it yet —
#: early in a run, or a dropout. Again distinct from "looked and found nothing".
CNV_GENE_NO_DATA_LABEL = "No data yet"
#: Label for a gene ROBIN cannot place at all — absent from the panel and from
#: the packaged gene references. Distinct from being off-panel: this one could
#: not even be looked for, so it says nothing about the locus.
CNV_GENE_UNKNOWN_LABEL = "Not in reference"


def compute_target_gene_cnv_states(
    log2_track: dict[str, "np.ndarray"],
    bin_width: int,
    genes: "Sequence[str]",
    sex_estimate: str = "Unknown",
    gene_frames: "Sequence[pd.DataFrame] | None" = None,
    cutoff_override: float | None = None,
) -> list[dict]:
    """Gain/loss state for each named target, in the order given.

    Every requested gene comes back, including ones with no event, so the table
    can state that a target was assessed and nothing was found. A gene that is
    not on the sample's panel is reported as such rather than as "no CNVs":
    off-panel sequence is background coverage, so the gene was not assessed, and
    the two are clinically different statements.

    The value taken for a gene is the mean across its bins, with a focal event
    reported at its own depth — the same rule the plots use for gene markers —
    judged against the cut-off in force, so the table agrees with the figures.
    """
    from robin.classification_config import resolve_cnv_thresholds
    from robin.cnv_plot_style import gene_bin_window, robust_gene_value

    frames = [f for f in (gene_frames or [load_gene_bed()]) if f is not None and not f.empty]
    override = None
    if cutoff_override is not None:
        try:
            candidate = abs(float(cutoff_override))
            override = candidate if np.isfinite(candidate) else None
        except (TypeError, ValueError):
            override = None

    def _bed_gene_names(value) -> set[str]:
        """Every gene name a BED row stands for, upper-cased.

        A row may cover several overlapping genes and names them together, joined
        with commas or slashes — ``CDKN2A,CDKN2B,CDKN2B-AS1``. Comparing a
        requested gene against the joined string never matches, which reported
        CDKN2A, CDK4 and MTAP as "not in reference" on samples that plainly had
        them: 20 of the 45 NGTD loci on one archived sample, including a CDKN2A
        homozygous deletion that was independently confirmed.
        """
        text = str(value).upper()
        for separator in (",", "/", ";"):
            text = text.replace(separator, "|")
        return {part.strip() for part in text.split("|") if part.strip()}

    def _locate(gene: str):
        """Return (hit, on_panel). The first frame is the sample's panel.

        A label may name several genes at one locus, separated by ``/`` — for
        example ``CDKN2B/CDKN2B-AS1``. Overlapping transcripts share the same CNV
        bins, so they cannot be told apart at this resolution and are reported as
        one row; the alternatives are tried in order until one is found.
        """
        for alternative in [part.strip() for part in str(gene).split("/") if part.strip()]:
            wanted = alternative.upper()
            for index, frame in enumerate(frames):
                names = frame["gene"].map(_bed_gene_names)
                hit = frame[names.map(lambda s: wanted in s)]
                if not hit.empty:
                    return hit, index == 0
        return None, False

    results: list[dict] = []
    for gene in genes:
        name = str(gene).strip()
        if not name:
            continue
        hit, on_panel = _locate(name)
        row = {
            "gene": name,
            "chrom": None,
            "value": None,
            "state": CNV_GENE_UNKNOWN_LABEL,
            "located": False,
            "on_panel": False,
        }
        if hit is None:
            results.append(row)
            continue
        if not on_panel:
            # Placed on the genome, but not a target on this panel: the only
            # coverage there is background, so it was not assessed.
            row["chrom"] = str(hit.iloc[0]["chrom"])
            row["state"] = CNV_GENE_NOT_LOCATED_LABEL
            results.append(row)
            continue

        # Bins are pooled across every interval the gene has, then judged once.
        # Taking the most extreme interval instead let an annotation fragment
        # decide: unique_genes.bed carries EGFR twice, as the real 63 kb gene and
        # as a 205 bp stub, and the stub resolves to a single bin with no noise
        # protection. In testing that stub read as a clear loss while the gene
        # as a whole was unchanged.
        best_value = None
        best_chrom = None
        for chrom, chrom_rows in hit.groupby(hit["chrom"].astype(str)):
            values = log2_track.get(chrom)
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            spans: list[tuple[int, int]] = []
            for _, interval in chrom_rows.iterrows():
                start, stop = gene_bin_window(
                    int(interval["start_pos"]) // bin_width,
                    int(interval["end_pos"]) // bin_width + 1,
                    len(array),
                )
                if stop > start:
                    spans.append((start, stop))
            if not spans:
                continue
            # Merge overlapping or touching spans so a bin is counted once.
            spans.sort()
            merged = [spans[0]]
            for start, stop in spans[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
                else:
                    merged.append((start, stop))
            # NaN between blocks: separate targets are not adjacent bins, so a
            # focal run must not be allowed to bridge them.
            pieces: list[np.ndarray] = []
            for start, stop in merged:
                if pieces:
                    pieces.append(np.array([np.nan]))
                pieces.append(array[start:stop])
            window = np.concatenate(pieces)
            if not np.isfinite(window).any():
                continue
            cutoff = abs(
                resolve_cnv_thresholds(chrom, sex_estimate, override)[0]
            )
            # Averaged across the gene, with a focal event kept at its own depth
            # when consecutive bins support it. The peak bin alone is a max-of-N
            # statistic and called a neutral five-bin gene about half the time.
            candidate = robust_gene_value(window, cutoff=cutoff)
            if best_value is None or abs(candidate) > abs(best_value):
                best_value, best_chrom = candidate, chrom

        if best_value is None:
            # On the panel, but no CNV bins cover it yet. Distinct from being
            # off-panel: this one is expected to fill in as the run accumulates.
            row["state"] = CNV_GENE_NO_DATA_LABEL
            row["located"] = True
            row["on_panel"] = True
            row["chrom"] = str(hit.iloc[0]["chrom"])
            results.append(row)
            continue

        gain_threshold, loss_threshold = resolve_cnv_thresholds(
            best_chrom, sex_estimate, override
        )
        if best_value >= gain_threshold:
            state = "GAIN"
        elif best_value <= loss_threshold:
            state = "LOSS"
        else:
            state = CNV_NO_EVENT_LABEL
        results.append(
            {
                "gene": name,
                "chrom": best_chrom,
                "value": best_value,
                "state": state,
                "located": True,
                "on_panel": True,
            }
        )
    return results


#: Heading used for the CNV load block wherever it is shown.
CNV_LOAD_TITLE = "CNV load"


def cnv_load_summary_text(load: dict[str, float] | None) -> str:
    """One-line CNV load summary, shared by the GUI and every PDF."""
    if not load or not load.get("assessed_mb"):
        return ""
    label = load.get("cutoff_label")
    heading = f"{CNV_LOAD_TITLE} (cut-off {label})" if label else CNV_LOAD_TITLE
    return (
        f"{heading}: {load['total_percent']:.1f}% "
        f"({load['total_mb']:,.0f} Mb)  \u00b7  "
        f"gain {load['gain_percent']:.1f}% ({load['gain_mb']:,.0f} Mb)  \u00b7  "
        f"loss {load['loss_percent']:.1f}% ({load['loss_mb']:,.0f} Mb)  \u00b7  "
        f"of {load['assessed_mb']:,.0f} Mb assessed"
    )


def load_gene_bed() -> pd.DataFrame:
    """Packaged unique gene BED used for annotating CNV events."""
    columns = ["chrom", "start_pos", "end_pos", "gene"]
    path = _resource_path("unique_genes.bed")
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path, sep="\t", names=columns)
    except Exception:
        logger.debug("Could not load gene bed resource", exc_info=True)
        return pd.DataFrame(columns=columns)


def load_panel_gene_bed(output_dir: str) -> tuple[str | None, pd.DataFrame]:
    """Load the target panel gene BED for the sample's analysis panel."""
    panel = None
    master_csv = os.path.join(output_dir, "master.csv")
    if os.path.exists(master_csv):
        try:
            master_df = pd.read_csv(master_csv)
            if not master_df.empty and "analysis_panel" in master_df.columns:
                panel_val = master_df.iloc[0]["analysis_panel"]
                if panel_val is not None and str(panel_val).strip().lower() not in ("", "nan"):
                    panel = str(panel_val).strip()
        except Exception as exc:
            logger.debug("Could not read analysis panel from master.csv: %s", exc)

    empty = pd.DataFrame(columns=["chrom", "start_pos", "end_pos", "gene"])
    if not panel:
        return None, empty

    bed_path = os.path.join(
        os.path.dirname(os.path.abspath(resources.__file__)),
        panel_bed_filename(panel),
    )
    if not os.path.exists(bed_path):
        logger.warning("Target panel BED not found for panel '%s' at %s", panel, bed_path)
        return panel, empty

    return panel, pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start_pos", "end_pos", "gene"],
    )


def load_target_coverage_df(output_dir: str) -> pd.DataFrame:
    """Load per-target coverage for panel gene coverage overlays."""
    path = os.path.join(output_dir, "target_coverage.csv")
    empty = pd.DataFrame(
        columns=["chrom", "startpos", "endpos", "name", "length", "coverage", "bases"]
    )
    if not os.path.exists(path):
        return empty
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.debug("Could not read target coverage from %s: %s", path, exc)
        return empty


def format_chromosome_cnv_status(
    chromosome: str,
    events: list[CNVEvent],
    cytoband_analysis: pd.DataFrame,
) -> str:
    """Summarize detected CNV changes for an individual chromosome plot caption."""
    chrom_events = [event for event in events if event.chromosome == chromosome]
    if chrom_events:
        parts = []
        for event in chrom_events:
            if event.event_type.startswith("WHOLE_CHR_"):
                parts.append(
                    f"Whole chromosome {event.event_type.replace('WHOLE_CHR_', '')}"
                )
            elif event.arm:
                parts.append(f"{event.arm}-arm {event.event_type}")
            else:
                parts.append(event.event_type)
        return "; ".join(parts)

    if not cytoband_analysis.empty:
        regional = [
            f"{row['cnv_state']}: {row['name']}"
            for _, row in cytoband_analysis.iterrows()
            if row["cnv_state"] in SIGNIFICANT_CNV_STATES
        ]
        if regional:
            if len(regional) > 3:
                return "; ".join(regional[:3]) + f"; +{len(regional) - 3} more"
            return "; ".join(regional)

    return "No significant CNV change"


def build_significant_regions(cytoband_analysis: pd.DataFrame) -> list[dict]:
    """Convert cytoband analysis rows into plot highlight regions."""
    regions = []
    if cytoband_analysis.empty:
        return regions

    for _, row in cytoband_analysis.iterrows():
        if row["cnv_state"] not in SIGNIFICANT_CNV_STATES:
            continue
        regions.append(
            {
                "start_pos": int(row["start_pos"]),
                "end_pos": int(row["end_pos"]),
                "type": row["cnv_state"],
                "name": row.get("name", ""),
            }
        )
    return regions


def panel_genes_in_region(
    panel_genes_df: pd.DataFrame,
    chrom: str,
    start_pos: int,
    end_pos: int,
) -> list[str]:
    """Return sorted unique panel gene labels overlapping a genomic interval."""
    if panel_genes_df.empty:
        return []

    hits = panel_genes_df[
        (panel_genes_df["chrom"] == chrom)
        & (panel_genes_df["start_pos"] <= end_pos)
        & (panel_genes_df["end_pos"] >= start_pos)
    ]
    labels: list[str] = []
    seen: set[str] = set()
    for gene_value in hits["gene"].astype(str):
        for part in str(gene_value).split(","):
            label = part.strip()
            if label and label.lower() != "nan" and label not in seen:
                seen.add(label)
                labels.append(label)
    return sorted(labels)


def format_panel_genes_for_table(panel_genes: list[str]) -> str:
    """Format panel gene labels for report tables."""
    return ", ".join(panel_genes) if panel_genes else "—"


def build_regional_cnv_events(
    cytoband_analysis: pd.DataFrame,
    panel_genes_df: pd.DataFrame,
) -> list[dict]:
    """Build focal cytoband CNV events with overlapping panel target genes."""
    regional_events: list[dict] = []
    if cytoband_analysis.empty:
        return regional_events

    for _, row in cytoband_analysis.iterrows():
        if row["cnv_state"] not in SIGNIFICANT_CNV_STATES:
            continue

        chrom = str(row["chrom"])
        start_pos = int(row["start_pos"])
        end_pos = int(row["end_pos"])
        regional_events.append(
            {
                "chrom": chrom.replace("chr", ""),
                "chromosome": chrom,
                "region": str(row.get("name", "")),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "start_mb": start_pos / 1_000_000,
                "end_mb": end_pos / 1_000_000,
                "length_mb": (end_pos - start_pos) / 1_000_000,
                "mean_cnv": float(row["mean_cnv"]),
                "state": str(row["cnv_state"]),
                "panel_genes": panel_genes_in_region(
                    panel_genes_df, chrom, start_pos, end_pos,
                ),
            }
        )

    return regional_events


def format_regional_event_table_row(event: dict) -> dict:
    """Format a regional CNV event dict for NiceGUI table rows."""
    return {
        "chrom": event["chrom"],
        "region": event["region"],
        "start_mb": f"{event['start_mb']:.2f}",
        "end_mb": f"{event['end_mb']:.2f}",
        "length_mb": f"{event['length_mb']:.2f}",
        "mean_cnv": f"{event['mean_cnv']:.3f}",
        "state": event["state"],
        "panel_genes": format_panel_genes_for_table(event["panel_genes"]),
    }


def _nan_safe(values, reducer, default: float = 0.0) -> float:
    """Reduce ignoring NaNs, returning ``default`` when nothing is finite.

    The log2 track carries NaN wherever a bin had no coverage, which the
    sample-minus-reference track this analysis used to run on never did. A plain
    mean over those bins returns NaN, which then silently disables every
    threshold comparison downstream (NaN compares False against everything).
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return default
    return float(reducer(array))


def analyze_cytoband_cnv(
    cnv_data: dict,
    chromosome: str,
    cnv_dict: dict,
    cytobands_bed: pd.DataFrame,
    centromere_bed: pd.DataFrame,
    gene_bed: pd.DataFrame,
    sex_estimate: str,
    cutoff_override: float | None = None,
) -> pd.DataFrame:
    """
    Analyze CNV values within each cytoband to detect duplications and deletions.

    Expects ``cnv_data`` on the **log2(ploidy / expected copy number)** scale, so
    that the calling cut-off means the same thing here as everywhere else. Build
    it with ``compute_cnv_log2_from_ploidy()``.

    This previously ran on the sample-minus-reference difference track, which is
    wrong twice over: it is in copies rather than log2, and the reference pass
    re-processes the same BAM on top of the reference baseline, so it inherits
    the sample's own aberration and cancels most of the signal. On a sample with
    a clear whole-chromosome gain (chr7 at ploidy 3.14) the reference sat at
    2.89, leaving a difference of 0.26 — under the cut-off — and the regional
    table reported nothing at all.

    Thresholds adapt to the variation in the data, and are then gated on the
    configured calling cut-off so that a region is only reported when it crosses
    both. Without the gate the adaptive rule reports regions on a completely flat
    chromosome, because it measures against that chromosome's own mean.
    """
    logger.debug(f"\n{'='*50}")
    logger.debug(f"Starting CNV analysis for {chromosome}")
    logger.debug(f"CNV data keys: {list(cnv_data.keys())}")

    if "bin_width" not in cnv_dict:
        logger.debug("No cnv_dict or bin_width available")
        return pd.DataFrame()

    logger.debug(f"Bin width: {cnv_dict['bin_width']}")

    if cnv_dict["bin_width"] > 10_000_000:
        logger.debug("Resolution insufficient for CNV calling")
        return pd.DataFrame()

    bin_width = cnv_dict["bin_width"]
    chromosome_cytobands = cytobands_bed[cytobands_bed["chrom"] == chromosome].copy()
    logger.debug(f"Number of cytobands for {chromosome}: {len(chromosome_cytobands)}")

    max_expected_cytobands = len(chromosome_cytobands)
    merged_cytobands = [None] * max_expected_cytobands
    merged_idx = 0
    whole_chr_event = False
    whole_chr_state = "NORMAL"

    if chromosome in cnv_data:
        logger.debug(
            f"\nAnalyzing chromosome {chromosome} for whole chromosome events:"
        )

        mask = np.ones(len(cnv_data[chromosome]), dtype=bool)
        centromere = centromere_bed[centromere_bed["chrom"] == chromosome]
        if not centromere.empty:
            cent_start_bin = int(centromere["start_pos"].iloc[0] / bin_width)
            cent_end_bin = int(centromere["end_pos"].iloc[0] / bin_width)
            mask[cent_start_bin:cent_end_bin] = False
            logger.debug(f"Excluded centromere region: {cent_start_bin}-{cent_end_bin}")
        # Unmappable bands would drag the chromosome mean down and inflate its
        # SD, which loosens the adaptive thresholds derived from both.
        unmappable = chromosome_cytobands[
            chromosome_cytobands["stain"].astype(str).isin(UNMAPPABLE_CYTOBAND_STAINS)
        ]
        for _, band in unmappable.iterrows():
            lo = max(0, int(band["start_pos"] / bin_width))
            hi = min(len(mask), int(band["end_pos"] / bin_width) + 1)
            if hi > lo:
                mask[lo:hi] = False
        chr_cnv = cnv_data[chromosome][mask]

        chr_mean = _nan_safe(chr_cnv, np.mean)
        chr_std = _nan_safe(chr_cnv, np.std)
        logger.debug(f"Chromosome-wide mean: {chr_mean:.3f}, std: {chr_std:.3f}")

        chromosome_means = []
        for chrom in cnv_data:
            if chrom.startswith("chr") and chrom[3:].isdigit():
                mask = np.ones(len(cnv_data[chrom]), dtype=bool)
                cent = centromere_bed[centromere_bed["chrom"] == chrom]
                if not cent.empty:
                    cent_start = int(cent["start_pos"].iloc[0] / bin_width)
                    cent_end = int(cent["end_pos"].iloc[0] / bin_width)
                    mask[cent_start:cent_end] = False
                chrom_data = cnv_data[chrom][mask]
                if len(chrom_data) > 0:
                    chromosome_means.append(_nan_safe(chrom_data, np.mean))

        means_std = _nan_safe(chromosome_means, np.std)
        means_mean = _nan_safe(chromosome_means, np.mean)
        logger.debug(
            f"Mean of chromosome means: {means_mean:.3f}, std of means: {means_std:.3f}"
        )

        if chromosome.startswith("chr") and chromosome[3:].isdigit():
            gain_threshold = means_mean + (1.0 * means_std)
            loss_threshold = means_mean - (1.0 * means_std)
            cytoband_gain_threshold = chr_mean + (1.0 * chr_std)
            cytoband_loss_threshold = chr_mean - (1.0 * chr_std)
        elif chromosome == "chrX":
            gain_threshold = means_mean + (1.0 * means_std)
            loss_threshold = means_mean - (1.0 * means_std)
            cytoband_gain_threshold = chr_mean + (1.0 * chr_std)
            cytoband_loss_threshold = chr_mean - (1.0 * chr_std)
        elif chromosome == "chrY":
            if sex_estimate in ("Male", "XY"):
                gain_threshold = means_mean + (1.0 * means_std)
                loss_threshold = means_mean - (1.0 * means_std)
                cytoband_gain_threshold = chr_mean + (1.0 * chr_std)
                cytoband_loss_threshold = chr_mean - (1.0 * chr_std)
            else:
                gain_threshold = means_mean + (1.2 * means_std)
                loss_threshold = means_mean - (1.2 * means_std)
                cytoband_gain_threshold = chr_mean + (1.2 * chr_std)
                cytoband_loss_threshold = chr_mean - (1.2 * chr_std)
        else:
            gain_threshold = means_mean + (1.0 * means_std)
            loss_threshold = means_mean - (1.0 * means_std)
            cytoband_gain_threshold = chr_mean + (1.0 * chr_std)
            cytoband_loss_threshold = chr_mean - (1.0 * chr_std)

        # Gate the adaptive thresholds on the configured calling cut-off, so a
        # region is only called when it crosses BOTH. The adaptive rule is
        # relative to the chromosome's own noise, which means a flat chromosome
        # still produces regions above and below its own mean — on a track held
        # at log2 -0.10 it reported three GAIN/LOSS regions, all well inside the
        # +-0.3 window. Taking the stricter of the two keeps the adaptive rule
        # where it is more conservative (a noisy sample raises the bar) while
        # guaranteeing nothing inside the calling window is ever reported.
        from robin.classification_config import resolve_cnv_thresholds

        calling_gain, calling_loss = resolve_cnv_thresholds(
            chromosome, sex_estimate, cutoff_override
        )
        gain_threshold = max(gain_threshold, calling_gain)
        loss_threshold = min(loss_threshold, calling_loss)
        cytoband_gain_threshold = max(cytoband_gain_threshold, calling_gain)
        cytoband_loss_threshold = min(cytoband_loss_threshold, calling_loss)

        logger.debug(
            f"Thresholds - Whole chr gain: {gain_threshold:.3f}, loss: {loss_threshold:.3f}"
        )
        logger.debug(
            f"Thresholds - Cytoband gain: {cytoband_gain_threshold:.3f}, loss: {cytoband_loss_threshold:.3f}"
        )

        # Whole-chromosome calls defer to the same rule the Arm / Whole-Chromosome
        # table uses, so the two tables cannot disagree. The rule this replaces
        # compared the chromosome against the spread of all chromosome means,
        # which inflates as the rest of the genome becomes more aberrant: a
        # chromosome uniformly gained at log2 +0.55 was reported on a quiet
        # genome and silently dropped once four other chromosomes were aberrant.
        # That is backwards — an aneuploid tumour is where these calls matter.
        from robin.analysis.cnv_classification import analyze_chromosome_arms
        from robin.classification_config import is_whole_chromosome_event

        try:
            (
                p_arm_mean,
                q_arm_mean,
                p_gain,
                p_loss,
                q_gain,
                q_loss,
            ) = analyze_chromosome_arms(
                cnv_data, chromosome, bin_width, sex_estimate, cytobands_bed
            )
            if p_arm_mean is not None and q_arm_mean is not None:
                whole_chr_event, whole_chr_state = is_whole_chromosome_event(
                    p_arm_mean,
                    q_arm_mean,
                    p_gain,
                    p_loss,
                    q_gain,
                    q_loss,
                    calling_gain,
                    calling_loss,
                )
        except Exception:
            logger.debug(
                "Whole-chromosome check failed for %s", chromosome, exc_info=True
            )
            whole_chr_event = False
        if whole_chr_event:
            logger.debug(
                f"WHOLE CHROMOSOME EVENT DETECTED: {chromosome} {whole_chr_state}"
            )

        if whole_chr_event:
            genes_in_chr = gene_bed[gene_bed["chrom"] == chromosome]["gene"].tolist()

            merged_cytobands[merged_idx] = {
                "chrom": chromosome,
                "start_pos": chromosome_cytobands["start_pos"].min(),
                "end_pos": chromosome_cytobands["end_pos"].max(),
                "name": f"{chromosome} WHOLE CHROMOSOME {whole_chr_state}",
                "mean_cnv": chr_mean,
                "cnv_state": whole_chr_state,
                "length": chromosome_cytobands["end_pos"].max()
                - chromosome_cytobands["start_pos"].min(),
                "genes": genes_in_chr,
            }
            merged_idx += 1

        current_group = None

        for _, cytoband in chromosome_cytobands.iterrows():
            start_bin = int(cytoband["start_pos"] / bin_width)
            end_bin = int(cytoband["end_pos"] / bin_width)

            if str(cytoband.get("stain", "")) in UNMAPPABLE_CYTOBAND_STAINS:
                # Centromeric, heterochromatic and acrocentric stalk bands carry
                # no uniquely mappable sequence, so their depth is not a copy
                # number. On the log2 track they read as deep losses (−3 to −5)
                # and would otherwise dominate the table: 34 regional events on
                # this sample, of which every one of the top ten was one of
                # these. Not assessed rather than normal — they are genuinely
                # unmeasurable, not measured and found unchanged.
                mean_cnv = 0
                state = "NO_DATA"
            elif start_bin < len(cnv_data[chromosome]):
                region_cnv = cnv_data[chromosome][start_bin : end_bin + 1]
                mean_cnv = _nan_safe(region_cnv, np.mean)

                if mean_cnv > cytoband_gain_threshold:
                    state = "GAIN"
                elif mean_cnv < cytoband_loss_threshold:
                    state = "LOSS"
                else:
                    state = "NORMAL"
            else:
                mean_cnv = 0
                state = "NO_DATA"

            if current_group is None:
                current_group = {
                    "chrom": cytoband["chrom"],
                    "start_pos": cytoband["start_pos"],
                    "end_pos": cytoband["end_pos"],
                    "name": cytoband["name"],
                    "mean_cnv": [mean_cnv],
                    "cnv_state": state,
                    "bands": [cytoband["name"]],
                    "length": cytoband["end_pos"] - cytoband["start_pos"],
                    "genes": [],
                }
            elif state == current_group["cnv_state"]:
                current_group["end_pos"] = cytoband["end_pos"]
                current_group["mean_cnv"].append(mean_cnv)
                current_group["bands"].append(cytoband["name"])
            else:
                if current_group["cnv_state"] in SIGNIFICANT_CNV_STATES:
                    genes_in_region = gene_bed[
                        (gene_bed["chrom"] == current_group["chrom"])
                        & (gene_bed["start_pos"] <= current_group["end_pos"])
                        & (gene_bed["end_pos"] >= current_group["start_pos"])
                    ]["gene"].tolist()
                    current_group["genes"] = genes_in_region

                current_group["name"] = (
                    f"{current_group['chrom']} {current_group['bands'][0]}-{current_group['bands'][-1]}"
                )
                current_group["mean_cnv"] = _nan_safe(current_group["mean_cnv"], np.mean)
                current_group["length"] = (
                    current_group["end_pos"] - current_group["start_pos"]
                )

                if current_group["cnv_state"] not in ("NORMAL", "NO_DATA"):
                    merged_cytobands[merged_idx] = current_group
                    merged_idx += 1

                current_group = {
                    "chrom": cytoband["chrom"],
                    "start_pos": cytoband["start_pos"],
                    "end_pos": cytoband["end_pos"],
                    "name": cytoband["name"],
                    "mean_cnv": [mean_cnv],
                    "cnv_state": state,
                    "bands": [cytoband["name"]],
                    "length": cytoband["end_pos"] - cytoband["start_pos"],
                    "genes": [],
                }

        if current_group is not None:
            if current_group["cnv_state"] in SIGNIFICANT_CNV_STATES:
                genes_in_region = gene_bed[
                    (gene_bed["chrom"] == current_group["chrom"])
                    & (gene_bed["start_pos"] <= current_group["end_pos"])
                    & (gene_bed["end_pos"] >= current_group["start_pos"])
                ]["gene"].tolist()
                current_group["genes"] = genes_in_region

            current_group["name"] = (
                f"{current_group['chrom']} {current_group['bands'][0]}-{current_group['bands'][-1]}"
            )
            current_group["mean_cnv"] = _nan_safe(current_group["mean_cnv"], np.mean)
            current_group["length"] = (
                current_group["end_pos"] - current_group["start_pos"]
            )

            if current_group["cnv_state"] not in ("NORMAL", "NO_DATA"):
                merged_cytobands[merged_idx] = current_group
                merged_idx += 1

    merged_cytobands = merged_cytobands[:merged_idx]

    merged_df = pd.DataFrame(merged_cytobands)
    if not merged_df.empty:
        merged_df = merged_df.sort_values("start_pos")

    return merged_df
