#!/usr/bin/env python3
"""
Build the curated ITD / insertion hotspot JSON consumed by ``robin.analysis.itd_work``.

Windows are derived from the packaged GENCODE annotation rather than hard-coded,
so coordinates always match the exon model ROBIN uses elsewhere
(``load_canonical_exon_features_for_panel``): Ensembl-canonical transcript,
GFF3 1-based inclusive converted to 0-based inclusive to match pysam CIGAR
insertion anchors.

Each curated entry names the gene and the exon numbers (transcript order, i.e.
the GFF ``exon_number`` attribute) that carry the hotspot. ``anchor`` is an
optional published hg38 position used only as a self-check that the exon
numbering resolved to the intended region; it is not written to the output.

Usage:
    python scripts/make_itd_hotspots.py            # write resources JSON
    python scripts/make_itd_hotspots.py --stdout   # print without writing
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = REPO_ROOT / "src" / "robin" / "resources"
ANNOTATION = RESOURCES / "gencode.v45.basic.annotation.gff3.gz"
OUTPUT = RESOURCES / "itd_hotspots.hg38.json"

# Padding either side of the exon span (bp). Insertions are anchored at the
# reference position preceding the inserted bases, so a small pad keeps events
# at the very start/end of an exon inside the window.
EXON_PAD = 20


@dataclass(frozen=True)
class HotspotSpec:
    """Curated hotspot definition, resolved against the canonical transcript."""

    gene: str
    exons: Sequence[int]
    label: str
    min_length: int
    min_supporting_reads: int = 2
    min_frequency: float = 0.05
    anchor: Optional[int] = None  # published hg38 position, sanity-check only
    note: str = ""


# Curated windows. Kept deliberately small: each is a recurrent ITD / in-frame
# insertion locus with an established somatic role, and each is on at least one
# packaged panel BED (rCNS2 / CNS_TD_v1_Jul26 / AML). Genes absent from the
# active panel are dropped at load time by filter_hotspots_by_panel().
#
# Deliberately excluded:
#   ASXL1 c.1934dup  - 1 bp duplication inside a G-homopolymer; on ONT data this
#                      is dominated by basecaller indel artefact.
#   KMT2A PTD        - exon-level tandem duplication, not a CIGAR insertion.
#   MET exon 14      - splice-site deletions, not insertions.
SPECS: Tuple[HotspotSpec, ...] = (
    HotspotSpec(
        gene="FLT3",
        exons=(14, 15),
        label="ITD",
        min_length=6,
        min_supporting_reads=2,
        min_frequency=0.01,
        anchor=28034100,
        note="juxtamembrane domain / TKD1 ITD",
    ),
    HotspotSpec(
        gene="BCOR",
        exons=(15,),
        label="ITD",
        min_length=6,
        min_supporting_reads=2,
        min_frequency=0.05,
        note="3' exon 15 ITD (CNS HGNET-BCOR, clear cell sarcoma of kidney)",
    ),
    HotspotSpec(
        gene="KIT",
        exons=(11,),
        label="ITD",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.05,
        anchor=54727443,
        note="juxtamembrane exon 11 ITD / insertion",
    ),
    HotspotSpec(
        gene="NPM1",
        exons=(11,),
        label="insertion",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.02,
        anchor=171410540,
        note="type A/B/D 4 bp insertion in the terminal exon",
    ),
    HotspotSpec(
        gene="CALR",
        exons=(9,),
        label="insertion",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.05,
        note="exon 9 type 2 (5 bp) and related insertions",
    ),
    HotspotSpec(
        gene="CEBPA",
        exons=(1,),
        label="insertion",
        min_length=3,
        min_supporting_reads=3,
        min_frequency=0.05,
        note="single-exon gene; N-terminal and bZIP in-frame insertions",
    ),
    HotspotSpec(
        gene="EGFR",
        exons=(20,),
        label="insertion",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.05,
        anchor=55181320,
        note="exon 20 in-frame insertions",
    ),
    HotspotSpec(
        gene="ERBB2",
        exons=(20,),
        label="insertion",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.05,
        anchor=39724740,
        note="exon 20 in-frame insertions (Y772_A775dup et al.)",
    ),
    HotspotSpec(
        gene="JAK2",
        exons=(12,),
        label="insertion",
        min_length=3,
        min_supporting_reads=2,
        min_frequency=0.05,
        note="exon 12 in-frame insertions",
    ),
    HotspotSpec(
        gene="NOTCH1",
        exons=(34,),
        label="insertion",
        min_length=3,
        min_supporting_reads=3,
        min_frequency=0.05,
        note="terminal exon PEST domain insertions",
    ),
)


def load_canonical_exons(
    genes: Sequence[str],
    annotation: Path,
) -> Dict[str, Dict[int, Tuple[str, int, int, str, str]]]:
    """gene → exon_number → (chrom, start, end, transcript_id, exon_id), 1-based."""
    wanted = {g.upper() for g in genes}
    canonical: Dict[str, str] = {}
    exons: Dict[str, Dict[int, Tuple[str, int, int, str, str]]] = defaultdict(dict)

    with gzip.open(annotation, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature not in ("transcript", "exon"):
                continue
            attrs = dict(
                kv.split("=", 1) for kv in fields[8].split(";") if "=" in kv
            )
            gene = (attrs.get("gene_name") or "").upper()
            if gene not in wanted:
                continue
            if feature == "transcript":
                if "Ensembl_canonical" in attrs.get("tag", ""):
                    canonical[gene] = attrs["transcript_id"]
            elif attrs.get("transcript_id") == canonical.get(gene):
                number = attrs.get("exon_number")
                if number is None:
                    continue
                exons[gene][int(number)] = (
                    fields[0],
                    int(fields[3]),
                    int(fields[4]),
                    attrs["transcript_id"],
                    attrs.get("exon_id", ""),
                )
    return exons


def build(annotation: Path = ANNOTATION) -> Dict[str, dict]:
    exon_map = load_canonical_exons([spec.gene for spec in SPECS], annotation)

    out: Dict[str, dict] = {}
    problems: List[str] = []
    for spec in SPECS:
        available = exon_map.get(spec.gene.upper())
        if not available:
            problems.append(f"{spec.gene}: no Ensembl-canonical transcript found")
            continue
        missing = [n for n in spec.exons if n not in available]
        if missing:
            problems.append(
                f"{spec.gene}: exon(s) {missing} absent from canonical transcript "
                f"(has 1..{max(available)})"
            )
            continue

        chroms = {available[n][0] for n in spec.exons}
        if len(chroms) != 1:
            problems.append(f"{spec.gene}: exons span multiple contigs {chroms}")
            continue

        chrom = chroms.pop()
        transcript = available[spec.exons[0]][3]
        # GFF3 is 1-based inclusive; ITD anchors are 0-based pysam positions.
        start = min(available[n][1] for n in spec.exons) - 1 - EXON_PAD
        end = max(available[n][2] for n in spec.exons) - 1 + EXON_PAD

        if spec.anchor is not None and not start <= spec.anchor <= end:
            problems.append(
                f"{spec.gene}: published anchor {chrom}:{spec.anchor} outside "
                f"resolved window {chrom}:{start}-{end} — check exon numbering"
            )
            continue

        out[spec.gene] = {
            "chrom": chrom,
            "start": max(0, start),
            "end": end,
            "min_length": spec.min_length,
            "min_frequency": spec.min_frequency,
            "min_supporting_reads": spec.min_supporting_reads,
            "label": spec.label,
            "transcript": transcript,
        }

    if problems:
        raise SystemExit(
            "Refusing to write hotspots; unresolved specs:\n  "
            + "\n  ".join(problems)
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--stdout", action="store_true", help="print JSON instead of writing"
    )
    args = parser.parse_args()

    hotspots = build(args.annotation)
    text = json.dumps(hotspots, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return 0

    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(hotspots)} hotspots → {args.output}")
    for gene, entry in sorted(hotspots.items()):
        span = entry["end"] - entry["start"] + 1
        print(
            f"  {gene:<7} {entry['chrom']}:{entry['start']}-{entry['end']} "
            f"({span} bp, {entry['label']}, min_length={entry['min_length']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
