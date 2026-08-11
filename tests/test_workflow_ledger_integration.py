"""The processed-BAM ledger as wired into the workflow coordinator.

These cover the seam between the ledger and ROBIN's fan-out: that the gate
consults it with the right identity, that batched jobs record every BAM they
consumed rather than just the one they were named after, and that job types
which do not accumulate are left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robin.analysis.processed_ledger import already_processed, record_processed
from robin.workflow_ray import (
    ACCUMULATING_TYPES,
    TRIGGERS,
    BatchedJob,
    Job,
    WorkflowContext,
    _already_accumulated,
    _job_source_bams,
    _record_accumulated,
)

SAMPLE = "S1"


def _bam(directory: Path, name: str) -> str:
    path = directory / name
    path.write_bytes(b"x" * 64)
    return str(path)


def _context(work_dir: Path, filepath: str, sample_id: str = SAMPLE) -> WorkflowContext:
    ctx = WorkflowContext(filepath)
    ctx.add_metadata("work_dir", str(work_dir))
    ctx.add_metadata("bam_metadata", {"sample_id": sample_id})
    return ctx


def _job(job_type: str, ctx: WorkflowContext) -> Job:
    return Job(1, job_type, job_type, [f"{job_type}:{job_type}"], 0, ctx)


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    (tmp_path / "out" / SAMPLE).mkdir(parents=True)
    return tmp_path / "out"


def test_every_analysis_triggered_by_preprocessing_is_guarded() -> None:
    """A new accumulating analysis added to TRIGGERS must not silently escape
    the ledger, or that analysis alone would start multiplying again."""
    assert ACCUMULATING_TYPES == set(TRIGGERS["preprocessing"])


def test_the_gate_lets_an_unseen_bam_through(work_dir: Path, tmp_path: Path) -> None:
    ctx = _context(work_dir, _bam(tmp_path, "run_0.bam"))
    assert not _already_accumulated(ctx, "target")


def test_the_gate_blocks_a_bam_already_in_the_totals(
    work_dir: Path, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam)
    record_processed(str(work_dir), SAMPLE, "target", [bam])

    assert _already_accumulated(ctx, "target")


def test_a_non_accumulating_type_is_never_gated(
    work_dir: Path, tmp_path: Path
) -> None:
    """Classifiers and IGV builds replace their output rather than adding to
    it, so gating them would only withhold refreshed results."""
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam)
    record_processed(str(work_dir), SAMPLE, "sturgeon", [bam])

    assert not _already_accumulated(ctx, "sturgeon")


def test_a_completed_job_is_recorded(work_dir: Path, tmp_path: Path) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam)
    _record_accumulated(_job("target", ctx), ctx)

    assert already_processed(str(work_dir), SAMPLE, "target", bam)


def test_a_non_accumulating_job_is_not_recorded(
    work_dir: Path, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam)
    _record_accumulated(_job("sturgeon", ctx), ctx)

    assert not already_processed(str(work_dir), SAMPLE, "sturgeon", bam)


def test_a_batched_job_records_every_bam_in_the_batch(
    work_dir: Path, tmp_path: Path
) -> None:
    """CNV batches many BAMs into one job. Recording only the primary context
    would leave the rest of the batch free to be counted a second time."""
    bams = [_bam(tmp_path, f"run_{i}.bam") for i in range(5)]
    contexts = [_context(work_dir, path) for path in bams]
    primary = contexts[0]
    primary.add_metadata(
        "_batched_job",
        BatchedJob(1, "cnv", "cnv", ["cnv:cnv"], 0, contexts, "batch-1", SAMPLE),
    )

    job = _job("cnv", primary)
    assert _job_source_bams(job, primary) == bams

    _record_accumulated(job, primary)
    for path in bams:
        assert already_processed(str(work_dir), SAMPLE, "cnv", path)


def test_an_unbatched_job_records_just_its_own_file(
    work_dir: Path, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam)
    assert _job_source_bams(_job("target", ctx), ctx) == [bam]


def test_a_job_without_a_work_dir_is_not_gated(tmp_path: Path) -> None:
    ctx = WorkflowContext(_bam(tmp_path, "run_0.bam"))
    ctx.add_metadata("bam_metadata", {"sample_id": SAMPLE})

    assert not _already_accumulated(ctx, "target")
    _record_accumulated(_job("target", ctx), ctx)  # must not raise


def test_an_unidentified_sample_is_not_gated(work_dir: Path, tmp_path: Path) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    ctx = _context(work_dir, bam, sample_id="unknown")
    _record_accumulated(_job("target", ctx), ctx)

    assert not _already_accumulated(ctx, "target")


def test_the_full_cycle_stops_a_second_pass_over_the_same_folder(
    work_dir: Path, tmp_path: Path
) -> None:
    """The bug this exists to prevent: watching a folder again re-added every
    BAM to totals that are summed, multiplying the sample's coverage."""
    bams = [_bam(tmp_path, f"run_{i}.bam") for i in range(10)]

    first_pass = [p for p in bams if not _already_accumulated(_context(work_dir, p), "target")]
    assert first_pass == bams
    for path in first_pass:
        ctx = _context(work_dir, path)
        _record_accumulated(_job("target", ctx), ctx)

    second_pass = [p for p in bams if not _already_accumulated(_context(work_dir, p), "target")]
    assert second_pass == []
