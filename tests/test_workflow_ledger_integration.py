"""The processed-BAM ledger as wired into the workflow coordinator.

These cover the seam between the ledger and ROBIN's fan-out: that the gate
consults it with the right identity, that batched jobs record every BAM they
consumed rather than just the one they were named after, and that job types
which do not accumulate are left alone.

They also cover the window the ledger cannot reach on its own. Nothing is
recorded there until a BAM's analysis has *finished*, so submitting the same
folder twice in quick succession - adding it twice in the GUI, or an
on_modified event for a file the initial scan just picked up - would still
double a sample's coverage. An in-session record of what has been submitted
closes that window.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from robin.analysis.processed_ledger import already_processed, record_processed
from robin.workflow_ray import (
    ACCUMULATING_TYPES,
    TRIGGERS,
    BatchedJob,
    Job,
    RayFileWatcher,
    WorkflowContext,
    _already_accumulated,
    _job_source_bams,
    _record_accumulated,
    submit_existing_paths,
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


class _Awaitable:
    """Stands in for a Ray actor method: ``coord.thing.remote(...)``."""

    def __init__(self, result=None, sink=None):
        self._result = result
        self._sink = sink

    async def remote(self, *args):
        if self._sink is not None:
            self._sink.extend(args[0])
        return self._result


class _FakeCoord:
    def __init__(self) -> None:
        self.submitted: list = []
        self.get_reference = _Awaitable(None)
        self.get_target_panel = _Awaitable(None)
        self.can_accept_jobs = _Awaitable(True)
        self.submit_jobs = _Awaitable(None, sink=self.submitted)

    @property
    def submitted_paths(self) -> list:
        return [job.context.filepath for job in self.submitted]


def _watch_folder(root: Path, count: int = 4) -> Path:
    folder = root / "watch"
    folder.mkdir()
    for i in range(count):
        (folder / f"run_{i}.bam").write_bytes(b"x" * 32)
    return folder


def _submit(coord, folder: Path, seen=None) -> None:
    asyncio.run(
        submit_existing_paths(
            coord,
            [str(folder)],
            ["preprocessing:preprocessing"],
            patterns=["*.bam"],
            seen=seen,
        )
    )


def test_adding_the_same_folder_twice_submits_each_bam_once(tmp_path: Path) -> None:
    """The accident this guards against: clicking "add folder" twice. The
    on-disk ledger cannot cover it, because nothing is recorded there until the
    first pass has finished analysing."""
    folder = _watch_folder(tmp_path)
    coord = _FakeCoord()
    seen: set = set()

    _submit(coord, folder, seen)
    _submit(coord, folder, seen)

    assert len(coord.submitted_paths) == 4
    assert sorted(coord.submitted_paths) == sorted(set(coord.submitted_paths))


def test_a_rescan_still_picks_up_files_the_watcher_missed(tmp_path: Path) -> None:
    """Re-adding a folder must stay useful: only files already submitted are
    skipped, not the whole folder."""
    folder = _watch_folder(tmp_path, count=3)
    coord = _FakeCoord()
    seen: set = set()

    _submit(coord, folder, seen)
    (folder / "run_late.bam").write_bytes(b"x" * 32)
    _submit(coord, folder, seen)

    assert len(coord.submitted_paths) == 4
    assert str(folder / "run_late.bam") in coord.submitted_paths


def test_without_a_shared_record_behaviour_is_unchanged(tmp_path: Path) -> None:
    """Callers that pass nothing keep the old semantics."""
    folder = _watch_folder(tmp_path, count=2)
    coord = _FakeCoord()

    _submit(coord, folder)
    _submit(coord, folder)

    assert len(coord.submitted_paths) == 4


def test_a_bulk_submitted_file_is_not_handled_again_by_the_watcher(
    tmp_path: Path,
) -> None:
    """A BAM still being written is submitted by the initial scan, then an
    on_modified event arrives for it moments later. Both must not count."""
    folder = _watch_folder(tmp_path, count=3)
    coord = _FakeCoord()
    watcher = RayFileWatcher(
        coord, ["preprocessing:preprocessing"], None, patterns=["*.bam"]
    )

    _submit(coord, folder, watcher.processed)
    assert len(coord.submitted_paths) == 3

    watcher._handle(str(folder / "run_0.bam"))
    assert watcher._pending_jobs == []
