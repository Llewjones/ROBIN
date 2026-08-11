"""Tests for the processed-BAM ledger that stops coverage being multiplied.

ROBIN's accumulators add each BAM's contribution to a running per-sample total.
Watching a folder a second time used to re-add every BAM, producing coverage
that was an exact integer multiple of the truth. The ledger records completed
work so that cannot happen, and these tests pin the behaviour that makes it
safe to do so.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from robin.analysis.processed_ledger import (
    LEDGER_FILENAME,
    already_processed,
    bam_key,
    filter_unprocessed,
    ledger_path,
    record_processed,
)

SAMPLE = "S1"


def _bam(directory: Path, name: str, size: int = 128) -> str:
    path = directory / name
    path.write_bytes(b"x" * size)
    return str(path)


@pytest.fixture()
def work_dir(tmp_path: Path) -> str:
    (tmp_path / SAMPLE).mkdir()
    return str(tmp_path)


def test_a_bam_is_not_skipped_until_it_has_been_recorded(
    work_dir: str, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    assert not already_processed(work_dir, SAMPLE, "target", bam)


def test_a_recorded_bam_is_skipped_which_is_what_stops_the_doubling(
    work_dir: str, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    record_processed(work_dir, SAMPLE, "target", [bam])
    assert already_processed(work_dir, SAMPLE, "target", bam)


def test_recording_one_bam_does_not_skip_a_different_one(
    work_dir: str, tmp_path: Path
) -> None:
    first = _bam(tmp_path, "run_0.bam")
    second = _bam(tmp_path, "run_1.bam")
    record_processed(work_dir, SAMPLE, "target", [first])
    assert not already_processed(work_dir, SAMPLE, "target", second)


def test_each_analysis_type_is_tracked_separately(
    work_dir: str, tmp_path: Path
) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    record_processed(work_dir, SAMPLE, "target", [bam])

    assert already_processed(work_dir, SAMPLE, "target", bam)
    assert not already_processed(work_dir, SAMPLE, "cnv", bam)

    record_processed(work_dir, SAMPLE, "cnv", [bam])
    assert already_processed(work_dir, SAMPLE, "cnv", bam)


def test_a_failed_analysis_is_retried_because_it_was_never_recorded(
    work_dir: str, tmp_path: Path
) -> None:
    """ROBIN has no retry mechanism, so re-watching is a failed job's only
    second chance. Recording on success alone is what preserves it."""
    bam = _bam(tmp_path, "run_0.bam")
    record_processed(work_dir, SAMPLE, "target", [bam])

    assert not already_processed(work_dir, SAMPLE, "fusion", bam)


def test_a_corrupt_ledger_fails_open_and_the_bam_is_processed(
    work_dir: str, tmp_path: Path
) -> None:
    """A false skip loses data silently; duplicate work does not."""
    bam = _bam(tmp_path, "run_0.bam")
    record_processed(work_dir, SAMPLE, "target", [bam])
    Path(ledger_path(work_dir, SAMPLE)).write_text("{ not json at all")

    assert not already_processed(work_dir, SAMPLE, "target", bam)


def test_a_missing_ledger_fails_open(work_dir: str, tmp_path: Path) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    assert not os.path.exists(ledger_path(work_dir, SAMPLE))
    assert not already_processed(work_dir, SAMPLE, "target", bam)


@pytest.mark.parametrize("sample_id", ["unknown", "", None])
def test_an_unidentified_sample_is_never_recorded_or_skipped(
    work_dir: str, tmp_path: Path, sample_id
) -> None:
    """Every unidentified BAM would share one ledger, so a skip could drop a
    different sample's data."""
    bam = _bam(tmp_path, "run_0.bam")
    record_processed(work_dir, sample_id, "target", [bam])

    assert not already_processed(work_dir, sample_id, "target", bam)
    assert filter_unprocessed(work_dir, sample_id, "target", [bam]) == [bam]


def test_a_bam_that_changed_size_is_treated_as_a_new_file(
    work_dir: str, tmp_path: Path
) -> None:
    """Guards against skipping a file that was still being written."""
    bam = _bam(tmp_path, "run_0.bam", size=128)
    record_processed(work_dir, SAMPLE, "target", [bam])
    assert already_processed(work_dir, SAMPLE, "target", bam)

    Path(bam).write_bytes(b"x" * 256)
    assert not already_processed(work_dir, SAMPLE, "target", bam)


def test_identically_named_bams_in_different_folders_share_an_entry(
    work_dir: str, tmp_path: Path
) -> None:
    """The same BAM reached by another path must not be counted twice."""
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    original = _bam(left, "run_0.bam")
    same_file_other_path = _bam(right, "run_0.bam")

    record_processed(work_dir, SAMPLE, "target", [original])
    assert already_processed(work_dir, SAMPLE, "target", same_file_other_path)


def test_missing_work_dir_never_gates(tmp_path: Path) -> None:
    bam = _bam(tmp_path, "run_0.bam")
    assert not already_processed(None, SAMPLE, "target", bam)
    assert filter_unprocessed(None, SAMPLE, "target", [bam]) == [bam]


def test_filter_unprocessed_returns_only_the_outstanding_bams(
    work_dir: str, tmp_path: Path
) -> None:
    bams = [_bam(tmp_path, f"run_{i}.bam") for i in range(5)]
    record_processed(work_dir, SAMPLE, "target", bams[:3])

    assert filter_unprocessed(work_dir, SAMPLE, "target", bams) == bams[3:]


def test_recording_is_additive_across_batches(work_dir: str, tmp_path: Path) -> None:
    bams = [_bam(tmp_path, f"run_{i}.bam") for i in range(6)]
    record_processed(work_dir, SAMPLE, "target", bams[:3])
    record_processed(work_dir, SAMPLE, "target", bams[3:])

    assert filter_unprocessed(work_dir, SAMPLE, "target", bams) == []


def test_recording_nothing_does_not_create_a_ledger(work_dir: str) -> None:
    record_processed(work_dir, SAMPLE, "target", [])
    assert not os.path.exists(ledger_path(work_dir, SAMPLE))


def test_bam_key_survives_a_deleted_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "gone.bam")
    assert bam_key(missing) == "gone.bam:-1"


def _record_batch(args) -> None:
    work_dir, bams = args
    record_processed(work_dir, SAMPLE, "target", bams)


def test_concurrent_workers_do_not_lose_records(
    work_dir: str, tmp_path: Path
) -> None:
    """Eight analysis workers writing one sample's ledger at once."""
    batches = [
        [_bam(tmp_path, f"w{worker}_{i}.bam") for i in range(40)]
        for worker in range(8)
    ]
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(_record_batch, [(work_dir, batch) for batch in batches]))

    with open(ledger_path(work_dir, SAMPLE)) as handle:
        recorded = json.load(handle)["target"]

    assert len(recorded) == 320


def test_the_ledger_lives_inside_the_sample_directory(work_dir: str) -> None:
    """Deleting a sample's output folder must clear its ledger - that is how
    ROBIN already tells users to re-analyse a sample."""
    assert ledger_path(work_dir, SAMPLE) == os.path.join(
        work_dir, SAMPLE, LEDGER_FILENAME
    )
