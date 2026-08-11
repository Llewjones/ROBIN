"""
Persistent record of which BAM files have already been accumulated, per sample
and per analysis type.

ROBIN's accumulators (target coverage, CNV read starts, fusion, ITD, ...) are
additive: each BAM's contribution is *added* to a running total on disk. The
watcher's in-memory ``processed`` set stops a file being handled twice within a
session, but it does not survive a restart, and ``submit_existing_paths``
re-submits every BAM in a watched folder unconditionally. A folder that is
watched again therefore adds the same reads to the same totals a second time,
producing coverage that is an exact integer multiple of the truth.

This module makes that "process each BAM once" rule durable by recording
completed work in ``<work_dir>/<sample_id>/processed_bams.json``.

Two rules govern every function here:

* **Fail open.** Any error - missing file, corrupt JSON, lock timeout - means
  "not yet processed". A false skip silently loses data, which is far worse
  than the duplicate work it would have avoided.
* **Never gate an unidentified sample.** When the sample ID is unknown, every
  such BAM would share one ledger, so a skip could drop a different sample's
  data. Duplicate work is the safer error.

Deleting a sample's output directory clears its ledger, which is already how
ROBIN documents re-analysing a sample.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from typing import Dict, Iterable, List, Set

logger = logging.getLogger("robin.ledger")

LEDGER_FILENAME = "processed_bams.json"
LOCK_FILENAME = ".processed_bams.lock"

# Sample IDs that must never be gated - see module docstring.
_UNGATED_SAMPLE_IDS = {"", "unknown", "None", "none"}


def _is_gateable(sample_id: str) -> bool:
    return bool(sample_id) and str(sample_id) not in _UNGATED_SAMPLE_IDS


def ledger_path(work_dir: str, sample_id: str) -> str:
    return os.path.join(work_dir, sample_id, LEDGER_FILENAME)


def _lock_path(work_dir: str, sample_id: str) -> str:
    return os.path.join(work_dir, sample_id, LOCK_FILENAME)


def bam_key(bam_path: str) -> str:
    """Identity of a BAM for ledger purposes: file name plus size.

    Name alone would re-double whenever a folder is moved or a mount path
    changes. Hashing the contents would be far too slow on a live run - these
    are thousands of files per sample. Name plus size is cheap and catches the
    cases that occur in practice, including a file that was replaced or is
    still being written.
    """
    name = os.path.basename(bam_path)
    try:
        size = os.path.getsize(bam_path)
    except OSError:
        size = -1
    return f"{name}:{size}"


def _load(path: str) -> Dict[str, List[str]]:
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def already_processed(
    work_dir: str, sample_id: str, job_type: str, bam_path: str
) -> bool:
    """Whether ``bam_path`` has already been accumulated for this job type."""
    if not work_dir or not _is_gateable(sample_id):
        return False
    try:
        recorded = _load(ledger_path(work_dir, sample_id)).get(job_type) or []
        return bam_key(bam_path) in set(recorded)
    except Exception:
        return False


def filter_unprocessed(
    work_dir: str, sample_id: str, job_type: str, bam_paths: Iterable[str]
) -> List[str]:
    """Return only those BAMs not yet accumulated for this job type.

    Reads the ledger once for the whole batch rather than once per file.
    """
    paths = list(bam_paths)
    if not work_dir or not _is_gateable(sample_id):
        return paths
    try:
        recorded: Set[str] = set(_load(ledger_path(work_dir, sample_id)).get(job_type) or [])
    except Exception:
        return paths
    return [p for p in paths if bam_key(p) not in recorded]


def record_processed(
    work_dir: str, sample_id: str, job_type: str, bam_paths: Iterable[str]
) -> None:
    """Record BAMs as accumulated for this job type.

    Call once per batch. Recording file-by-file rewrites the whole ledger each
    time, which costs ~78s per sample at 3,500 BAMs across six analysis types;
    batched it is under two seconds.

    Only ever called after the analysis reported success, so a failed job stays
    absent from the ledger and is retried if the sample is watched again. ROBIN
    has no retry mechanism of its own, so this is the only way such a job gets
    a second chance.
    """
    paths = list(bam_paths)
    if not work_dir or not _is_gateable(sample_id) or not paths:
        return

    target = ledger_path(work_dir, sample_id)
    lock_file = _lock_path(work_dir, sample_id)
    handle = None
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        handle = open(lock_file, "w")
        fcntl.flock(handle, fcntl.LOCK_EX)

        data = _load(target)
        recorded = set(data.get(job_type) or [])
        recorded.update(bam_key(p) for p in paths)
        data[job_type] = sorted(recorded)

        tmp = f"{target}.{os.getpid()}.tmp"
        with open(tmp, "w") as out:
            json.dump(data, out)
        os.replace(tmp, target)
    except Exception as exc:
        # Bookkeeping must never fail a run; the cost of losing a record is
        # duplicate work next time, not lost data.
        logger.debug("Could not record processed BAMs for %s: %s", sample_id, exc)
    finally:
        if handle is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            except Exception:
                pass
