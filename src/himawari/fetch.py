"""Downloading HSD files, from AWS (default) or P-Tree FTP (fallback).

Both backends expose the same two operations, so the notebooks never care
which one is in use:

    backend.size(task)          -> int | None      remote byte count
    backend.fetch(task, dest)   -> int             bytes written

AWS is the default because the objects are byte-identical to P-Tree's
/jma/hsd tree, the bucket is public (no credentials), and plain HTTPS
parallelises to a dozen workers without upsetting anyone. P-Tree stays
implemented because it is the authoritative JAXA source and the only place
the /pub geophysical products live.

Every download goes to `<name>.part` and is renamed only after the byte
count matches, so an interrupted run never leaves a truncated file that a
later run would mistake for good data.
"""

from __future__ import annotations

import bz2
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.etree import ElementTree

import requests

from .catalog import FrameTask, aws_bucket, scene_dir

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# --- Result bookkeeping ----------------------------------------------------

@dataclass
class Outcome:
    task: FrameTask
    status: str  # "ok" | "cached" | "missing" | "error"
    bytes: int = 0
    detail: str = ""


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    seconds: float = 0.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return out

    @property
    def downloaded_bytes(self) -> int:
        return sum(o.bytes for o in self.outcomes if o.status == "ok")

    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status in ("missing", "error")]

    def incomplete_times(self) -> list[datetime]:
        """Timeslots where at least one file did not arrive."""
        bad = {o.task.time for o in self.failures()}
        return sorted(bad)

    def summary(self) -> str:
        c = self.counts()
        gb = self.downloaded_bytes / 2**30
        parts = [f"{k}={v}" for k, v in sorted(c.items())]
        return (
            f"{len(self.outcomes)} files | " + ", ".join(parts) +
            f" | {gb:.2f} GB in {self.seconds/60:.1f} min"
            + (f" ({gb*1024/max(self.seconds,1e-9):.1f} MB/s)" if self.seconds else "")
        )

    def to_csv(self, path) -> Path:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_utc", "band", "segment", "status", "bytes", "detail"])
            for o in self.outcomes:
                w.writerow([
                    o.task.time.strftime("%Y-%m-%d %H:%M"),
                    o.task.band, o.task.segment, o.status, o.bytes, o.detail,
                ])
        return path


# --- AWS -------------------------------------------------------------------

class AwsBackend:
    """Public noaa-himawari9 bucket over anonymous HTTPS.

    No boto3, no credentials, no signing: the bucket allows unauthenticated
    GET and ListObjectsV2, so `requests` is the whole client.
    """

    name = "aws"

    def __init__(self, satellite: str = "H09", timeout: int = 60, retries: int = 4):
        self.bucket = aws_bucket(satellite)
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            # A generous pool: one connection per worker, reused across files.
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=8, max_retries=0
            )
            s.mount("https://", adapter)
            self._local.session = s
        return s

    def size(self, task: FrameTask) -> int | None:
        try:
            r = self.session.head(task.url, timeout=self.timeout)
            if r.status_code == 200:
                return int(r.headers["Content-Length"])
        except requests.RequestException:
            pass
        return None

    def list_slot(self, t: datetime) -> dict[str, int]:
        """Every object in one timeslot directory, as {filename: bytes}.

        One request instead of ~110 HEADs, which is what makes the size
        estimate in notebook 01 fast enough to actually run.
        """
        prefix = f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/"
        out: dict[str, int] = {}
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            r = self.session.get(
                f"https://{self.bucket}.s3.amazonaws.com/",
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
            root = ElementTree.fromstring(r.content)
            for c in root.findall(f"{_S3_NS}Contents"):
                key = c.findtext(f"{_S3_NS}Key", "")
                out[key.rsplit("/", 1)[-1]] = int(c.findtext(f"{_S3_NS}Size", "0"))
            if root.findtext(f"{_S3_NS}IsTruncated", "false") != "true":
                break
            token = root.findtext(f"{_S3_NS}NextContinuationToken")
        return out

    def available_slots(self, day: datetime) -> list[str]:
        """The HHMM directories that actually exist for a given UTC day."""
        prefix = f"AHI-L1b-FLDK/{day:%Y/%m/%d}/"
        r = self.session.get(
            f"https://{self.bucket}.s3.amazonaws.com/",
            params={"list-type": "2", "prefix": prefix, "delimiter": "/",
                    "max-keys": "1000"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        root = ElementTree.fromstring(r.content)
        slots = []
        for p in root.findall(f"{_S3_NS}CommonPrefixes"):
            name = (p.findtext(f"{_S3_NS}Prefix") or "").rstrip("/").rsplit("/", 1)[-1]
            if name:
                slots.append(name)
        return sorted(slots)

    def fetch(self, task: FrameTask, dest: Path) -> int:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._fetch_once(task, dest)
            except _NotFound:
                raise
            except Exception as exc:  # noqa: BLE001 - retry anything transient
                last = exc
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"{self.retries} attempts failed: {last}")

    def _fetch_once(self, task: FrameTask, dest: Path) -> int:
        part = dest.with_suffix(dest.suffix + ".part")
        with self.session.get(task.url, stream=True, timeout=self.timeout) as r:
            if r.status_code == 404:
                raise _NotFound(task.key)
            r.raise_for_status()
            expected = int(r.headers.get("Content-Length", 0))
            written = 0
            with part.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
        if expected and written != expected:
            part.unlink(missing_ok=True)
            raise OSError(f"short read: {written} of {expected} bytes")
        os.replace(part, dest)
        return written


class _NotFound(Exception):
    """The object genuinely is not there; retrying will not help."""


# --- P-Tree ----------------------------------------------------------------

class PTreeBackend:
    """ftp.ptree.jaxa.jp, the authoritative JAXA distribution point.

    Credentials come from the environment (P_TREE_USER / P_TREE_PASS), which
    `load_dotenv()` populates from a gitignored `.env`. They are never written
    to a notebook cell, an output, or the manifest.

    ftplib connections are not thread-safe, so each worker thread opens and
    keeps its own. Be conservative with `workers`: P-Tree is a shared academic
    service, not a CDN.
    """

    name = "ptree"

    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        host: str = "ftp.ptree.jaxa.jp",
        timeout: int = 120,
        retries: int = 3,
    ):
        self.host = host
        self.user = user or os.environ.get("P_TREE_USER", "")
        self.password = password or os.environ.get("P_TREE_PASS", "")
        if not self.user or not self.password:
            raise RuntimeError(
                "P-Tree needs credentials. Register at "
                "https://www.eorc.jaxa.jp/ptree/registration_top.html then put "
                "P_TREE_USER and P_TREE_PASS in .env (see .env.example)."
            )
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    def _conn(self):
        from ftplib import FTP

        ftp = getattr(self._local, "ftp", None)
        if ftp is not None:
            try:
                ftp.voidcmd("NOOP")
                return ftp
            except Exception:  # noqa: BLE001 - stale connection, reopen below
                try:
                    ftp.close()
                except Exception:  # noqa: BLE001
                    pass
        ftp = FTP(self.host, timeout=self.timeout)
        ftp.login(self.user, self.password)
        ftp.set_pasv(True)
        self._local.ftp = ftp
        return ftp

    def size(self, task: FrameTask) -> int | None:
        try:
            ftp = self._conn()
            ftp.voidcmd("TYPE I")
            return ftp.size(task.ftp_path)
        except Exception:  # noqa: BLE001
            return None

    def fetch(self, task: FrameTask, dest: Path) -> int:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._fetch_once(task, dest)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if "550" in str(exc):  # no such file
                    raise _NotFound(task.ftp_path) from exc
                self._local.ftp = None  # force a reconnect
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"{self.retries} attempts failed: {last}")

    def _fetch_once(self, task: FrameTask, dest: Path) -> int:
        ftp = self._conn()
        ftp.voidcmd("TYPE I")
        expected = ftp.size(task.ftp_path)
        part = dest.with_suffix(dest.suffix + ".part")
        written = 0
        with part.open("wb") as fh:
            def _write(block: bytes) -> None:
                nonlocal written
                fh.write(block)
                written += len(block)

            ftp.retrbinary(f"RETR {task.ftp_path}", _write, blocksize=1 << 20)
        if expected and written != expected:
            part.unlink(missing_ok=True)
            raise OSError(f"short read: {written} of {expected} bytes")
        os.replace(part, dest)
        return written


# --- Driver ----------------------------------------------------------------

def is_cached(task: FrameTask, cache_dir, min_bytes: int = 1024) -> bool:
    """A file counts as cached if it exists and is not obviously truncated.

    Cheap by design: this runs once per file per notebook run. Use
    `verify_bz2` for the expensive, thorough check.
    """
    p = task.path(cache_dir)
    try:
        return p.stat().st_size >= min_bytes
    except OSError:
        return False


def verify_bz2(path: Path, chunk: int = 1 << 22) -> bool:
    """Fully decompress a file to prove the bz2 stream is intact.

    Slow (it reads every byte), so it is opt-in. Worth running once over a
    completed cache before a long processing job, because satpy's failure mode
    on a truncated member is an opaque struct error 300 frames in.
    """
    try:
        with bz2.open(path, "rb") as fh:
            while fh.read(chunk):
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def download(
    tasks: Sequence[FrameTask],
    cache_dir,
    backend=None,
    workers: int = 8,
    skip_existing: bool = True,
    progress: bool = True,
    on_done: Callable[[Outcome], None] | None = None,
) -> Report:
    """Fetch every task into `cache_dir`, in parallel, resumably.

    Safe to re-run: already-cached files are skipped, so the normal way to
    recover from a network drop is to run the same cell again.
    """
    cache_dir = Path(cache_dir)
    backend = backend or AwsBackend()

    pending: list[FrameTask] = []
    report = Report()
    for t in tasks:
        if skip_existing and is_cached(t, cache_dir):
            report.outcomes.append(Outcome(t, "cached", t.path(cache_dir).stat().st_size))
        else:
            pending.append(t)

    if not pending:
        return report

    for t in {task.time for task in pending}:
        scene_dir(cache_dir, t).mkdir(parents=True, exist_ok=True)

    bar = _progress_bar(len(pending), enabled=progress)
    started = time.perf_counter()

    def _one(task: FrameTask) -> Outcome:
        dest = task.path(cache_dir)
        try:
            n = backend.fetch(task, dest)
            return Outcome(task, "ok", n)
        except _NotFound:
            return Outcome(task, "missing", 0, "not on server")
        except Exception as exc:  # noqa: BLE001
            return Outcome(task, "error", 0, f"{type(exc).__name__}: {exc}")

    # FTP throughput collapses past a handful of connections, and P-Tree is a
    # shared service; cap it regardless of what the caller asked for.
    if getattr(backend, "name", "") == "ptree":
        workers = min(workers, 4)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, t): t for t in pending}
        for fut in as_completed(futures):
            outcome = fut.result()
            report.outcomes.append(outcome)
            if on_done:
                on_done(outcome)
            if bar is not None:
                bar.update(1)
                if outcome.status == "ok":
                    bar.set_postfix_str(
                        f"{report.downloaded_bytes / 2**30:.2f} GB", refresh=False
                    )

    if bar is not None:
        bar.close()
    report.seconds = time.perf_counter() - started
    report.outcomes.sort(key=lambda o: (o.task.time, o.task.band, o.task.segment))
    return report


def missing(tasks: Iterable[FrameTask], cache_dir) -> list[FrameTask]:
    return [t for t in tasks if not is_cached(t, cache_dir)]


def complete_times(
    tasks: Sequence[FrameTask], cache_dir
) -> tuple[list[datetime], list[datetime]]:
    """Split the requested timeslots into (fully cached, incomplete).

    Notebook 02 renders only the complete ones: a scene missing a segment
    produces a frame with a blank stripe across it, which is worse than a
    dropped frame.
    """
    by_time: dict[datetime, list[FrameTask]] = {}
    for t in tasks:
        by_time.setdefault(t.time, []).append(t)
    good, bad = [], []
    for when, group in sorted(by_time.items()):
        (good if all(is_cached(t, cache_dir) for t in group) else bad).append(when)
    return good, bad


def estimate(
    tasks: Sequence[FrameTask],
    backend=None,
    sample_slots: int = 6,
) -> dict:
    """Project total download size by listing a spread of sample timeslots.

    Sampling more than one slot matters more than you would expect. bz2
    compresses a night-time visible band - a near-uniform dark field - to a
    fraction of its daylight size, so a single 17:00 UTC sample understates a
    three-day true-colour download by about half. Spreading the samples evenly
    across the period averages the diurnal swing out.

    Returns min/max per slot alongside the mean so you can see how wide that
    swing is for the band set you picked.
    """
    backend = backend or AwsBackend()
    if not hasattr(backend, "list_slot"):
        raise TypeError("estimate() needs a backend with list_slot(); use AwsBackend")

    times = sorted({t.time for t in tasks})
    wanted = {(t.band, t.segment) for t in tasks}

    n = max(1, min(sample_slots, len(times)))
    picks = [times[round(i * (len(times) - 1) / max(1, n - 1))] for i in range(n)]

    per_slot: list[int] = []
    for when in dict.fromkeys(picks):
        try:
            listing = backend.list_slot(when)
        except Exception:  # noqa: BLE001 - a missing sample slot is not fatal
            continue
        total = 0
        for name, size in listing.items():
            parts = name.split("_")
            if len(parts) < 8:
                continue
            band, seg = parts[4], int(parts[7][1:3])
            if (band, seg) in wanted:
                total += size
        if total:
            per_slot.append(total)

    if not per_slot:
        raise RuntimeError("no sample timeslot could be listed; check the dates")

    mean = sum(per_slot) / len(per_slot)
    return {
        "files": len(tasks),
        "timeslots": len(times),
        "sampled": len(per_slot),
        "bytes_per_slot": mean,
        "mb_per_slot": mean / 2**20,
        "mb_per_slot_min": min(per_slot) / 2**20,
        "mb_per_slot_max": max(per_slot) / 2**20,
        "total_bytes": mean * len(times),
        "total_gb": mean * len(times) / 2**30,
    }


def _progress_bar(total: int, enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, unit="file", desc="download", smoothing=0.05)
    except ImportError:
        return None
