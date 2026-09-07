"""Project configuration: one place the three notebooks agree on.

Everything comes from `config.yml` at the repository root, so changing the
event window or the map extent in one file changes it for the whole pipeline.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import catalog
from .render import GRIDS, Grid

ROOT = Path(__file__).resolve().parents[2]


def bootstrap() -> Path:
    """Make `import himawari` work from a notebook and load .env. Returns root.

    Notebooks live in `notebook/`, the package lives in `src/`, and neither is
    installed. Rather than require a `pip install -e .` before anything runs,
    this puts `src` on the path.
    """
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return ROOT


@dataclass
class Product:
    composite: str
    grid: Grid

    def __str__(self) -> str:
        return f"{self.composite} on {self.grid.name} {self.grid.shape[1]}x{self.grid.shape[0]}"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # --- window ---
    @property
    def start(self) -> datetime:
        return _as_utc(self.raw["window"]["start_utc"])

    @property
    def end(self) -> datetime:
        return _as_utc(self.raw["window"]["end_utc"])

    @property
    def step_minutes(self) -> int:
        return int(self.raw["window"].get("step_minutes", 10))

    def timeslots(self) -> list[datetime]:
        return catalog.timeslots(self.start, self.end, self.step_minutes)

    # --- local time ---
    @property
    def tz(self):
        """Local timezone used for on-screen labels, from event.timezone_offset_hours."""
        from datetime import timedelta, timezone as _tz

        hours = float((self.raw.get("event") or {}).get("timezone_offset_hours", 0))
        return _tz(timedelta(hours=hours))

    @property
    def tz_label(self) -> str:
        return str((self.raw.get("event") or {}).get("timezone_label", "local"))

    # --- event / marker ---
    @property
    def marker(self) -> tuple[float, float, str] | None:
        """(lon, lat, label) drawn on every frame, or None for no marker.

        Nothing about the pipeline depends on this being a volcano; it is just
        a point of interest. Omit the `event` block for an unmarked frame.
        """
        ev = self.raw.get("event") or {}
        if ev.get("lon") is None or ev.get("lat") is None:
            return None
        return (float(ev["lon"]), float(ev["lat"]), str(ev.get("name", "")))

    # --- grids ---
    @property
    def grids(self) -> dict[str, Grid]:
        """Built-in grids plus any defined in config.yml under `grids:`.

        A config entry overrides a built-in of the same name, so an unfamiliar
        region needs no code change:

            grids:
              tonga:
                extent: [-180, -30, -165, -12]   # lon_min lat_min lon_max lat_max
                resolution: 0.02
                label: "Hunga Tonga and the southwest Pacific"
        """
        out = dict(GRIDS)
        for name, spec in (self.raw.get("grids") or {}).items():
            ext = spec["extent"]
            if len(ext) != 4:
                raise ValueError(
                    f"grid {name!r}: extent must be "
                    "[lon_min, lat_min, lon_max, lat_max]"
                )
            out[name] = Grid(name, tuple(float(v) for v in ext),
                             float(spec.get("resolution", 0.02)),
                             str(spec.get("label", name)))
        return out

    # --- products ---
    @property
    def products(self) -> list[Product]:
        grids = self.grids
        out = []
        for entry in self.raw.get("products", []):
            grid_name = entry.get("grid", "regional")
            if grid_name not in grids:
                raise KeyError(f"unknown grid {grid_name!r}; known: {sorted(grids)}")
            out.append(Product(entry["composite"], grids[grid_name]))
        return out

    @property
    def composites(self) -> list[str]:
        return [p.composite for p in self.products]

    @property
    def bands(self) -> list[str]:
        """Exactly the bands the configured products need - nothing more.

        Derived from the product list rather than from a named band set, so
        dropping a composite from config.yml really does shrink the download.
        """
        return catalog.bands_for(self.composites)

    def segments(self) -> list[int]:
        """Segments spanning every configured grid."""
        lo, hi = 10, 1
        for p in self.products:
            segs = catalog.segments_for_extent(p.grid.extent)
            lo, hi = min(lo, segs[0]), max(hi, segs[-1])
        return list(range(lo, hi + 1))

    def tasks(self) -> list[catalog.FrameTask]:
        return catalog.frame_tasks(
            self.timeslots(), self.bands, self.segments(), self.satellite
        )

    # --- source ---
    @property
    def satellite(self) -> str:
        return self.raw.get("source", {}).get("satellite", catalog.SATELLITE)

    @property
    def backend_name(self) -> str:
        return self.raw.get("source", {}).get("backend", "aws")

    @property
    def download_workers(self) -> int:
        return int(self.raw.get("source", {}).get("workers", 12))

    def backend(self):
        from .fetch import AwsBackend, PTreeBackend

        if self.backend_name == "ptree":
            return PTreeBackend(
                os.environ.get("P_TREE_USER"), os.environ.get("P_TREE_PASS")
            )
        return AwsBackend(self.satellite)

    # --- paths ---
    def _path(self, key: str, default: str) -> Path:
        raw = self.raw.get("paths", {}).get(key, default)
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p

    @property
    def cache_dir(self) -> Path:
        return self._path("cache_dir", "himawari_cache")

    @property
    def frames_dir(self) -> Path:
        return self._path("frames_dir", "frames")

    @property
    def output_dir(self) -> Path:
        return self._path("output_dir", "output")

    @property
    def temp_dir(self) -> Path:
        return self._path("temp_dir", "temp")

    @property
    def resample_cache(self) -> Path:
        """Where satpy caches nearest-neighbour lookups.

        Defaults to a subdirectory of the HSD cache, but is overridable via
        `paths.resample_cache`. Two reasons to override it:

        * the cache is keyed on source and target geometry, not on the
          resampler's other arguments, so an entry written before a change to
          `radius_of_influence` or `reduce_data` stays in use afterwards and
          silently reproduces the old, wrong result;
        * a second config running concurrently should not write into a cache
          another run is reading.
        """
        raw = (self.raw.get("paths") or {}).get("resample_cache")
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else ROOT / p
        return self.cache_dir / "_resample"

    def make_dirs(self) -> None:
        for d in (self.cache_dir, self.frames_dir, self.output_dir,
                  self.temp_dir, self.resample_cache):
            d.mkdir(parents=True, exist_ok=True)

    def overlay_path(self, grid: Grid) -> Path:
        return self.temp_dir / f"overlay_{grid.name}.png"

    # --- render / animation ---
    @property
    def render_workers(self) -> int:
        return int(self.raw.get("render", {}).get("workers", 1))

    @property
    def skip_dark(self) -> bool:
        return bool(self.raw.get("render", {}).get("skip_dark", True))

    @property
    def animation(self) -> dict:
        return self.raw.get("animation", {})

    def summary(self) -> str:
        lines = [
            f"config      {self.path}",
            f"window      {self.start:%Y-%m-%d %H:%M} to {self.end:%Y-%m-%d %H:%M} UTC"
            f"  ({self.step_minutes} min, {len(self.timeslots())} slots)",
            f"            {self.start.astimezone(self.tz):%d %b %H:%M} to "
            f"{self.end.astimezone(self.tz):%d %b %H:%M} {self.tz_label}",
            f"source      {self.backend_name}  {self.satellite}",
            f"marker      {self.marker if self.marker else '(none)'}",
            f"bands       {' '.join(self.bands)}",
            f"segments    {self.segments()}",
            f"cache       {self.cache_dir}",
            "products:",
        ]
        lines += [f"            {p}" for p in self.products]
        return "\n".join(lines)


def _as_utc(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.strptime(str(value).strip(), "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def load(path=None) -> Config:
    path = Path(path) if path else ROOT / "config.yml"
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), path)
