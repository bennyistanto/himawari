"""Turning cached HSD files into annotated, map-projected PNG frames.

The expensive parts of this stage are, in order:

  1. bz2 decompression, which satpy redoes every time a Scene is built
  2. computing the resampling neighbour indices
  3. the composite arithmetic itself

So the design pushes back on all three: one Scene per timeslot serving every
requested composite (1), a disk-cached nearest-neighbour lookup reused across
all frames because the source and target geometry never change (2), and dask
doing (3) lazily so only the pixels inside the target area are ever computed.

Frames come out as plain PNGs in the target grid's own pixel space, which
means a pre-rendered transparent coastline overlay lines up exactly and can be
pasted onto all several hundred of them for free.
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import re
import shutil
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from .catalog import (
    COMPOSITE_REQUIREMENTS,
    SUBSAT_LON,
    geos_line_col,
    scene_dir,
    segment_of_line,
)

# The project ships an extra AHI composite (SO2 / volcanic emissions) that
# satpy does not, plus its enhancement. Both live under config/satpy.
_LOCAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "satpy"


def use_local_config(extra: Sequence[str] = ()) -> list[str]:
    """Prepend this project's satpy config directory to the search path.

    Satpy merges every `composites/ahi.yaml` it finds along the path rather
    than letting the first one win, so this adds `volcanic_emissions` to the
    built-in AHI set instead of replacing it. Call once per session, before
    building any Scene.
    """
    import satpy

    paths = [str(_LOCAL_CONFIG), *map(str, extra)]
    current = satpy.config.get("config_path", [])
    for p in current:
        if p not in paths:
            paths.append(p)
    satpy.config.set(config_path=paths)
    return paths


# --- Target grids ----------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    """A named lat/lon output grid."""

    name: str
    extent: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    resolution: float                          # degrees per pixel
    label: str = ""

    def area(self):
        from pyresample import create_area_def

        return create_area_def(
            area_id=self.name,
            projection={"proj": "longlat", "datum": "WGS84"},
            area_extent=list(self.extent),
            resolution=self.resolution,
            units="degrees",
            description=self.label or self.name,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.area().shape


# A plain longlat "projection" is deliberate: the output pixel grid is then
# exactly plate carree, so cartopy's PlateCarree overlays register 1:1 with the
# image and no reprojection is needed to draw coastlines on top.
GRIDS = {
    # Java, Sumatra, the Sunda Strait and the near Indian Ocean.
    "regional": Grid("regional", (95.0, -12.0, 120.0, 2.0), 0.02,
                     "Sunda Strait and the eastern Indian Ocean"),
    # Same box at true-colour resolution (~550 m).
    "regional_hires": Grid("regional_hires", (95.0, -12.0, 120.0, 2.0), 0.005,
                           "Sunda Strait and the eastern Indian Ocean"),
    # Follow the plume a long way west.
    "wide": Grid("wide", (80.0, -20.0, 125.0, 5.0), 0.02,
                 "Indian Ocean plume tracking"),
    # Tight on the vent.
    "zoom": Grid("zoom", (103.0, -9.0, 110.0, -3.0), 0.005,
                 "Anak Krakatau and the Sunda Strait"),
}

KRAKATAU = (105.423, -6.102, "Anak Krakatau")
"""The worked example's marker. Any (lon, lat, label) works; see config.yml."""


# --- Scene loading ---------------------------------------------------------

def scene_files(cache_dir, when: datetime, bands: Sequence[str] | None = None) -> list[str]:
    """Cached HSD files for one timeslot, optionally restricted to some bands."""
    d = scene_dir(cache_dir, when)
    if not d.is_dir():
        return []
    files = sorted(d.glob("HS_*_FLDK_*.DAT.bz2"))
    if bands is not None:
        keep = {b.upper() for b in bands}
        files = [f for f in files if f.name.split("_")[4] in keep]
    return [str(f) for f in files]


def load_scene(
    cache_dir,
    when: datetime,
    composites: Sequence[str],
    pad_data: bool = False,
):
    """Build a Scene for one timeslot with every requested composite loaded.

    `pad_data=False` is the important argument. With the default, satpy pads
    the missing segments with NaN out to the full 5500x5500 (or 22000x22000 for
    B03) disk, so a three-segment regional subset would carry the memory and
    compute cost of the entire hemisphere. Turning it off builds an area
    covering only the segments actually present - which is legal here because
    segments 5-7 are contiguous.
    """
    from satpy import Scene

    needed: set[str] = set()
    for name in composites:
        needed.update(COMPOSITE_REQUIREMENTS.get(name, []))

    files = scene_files(cache_dir, when, needed or None)
    if not files:
        raise FileNotFoundError(f"no cached HSD for {when:%Y-%m-%d %H:%M} UTC")

    # Fail here, loudly, rather than deep inside the resampler.
    #
    # If a band a composite needs is not among the files, satpy logs
    # "Required file type 'hsd_bNN' not found" and quietly returns an empty
    # Scene. The next thing to touch it is `crop`, which then dies with
    # "No dataset areas available" and says nothing about the real cause.
    supplied = {Path(f).name.split("_")[4] for f in files}
    absent = sorted(needed - supplied)
    if absent:
        raise FileNotFoundError(
            f"{when:%Y-%m-%d %H:%M} UTC is missing band(s) {' '.join(absent)}, "
            f"needed by {', '.join(composites)}.\n"
            f"On disk for this timeslot: {' '.join(sorted(supplied))}.\n"
            "Re-run notebook 01 to fetch them, or check that config.yml's "
            "product list matches what was downloaded."
        )

    scn = Scene(filenames=files, reader="ahi_hsd", reader_kwargs={"mask_space": True})
    with warnings.catch_warnings():
        # Composites that blend day and night warn about all-night or all-day
        # scenes; that is expected over a three-day period, not a problem.
        warnings.filterwarnings("ignore", category=UserWarning)
        scn.load(list(composites), pad_data=pad_data)

    # Deliberately no check that `composite in scn` here: a composite mixing
    # resolutions - true_color combines 0.5 km B03 with 1 km B01/B02/B04 - is
    # only generated once `resample_scene` has put everything on one grid, so
    # its absence at this point is normal rather than a fault.
    return scn


def view_zenith(lat: float, lon: float, subsat_lon: float = SUBSAT_LON) -> float:
    """Angle between the satellite direction and the local vertical, degrees.

    Zero at the sub-satellite point, about 81 degrees at the limb.
    """
    R_e, R_sat = 6371.0, 42164.0
    la, dlo = math.radians(lat), math.radians(lon - subsat_lon)
    psi = math.acos(max(-1.0, min(1.0, math.cos(la) * math.cos(dlo))))
    num = R_sat * math.sin(psi)
    den = math.sqrt(R_sat**2 + R_e**2 - 2 * R_sat * R_e * math.cos(psi))
    return math.degrees(math.asin(max(-1.0, min(1.0, num / den))))


def search_radius(grid: Grid, nadir_m: float = 2000.0,
                  subsat_lon: float = SUBSAT_LON) -> float:
    """Nearest-neighbour search radius for a target grid, in metres.

    A fixed radius does not work across the disk. A geostationary pixel is
    `nadir_m` across beneath the satellite and grows towards the limb roughly
    as 1/cos(view zenith), so a radius tuned near nadir silently starves a
    region seen obliquely: at Hunga Tonga, 55 to 68 degrees off vertical, a
    5 km radius left 93% of the target grid with no source neighbour at all
    and produced a nearly black frame.

    Sized at twice the local pixel footprint, evaluated at the most oblique
    corner of the grid, and clamped so a near-nadir grid does not smear.
    """
    lon_min, lat_min, lon_max, lat_max = grid.extent
    corners = [(lat_min, lon_min), (lat_min, lon_max),
               (lat_max, lon_min), (lat_max, lon_max)]
    worst = max(view_zenith(la, lo, subsat_lon) for la, lo in corners)
    stretch = 1.0 / max(math.cos(math.radians(worst)), 0.15)
    return float(min(max(2.0 * nadir_m * stretch, 5000.0), 50000.0))


def touches_antimeridian(grid: Grid, margin: float = 0.0) -> bool:
    """True if the grid, widened by `margin`, reaches or crosses 180 degrees.

    A lon/lat bounding box cannot describe such a region: widening
    (-180, -165) by half a degree gives a western edge of -180.5, which is not
    a longitude. Anything taking a `ll_bbox` has to be skipped for these grids.
    """
    lon_min, _, lon_max, _ = grid.extent
    return (lon_min > lon_max                      # explicitly wraps
            or lon_min - margin <= -180.0
            or lon_max + margin >= 180.0)


def resample_scene(
    scn,
    grid: Grid,
    resampler: str = "nearest",
    cache_dir=None,
    crop: bool = True,
    crop_margin: float = 0.5,
):
    """Resample onto the target grid, caching the neighbour lookup.

    Two things make this cheap enough to run a few hundred times.

    `cache_dir`: the source geometry (Himawari-9's fixed grid) and the target
    geometry never change across a run, so the KD-tree result is identical for
    every frame. Caching turns it from a per-frame cost into a one-off. The
    cache key covers both geometries, so the 2 km IR path and the 0.5 km true
    colour path get their own entries automatically.

    `crop`: segments 5-7 span the full width of the disk - 22000 columns at
    0.5 km - while the target box is a fraction of that. Cropping to the box
    first means the composite arithmetic, which is lazy up to this point, only
    ever runs on pixels that survive into the output. Measured at 157s -> 91s
    for a true colour frame on the hires grid, pixel-for-pixel identical.

    The crop is skipped automatically for grids touching the antimeridian.
    It takes a lon/lat bounding box, and no such box describes a region
    spanning 180 degrees; asking for one silently returns a wedge. Over Hunga
    Tonga that left 7% of the target grid with data. Cropping is only ever an
    optimisation, so dropping it there costs time and nothing else.
    """
    if crop and not touches_antimeridian(grid, crop_margin):
        lon_min, lat_min, lon_max, lat_max = grid.extent
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scn = scn.crop(ll_bbox=(
                lon_min - crop_margin, lat_min - crop_margin,
                lon_max + crop_margin, lat_max + crop_margin,
            ))

    kwargs: dict = {}
    if resampler == "nearest":
        kwargs["radius_of_influence"] = search_radius(grid)
        # `reduce_data` pre-filters the source with a lon/lat bounding box,
        # which collapses across the antimeridian: a target grid spanning
        # -180 to -165 keeps only the sliver whose source longitudes happen to
        # be negative. Measured over Hunga Tonga: 5.1% of the target grid
        # received data with reduction on, 100% with it off.
        #
        # Turning it off costs nothing here because `crop` above has already
        # reduced the source to the target box, which is what reduction was
        # for.
        kwargs["reduce_data"] = False
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(cache_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return scn.resample(grid.area(), resampler=resampler, **kwargs)


# --- Composite dependency resolution ---------------------------------------
#
# `catalog.COMPOSITE_REQUIREMENTS` is a static table, because the download
# stage needs to know which bands to fetch without importing satpy. Static
# tables drift. These helpers derive the same answer from satpy's own config so
# the table can be checked against the truth, and `verify_requirements()` is
# cheap enough to call at the top of a processing run.

_BAND_NAME = re.compile(r"^B\d{2}$")


def _ahi_wavelength_bands() -> list[tuple[float, float, float, str]]:
    """(min, central, max, band name) for every AHI band, from the reader config.

    Needed because satpy states some composite prerequisites by wavelength
    rather than by name - `hybrid_green` asks for 0.51 um, not for B02.
    """
    from satpy.readers.core.config import configs_for_reader

    out: list[tuple[float, float, float, str]] = []
    seen: set[str] = set()
    for config_files in configs_for_reader("ahi_hsd"):
        for path in config_files:
            with open(path, encoding="utf-8") as fh:
                doc = yaml.load(fh, Loader=_UnsafeIgnoringLoader)
            for name, info in (doc.get("datasets") or {}).items():
                wl = info.get("wavelength")
                if (_BAND_NAME.match(name) and name not in seen
                        and isinstance(wl, (list, tuple)) and len(wl) >= 3):
                    seen.add(name)
                    out.append((float(wl[0]), float(wl[1]), float(wl[2]), name))
    return sorted(out, key=lambda t: t[1])


def _band_for_wavelength(value: float,
                         bands: list[tuple[float, float, float, str]]) -> str | None:
    """Resolve a requested wavelength to a band the way satpy does.

    Matching has to be by the band's [min, max] range, not by equality with its
    central wavelength. The composite definitions and the reader disagree on
    the exact numbers - `natural_color` asks for 0.85 and 1.63 um where the
    reader lists centres of 0.86 and 1.6 - and an equality test silently drops
    those prerequisites instead of failing, which is precisely the kind of
    quiet wrong answer this whole module exists to avoid.
    """
    for lo, centre, hi, name in bands:
        if lo <= value <= hi:
            return name
    if not bands:
        return None
    lo, centre, hi, name = min(bands, key=lambda t: abs(t[1] - value))
    return name if abs(centre - value) <= 0.1 else None


class _UnsafeIgnoringLoader(yaml.SafeLoader):
    """SafeLoader that tolerates satpy's `!!python/name:` tags.

    The reader YAML embeds Python object references for file handlers. We only
    want the wavelength numbers, so map any unknown tag to None rather than
    executing anything.
    """


_UnsafeIgnoringLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: None
)
_UnsafeIgnoringLoader.add_multi_constructor(
    "!", lambda loader, suffix, node: None
)


def required_bands(composite: str, sensor: str = "ahi") -> list[str]:
    """Bands a composite needs, resolved recursively from satpy's own config.

    Satpy states prerequisites in three different shapes and all three occur in
    the AHI set, which is exactly why hand-transcribing them is a bad idea:

    * `DataQuery(name='B03', modifiers=...)` - a named band
    * `DataQuery(wavelength=0.51, ...)`      - a band identified only by
      wavelength, e.g. inside `hybrid_green`
    * `'true_color_reproduction_corr'`       - a bare string naming another
      composite

    Nested compositors also appear under generated names such as
    `_night_microphysics_dep_0`, so resolution has to recurse through them.
    """
    from satpy.composites.config_loader import load_compositor_configs_for_sensors

    comps, _ = load_compositor_configs_for_sensors([sensor])
    by_name: dict[str, object] = {}
    for key, value in comps[sensor].items():
        name = key.get("name")
        if name is not None:
            by_name.setdefault(str(name), value)

    wl_bands = _ahi_wavelength_bands() if sensor == "ahi" else []

    def walk(name: str, seen: frozenset[str]) -> set[str]:
        if _BAND_NAME.match(name):
            return {name}
        if name in seen or name not in by_name:
            return set()
        seen = seen | {name}
        found: set[str] = set()
        obj = by_name[name]
        for group in ("prerequisites", "optional_prerequisites"):
            for pre in obj.attrs.get(group, []) or []:
                if isinstance(pre, str):
                    found |= walk(pre, seen)
                    continue
                pname = pre.get("name") if hasattr(pre, "get") else None
                if pname is not None:
                    found |= walk(str(pname), seen)
                    continue
                wl = pre.get("wavelength") if hasattr(pre, "get") else None
                if wl is not None:
                    band = _band_for_wavelength(float(wl), wl_bands)
                    if band:
                        found.add(band)
        return found

    return sorted(walk(composite, frozenset()))


def verify_requirements(composites: Sequence[str] | None = None,
                        sensor: str = "ahi") -> dict[str, tuple[list[str], list[str]]]:
    """Cross-check the static band table against satpy. Returns mismatches only.

    Empty result means the table is correct. A non-empty one maps composite ->
    (what satpy needs, what the table claims).
    """
    names = list(composites) if composites else list(COMPOSITE_REQUIREMENTS)
    bad: dict[str, tuple[list[str], list[str]]] = {}
    for name in names:
        truth = required_bands(name, sensor)
        claimed = sorted(COMPOSITE_REQUIREMENTS.get(name, []))
        if truth and truth != claimed:
            bad[name] = (truth, claimed)
    return bad


# --- Day / night gating ----------------------------------------------------

VIS_BANDS = frozenset({"B01", "B02", "B03", "B04", "B05", "B06"})
"""Reflective bands: useless in darkness, and the expensive ones to read."""


DAY_NIGHT_BLENDS = frozenset({
    "true_color_night_ir",
    "true_color_night_ash",
    "true_color_night_microphysics",
    "true_color_with_night_ir",
    "true_color_with_night_ir_hires",
    "true_color_reproduction",
    "geo_color",
})
"""Composites that fade to an infrared product across the terminator.

These list reflective bands among their requirements, but that does not make
them daylight-only: the whole point is that they keep working after dark. They
must be exempt from the dark-skip gate or the night half of the animation -
the half they exist for - is never rendered.
"""


def needs_sunlight(composite: str) -> bool:
    """True if a composite produces nothing useful over a fully dark scene."""
    if composite in DAY_NIGHT_BLENDS:
        return False
    return bool(VIS_BANDS & set(COMPOSITE_REQUIREMENTS.get(composite, [])))


def is_daylight(when: datetime, grid: Grid, max_zenith: float = 88.0) -> bool:
    """True if the sun is up anywhere in the grid.

    Rendering true colour over a fully dark box costs a minute and produces a
    black rectangle. Over three days that is roughly half the timeslots, so
    this gate is worth more than any other single optimisation here.

    Checks the four corners and the centre; a box this size cannot have
    daylight strictly inside it while all five points are dark.
    """
    from pyorbital.astronomy import sun_zenith_angle

    lon_min, lat_min, lon_max, lat_max = grid.extent
    probes = [
        (lon_min, lat_min), (lon_min, lat_max),
        (lon_max, lat_min), (lon_max, lat_max),
        ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2),
    ]
    naive = when.replace(tzinfo=None) if when.tzinfo else when
    return any(
        float(sun_zenith_angle(naive, lon, lat)) < max_zenith for lon, lat in probes
    )


def group_composites(composites: Sequence[str]) -> list[list[str]]:
    """Split composites into Scenes that share a band cost.

    Anything needing the reflective bands drags in B03, whose three segments
    decompress to roughly 290 MB and dominate the read. Keeping the IR-only
    products in their own Scene means they never pay for it: measured at 15s
    for four IR composites versus 65s for true colour alone.
    """
    vis = [c for c in composites if needs_sunlight(c)]
    ir = [c for c in composites if not needs_sunlight(c)]
    return [g for g in (ir, vis) if g]


def observation_window(scn, grid: "Grid | None" = None):
    """When the pixels in this scene were actually observed.

    The time in an HSD filename is not the time the imagery was taken. JMA
    defines that field as the "Observation start time (timeline)" (Himawari
    Standard Data User's Guide v1.3, Table 4) - the start of the 10-minute
    schedule slot. AHI then sweeps the disk from north to south across that
    slot, and every segment carries its own observation time in header block
    #9 of its file.

    Measured on this archive, segments 5-7 are observed 245 s, 305 s and 364 s
    after the slot label, with a spread under one second across three days. A
    frame stamped with its filename time is therefore four to six minutes
    early, and additionally spans about 200 s of real time between its top and
    bottom edges.

    Note this cannot come from `scn[composite].attrs`: satpy reports the
    nominal slot there (03:00:00 to 03:10:00 for the 0300 file), not the scan.
    The real times live on the reader's file handlers.

    Returns (stamp, span_start, span_end). `stamp` is the observation start of
    the segment containing the centre of `grid`, which is the most defensible
    single time to print on a regional frame. All three are None if the times
    cannot be recovered.
    """
    rows: list[tuple[int, datetime, datetime]] = []
    try:
        for reader in scn._readers.values():
            for handlers in getattr(reader, "file_handlers", {}).values():
                for fh in handlers:
                    seg = fh.filename_info.get("segment")
                    t0 = getattr(fh, "observation_start_time", None)
                    t1 = getattr(fh, "observation_end_time", None)
                    if seg is not None and t0 is not None:
                        rows.append((int(seg), t0, t1 or t0))
                # Every band is swept by the same mirror on the same schedule,
                # so one file type is enough.
                break
            break
    except Exception:  # noqa: BLE001 - internal satpy layout; never fatal
        return None, None, None

    if not rows:
        return None, None, None

    rows.sort()
    span_start, span_end = rows[0][1], max(r[2] for r in rows)

    stamp = span_start
    if grid is not None:
        lon_min, lat_min, lon_max, lat_max = grid.extent
        line, _ = geos_line_col((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
        if not math.isnan(line):
            want = segment_of_line(line)
            for seg, t0, _ in rows:
                if seg == want:
                    stamp = t0
                    break
    return stamp, span_start, span_end


def to_image(scn, composite: str) -> "np.ndarray":
    """Enhanced composite as an (H, W, 3|4) uint8 array.

    This is the same enhancement pipeline satpy's writers use, pulled out so
    the array can be annotated before it hits disk instead of being written,
    reopened and rewritten.
    """
    from satpy.writers.core.image import get_enhanced_image

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        img = get_enhanced_image(scn[composite])
        data = img.finalize(fill_value=0, dtype=np.uint8)[0]
    arr = np.asarray(data.transpose("y", "x", "bands"))
    return arr


# --- Coastlines ------------------------------------------------------------

def coastline_overlay(
    grid: Grid,
    out_path,
    linewidth: int = 1,
    colour: tuple[int, int, int, int] = (255, 210, 77, 217),
    graticule: bool = True,
    graticule_step: float = 5.0,
    resolution: str = "10m",
    force: bool = False,
) -> Path | None:
    """Render coastlines and a graticule once, as a transparent PNG.

    Drawn directly into the grid's own pixel space with PIL rather than through
    matplotlib. Two reasons:

    * Registration is exact by construction. The output grid is plate carree,
      so mapping a (lon, lat) vertex to a pixel is one linear scale per axis -
      no figure DPI, axes margins or bounding-box rounding to get wrong.
    * It sidesteps matplotlib entirely. On this machine matplotlib 3.11.1 from
      conda-forge dies with a delay-loaded DLL fault (0xc06d007f) inside the
      Agg renderer on any savefig, which would otherwise take the whole
      pipeline down at its very last step.

    Cartopy is still used, but only as a data source: `natural_earth()` just
    downloads the shapefile and returns a path, which does not touch the
    broken rendering path.

    Returns None if the shapefile cannot be fetched (first use needs network);
    frames then render without coastlines rather than failing.
    """
    out_path = Path(out_path)
    if out_path.exists() and not force:
        return out_path

    try:
        import cartopy.io.shapereader as shpreader
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    try:
        shp = shpreader.natural_earth(
            resolution=resolution, category="physical", name="coastline"
        )
        geoms = list(shpreader.Reader(shp).geometries())
    except Exception:  # noqa: BLE001 - offline, or the archive moved
        return None

    h, w = grid.shape
    lon_min, lat_min, lon_max, lat_max = grid.extent
    sx = w / (lon_max - lon_min)
    sy = h / (lat_max - lat_min)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def project(coords):
        return [((x - lon_min) * sx, (lat_max - y) * sy) for x, y in coords]

    for geom in geoms:
        # Cheap reject on the bounding box: 4133 coastline strings worldwide,
        # a couple of dozen of which touch this box.
        gx0, gy0, gx1, gy1 = geom.bounds
        if gx1 < lon_min or gx0 > lon_max or gy1 < lat_min or gy0 > lat_max:
            continue
        parts = getattr(geom, "geoms", [geom])
        for part in parts:
            pts = project(part.coords)
            if len(pts) > 1:
                draw.line(pts, fill=colour, width=linewidth, joint="curve")

    if graticule:
        faint = (255, 255, 255, 56)
        lon = _ceil_to(lon_min, graticule_step)
        while lon <= lon_max:
            x = (lon - lon_min) * sx
            _dashed_line(draw, [(x, 0), (x, h)], faint)
            lon += graticule_step
        lat = _ceil_to(lat_min, graticule_step)
        while lat <= lat_max:
            y = (lat_max - lat) * sy
            _dashed_line(draw, [(0, y), (w, y)], faint)
            lat += graticule_step

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _ceil_to(value: float, step: float) -> float:
    import math

    return math.ceil(value / step) * step


def _dashed_line(draw, ends, fill, dash: int = 6, gap: int = 8) -> None:
    (x0, y0), (x1, y1) = ends
    length = max(abs(x1 - x0), abs(y1 - y0))
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
            fill=fill, width=1,
        )
        pos = end + gap


# --- Annotation ------------------------------------------------------------

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _lonlat_to_px(grid: Grid, lon: float, lat: float) -> tuple[float, float]:
    lon_min, lat_min, lon_max, lat_max = grid.extent
    h, w = grid.shape
    return (
        (lon - lon_min) / (lon_max - lon_min) * w,
        (lat_max - lat) / (lat_max - lat_min) * h,
    )


COMPOSITE_LABELS = {
    "true_color": "True colour",
    "true_color_reproduction": "True Colour Reproduction (JMA)",
    "ash": "Ash RGB",
    "dust": "Dust RGB",
    "night_microphysics": "Night microphysics RGB",
    "volcanic_emissions": "Volcanic emissions (SO2) RGB",
    "so2": "Volcanic emissions (SO2) RGB",
    "natural_color": "Natural colour",
    "B13": "Band 13, 10.4 um clean IR window",
}


def annotate(
    arr: "np.ndarray",
    when: datetime,
    composite: str,
    grid: Grid,
    overlay_path: Path | None = None,
    marker: tuple[float, float, str] | None = None,
    observed: datetime | None = None,
    tz=None,
    tz_label: str = "UTC",
    credit: str = "Himawari AHI  |  JMA / JAXA  |  via NOAA AWS Open Data",
):
    """Paste the coastline overlay and burn in the labels. Returns a PIL Image.

    `when` is the filename's timeline slot; `observed` is when the pixels were
    actually scanned (see `observation_window`). When `observed` is given it is
    what the big clock shows, because that is the honest label - the two differ
    by four to six minutes for this region.
    """
    from PIL import Image, ImageDraw

    # Single-band products (B13, B08) come back as (H, W, 1), which PIL cannot
    # interpret; RGB and RGBA composites come back as (H, W, 3|4). Squeeze the
    # single-band case to 2-D greyscale and drop any alpha from the rest.
    if arr.ndim == 3:
        arr = arr[:, :, 0] if arr.shape[2] == 1 else arr[:, :, :3]
    img = Image.fromarray(arr).convert("RGBA")

    if overlay_path is not None and Path(overlay_path).exists():
        overlay = Image.open(overlay_path).convert("RGBA")
        if overlay.size == img.size:
            img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    w, h = img.size
    scale = max(1.0, w / 1250.0)
    big, small = _font(int(26 * scale)), _font(int(17 * scale))

    if marker:
        lon, lat, name = marker
        x, y = _lonlat_to_px(grid, lon, lat)
        if 0 <= x < w and 0 <= y < h:
            r = 5 * scale
            draw.ellipse([x - r, y - r, x + r, y + r], outline="#ff3b30",
                         width=max(2, int(2 * scale)))
            if name:
                _text(draw, (x + r + 4 * scale, y - r), name, small)

    # The headline clock is the timeline slot: the round 10-minute grid a
    # viewer expects, and the one an animation steps through evenly.
    #
    # The scan time goes on the second line rather than the first. It is the
    # honest answer to "when were these pixels taken" - JMA's timeline start
    # plus the sweep delay down to this segment, about five minutes for the
    # Sunda Strait - but leading with it makes every frame read :05 or :15,
    # which looks like a mistake against a 10-minute instrument, and the
    # trailing seconds are precision no reader needs.
    if tz is not None:
        base = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        local = base.astimezone(tz)
    else:
        local, tz_label = when, "UTC"
    _text(draw, (16 * scale, 14 * scale),
          local.strftime(f"%d %b %Y  %H:%M {tz_label}"), big)
    second = f"{when:%H:%M} UTC   {COMPOSITE_LABELS.get(composite, composite)}"
    if observed is not None:
        second += f"   (scanned {observed:%H:%M:%S})"
    _text(draw, (16 * scale, 14 * scale + big.size + 6 * scale), second, small)
    _text(draw, (16 * scale, h - small.size - 16 * scale), credit, small,
          fill="#dddddd")
    return img


def _text(draw, xy, text, font, fill="white") -> None:
    """White text with a dark outline, legible over both cloud and ocean."""
    x, y = xy
    draw.text((x, y), text, font=font, fill="black",
              stroke_width=max(2, font.size // 10), stroke_fill="black")
    draw.text((x, y), text, font=font, fill=fill)


# --- Driver ----------------------------------------------------------------

def frame_path(frames_dir, composite: str, grid: Grid, when: datetime) -> Path:
    return (
        Path(frames_dir) / f"{composite}_{grid.name}" /
        f"{composite}_{grid.name}_{when:%Y%m%d_%H%M}.png"
    )


@contextlib.contextmanager
def scene_scratch(base):
    """Give satpy a private directory for this timeslot's bz2 expansion.

    Satpy decompresses every .bz2 segment to a temporary file and registers
    `weakref.finalize(self, self._cleanup)` to delete it. Finalisation happens
    whenever the garbage collector gets round to it, so on Windows the delete
    frequently races a handle dask still holds open and raises

        PermissionError [WinError 32] The process cannot access the file
        because it is being used by another process

    That exception propagates out of the render and, worse, leaves the
    decompressed file behind. A decompressed 0.5 km segment is 92 MB, so a run
    that keeps failing fills the disk and then fails for that reason instead:
    one three-day pass leaked 10,432 files totalling 230 GB into the system temp directory and
    lost 202 of 427 timeslots.

    This bounds the damage to a single timeslot. Each gets its own directory,
    an explicit `gc.collect()` runs the finalisers at a point where we still
    control the directory, and whatever they fail to remove goes with the tree.
    """
    import satpy

    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    d = root / f"hsd_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    previous = satpy.config.get("tmp_dir")
    satpy.config.set(tmp_dir=str(d))
    try:
        yield d
    finally:
        satpy.config.set(tmp_dir=previous)
        # Drop our references, then force finalisers so _cleanup runs now
        # rather than at some unpredictable later point.
        gc.collect()
        shutil.rmtree(d, ignore_errors=True)


def purge_scratch(base, keep_pids=()):
    """Remove scratch directories left by earlier runs. Returns bytes freed."""
    root = Path(base)
    if not root.is_dir():
        return 0
    freed = 0
    for d in root.glob("hsd_*"):
        if not d.is_dir():
            continue
        try:
            pid = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            pid = None
        if pid in keep_pids:
            continue
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    return freed


def render_timeslot(
    when: datetime,
    cache_dir,
    frames_dir,
    composites: Sequence[str],
    grid: Grid,
    overlay_path: Path | None = None,
    resample_cache=None,
    overwrite: bool = False,
    skip_dark: bool = True,
    marker: tuple[float, float, str] | None = None,
    tz=None,
    tz_label: str = "UTC",
) -> dict[str, Path | None]:
    """Render the requested composites for one timeslot.

    Composites are grouped so that each Scene is built once and serves every
    product that shares its band cost. That matters because satpy decompresses
    each .bz2 member to a temporary file on every Scene construction, and B03
    alone expands to roughly 290 MB across three segments.

    `skip_dark` drops reflective-band composites over a night-time box, where
    they would cost a minute each to produce a black rectangle. The returned
    dict maps those to None, which `render_many` and the animation stage both
    read as "no frame here", not as a failure.
    """
    out: dict[str, Path | None] = {}
    todo: list[str] = []
    for c in composites:
        path = frame_path(frames_dir, c, grid, when)
        if path.exists() and not overwrite:
            out[c] = path
        elif skip_dark and needs_sunlight(c) and not is_daylight(when, grid):
            out[c] = None
        else:
            todo.append(c)
    if not todo:
        return out

    for group in group_composites(todo):
        with scene_scratch(Path(cache_dir) / "_scratch"):
            scn = load_scene(cache_dir, when, group)
            # Read the scan times from the un-resampled Scene: resampling does
            # not carry the reader's file handlers through.
            observed, _span0, _span1 = observation_window(scn, grid)
            local = resample_scene(scn, grid, cache_dir=resample_cache)
            for c in group:
                try:
                    arr = to_image(local, c)
                except KeyError:
                    out[c] = None
                    continue
                img = annotate(arr, when, c, grid, overlay_path, marker=marker,
                               observed=observed, tz=tz, tz_label=tz_label)
                path = frame_path(frames_dir, c, grid, when)
                path.parent.mkdir(parents=True, exist_ok=True)
                # No optimize=True: it roughly doubles save time on a
                # 5000x2800 frame for a few percent of size, and there are
                # hundreds of them.
                img.convert("RGB").save(path)
                out[c] = path
            # Drop references before the context manager forces finalisers.
            del local, scn
    return out


def _worker_init(dask_threads: int, scratch_root=None) -> None:
    """Per-process setup for parallel rendering.

    Each spawned process is a fresh interpreter, so it needs the local satpy
    config registered again, and its own dask thread budget capped - otherwise
    N processes each start a pool sized to all 80 cores and spend their time
    fighting each other for cache lines.
    """
    import warnings as _w

    import dask

    _w.filterwarnings("ignore")
    dask.config.set(num_workers=max(1, dask_threads))
    dask.config.set(scheduler="threads")
    use_local_config()
    # Clear scratch left by a previous run that died before its context
    # managers unwound. Own-process directories are never touched, so this is
    # safe with several workers starting at once.
    if scratch_root is not None:
        purge_scratch(scratch_root, keep_pids={os.getpid()})


def _render_one(args):
    (when, cache_dir, frames_dir, composites, grid, overlay, rcache,
     overwrite, skip_dark, marker, tz, tz_label) = args
    try:
        return when, render_timeslot(
            when, cache_dir, frames_dir, composites, grid,
            overlay_path=overlay, resample_cache=rcache,
            overwrite=overwrite, skip_dark=skip_dark, marker=marker,
            tz=tz, tz_label=tz_label,
        ), None
    except Exception as exc:  # noqa: BLE001 - one bad scene must not stop 400
        return when, {}, f"{type(exc).__name__}: {exc}"


def render_many(
    times: Iterable[datetime],
    cache_dir,
    frames_dir,
    composites: Sequence[str],
    grid: Grid,
    overlay_path: Path | None = None,
    resample_cache=None,
    overwrite: bool = False,
    skip_dark: bool = True,
    marker: tuple[float, float, str] | None = None,
    tz=None,
    tz_label: str = "UTC",
    workers: int = 1,
    dask_threads: int | None = None,
    progress: bool = True,
) -> dict[datetime, dict[str, Path | None]]:
    """Render a whole series, skipping timeslots that already have frames.

    Safe to interrupt and re-run: a timeslot whose PNGs all exist is skipped.

    `workers` renders that many timeslots in parallel processes. Threads are no
    use here - the work is dask-bound inside each frame and would just contend
    - but separate processes scale nearly linearly. Watch memory rather than
    cores when choosing a number: a 0.5 km true-colour scene is the high-water
    mark, so on a machine with limited RAM keep this low even if it has plenty
    of cores.

    Populate the resample cache first with a single-worker call on one
    timeslot. Started cold, every worker computes the same KD-tree at once and
    they race to write the same cache file.

    On Windows, `workers > 1` uses ProcessPoolExecutor, which spawns fresh
    interpreters that re-import the calling module. A plain script must
    therefore put its work behind `if __name__ == "__main__":` or every worker
    re-runs the script and the pool dies with BrokenProcessPool. Notebooks are
    unaffected, because a Jupyter kernel's `__main__` has no module file to
    re-import.
    """
    times = list(times)
    if not times:
        return {}

    if dask_threads is None:
        dask_threads = max(1, (os.cpu_count() or 8) // max(1, workers))

    payload = [
        (t, str(cache_dir), str(frames_dir), list(composites), grid,
         overlay_path, resample_cache, overwrite, skip_dark, marker,
         tz, tz_label)
        for t in times
    ]

    results: dict[datetime, dict[str, Path | None]] = {}
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(times), unit="slot", desc="render")
        except ImportError:
            pass

    def _record(when, res, err) -> None:
        results[when] = res
        if err:
            msg = f"{when:%Y-%m-%d %H:%M}  {err}"
            bar.write(msg) if bar is not None else print(msg)
        if bar is not None:
            bar.update(1)

    if workers <= 1:
        _worker_init(dask_threads, Path(cache_dir) / "_scratch")
        for args in payload:
            _record(*_render_one(args))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(dask_threads, Path(cache_dir) / "_scratch")
        ) as pool:
            for when, res, err in pool.map(_render_one, payload, chunksize=1):
                _record(when, res, err)

    if bar is not None:
        bar.close()
    return results


def rendered_frames(frames_dir, composite: str, grid: Grid) -> list[Path]:
    """Every frame for one composite/grid, in chronological order.

    Filenames end in ...YYYYmmdd_HHMM.png, so a lexical sort is chronological.
    """
    d = Path(frames_dir) / f"{composite}_{grid.name}"
    return sorted(d.glob(f"{composite}_{grid.name}_*.png")) if d.is_dir() else []


# Labels for the products added for the day/night work.
COMPOSITE_LABELS.update({
    "true_color_with_night_ir": "True colour by day, IR by night",
    "true_color_with_night_ir_hires": "True colour by day, IR by night",
    "geo_color": "GeoColor: true colour by day, IR and city lights by night",
    "night_ir_with_background": "Night IR with city lights",
    "night_ir_alpha": "Night IR",
    "B08": "Band 8, 6.2 um upper water vapour",
})
