"""Naming, geometry and time bookkeeping for Himawari Standard Data (HSD).

A single AHI full-disk observation is distributed as one file per
(band x segment). Filenames look like:

    HS_H09_20260905_0600_B13_FLDK_R20_S0610.DAT.bz2
       |    |        |    |   |    |   |  |
       |    |        |    |   |    |   |  +- 10 segments in the scan
       |    |        |    |   |    |   +---- this is segment 6
       |    |        |    |   |    +-------- 2.0 km resolution
       |    |        |    |   +------------- full disk
       |    |        |    +----------------- band 13 (10.4 um)
       |    |        +---------------------- observation time, UTC
       |    +------------------------------- date, UTC
       +------------------------------------ Himawari-9

The disk is cut into 10 horizontal strips of equal line count, so a regional
study only needs the few segments its latitudes fall into. `segments_for_extent`
works those out from the geostationary projection geometry rather than from a
hardcoded table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

SATELLITE = "H09"
"""Himawari-9 has been the operational satellite since December 2022."""

SUBSAT_LON = 140.7
"""Sub-satellite longitude of Himawari-9, degrees east."""

N_SEGMENTS = 10
"""Full-disk scenes are always split into 10 segments, at every resolution."""

RESOLUTION_CODE = {
    "B01": "R10",  # 0.47 um  blue           1.0 km
    "B02": "R10",  # 0.51 um  green          1.0 km
    "B03": "R05",  # 0.64 um  red            0.5 km
    "B04": "R10",  # 0.86 um  veggie         1.0 km
    "B05": "R20",  # 1.6 um   snow / ice     2.0 km
    "B06": "R20",  # 2.3 um   cloud phase
    "B07": "R20",  # 3.9 um   shortwave IR, fire and night cloud
    "B08": "R20",  # 6.2 um   upper water vapour
    "B09": "R20",  # 6.9 um   mid water vapour
    "B10": "R20",  # 7.3 um   lower water vapour, SO2 sensitive
    "B11": "R20",  # 8.6 um   SO2 / cloud phase
    "B12": "R20",  # 9.6 um   ozone, SO2 sensitive
    "B13": "R20",  # 10.4 um  clean IR window
    "B14": "R20",  # 11.2 um  IR window
    "B15": "R20",  # 12.4 um  dirty IR window
    "B16": "R20",  # 13.3 um  CO2
}

PIXELS_PER_SEGMENT = {"R05": 2200, "R10": 1100, "R20": 550}
"""Lines per segment at each resolution code (0.5 / 1.0 / 2.0 km)."""

HOUSEKEEPING_SLOTS = frozenset({"0240", "1440"})
"""AHI pauses full-disk scanning twice a day for housekeeping.

Verified against the AWS listing for 2026-09-04 and 2026-09-05: both days carry
142 of the 144 nominal 10-minute slots, and the two absentees are exactly 02:40
and 14:40 UTC. Skipping them up front avoids hundreds of pointless 404s over a
three-day download.
"""

# --- Band sets -------------------------------------------------------------
#
# Sizes are measured, not estimated: they come from summing the S3
# Content-Length of every object under AHI-L1b-FLDK/2026/09/05/0600/ for
# segments 5-7.

COMPOSITE_REQUIREMENTS: dict[str, list[str]] = {
    "true_color": ["B01", "B02", "B03", "B04"],
    "true_color_reproduction": ["B01", "B02", "B03", "B04"],
    "natural_color": ["B03", "B04", "B05"],
    "ash": ["B11", "B13", "B14", "B15"],
    "dust": ["B11", "B13", "B14", "B15"],
    "night_microphysics": ["B07", "B13", "B14", "B15"],
    "cloudtop": ["B07", "B14", "B15"],
    # Added locally in config/satpy - satpy ships SO2 for ABI/FCI but not AHI.
    "volcanic_emissions": ["B09", "B10", "B11", "B13"],

    # Day/night blends. A DayNightCompositor fades between a daylight product
    # and a night-time one across the terminator using solar zenith angle, so
    # a single animation runs continuously through sunset instead of going
    # black. All of these need only bands the ash and true-colour sets already
    # bring in, so adding them costs render time and no download.
    "true_color_with_night_ir": ["B01", "B02", "B03", "B04", "B07", "B13", "B15"],
    "true_color_with_night_ir_hires": ["B01", "B02", "B03", "B04", "B07", "B13", "B15"],
    "geo_color": ["B01", "B02", "B03", "B04", "B07", "B13"],
    "night_ir_with_background": ["B07", "B13", "B15"],
    "night_ir_alpha": ["B07", "B13", "B15"],

    # Local blends defined in config/satpy, avoiding satpy's Black Marble
    # download. Day half is true colour in all three; the night half differs.
    "true_color_night_ir": ["B01", "B02", "B03", "B04", "B13"],
    "true_color_night_ash": ["B01", "B02", "B03", "B04", "B11", "B13", "B14", "B15"],
    "true_color_night_microphysics":
        ["B01", "B02", "B03", "B04", "B07", "B13", "B14", "B15"],

    # Single bands, rendered as greyscale.
    "B13": ["B13"],   # 10.4 um clean IR window: the workhorse, day and night
    "B08": ["B08"],   # 6.2 um upper water vapour
}
"""Bands each satpy composite needs.

Used to decide what to download and which cached files to hand a Scene, so it
has to live here, in a module that does not import satpy.

**This table is generated, not transcribed.** Do not edit it by hand - run
`himawari.render.verify_requirements()`, which resolves the same answer out of
satpy's own composite config and reports any drift. Two entries here were
wrong when written by eye: `night_microphysics` is B15-B13, **B14-B07**, B13 -
it needs B14, not just B07/B13/B15 - and `cloudtop` uses B14 rather than B13.

Some entries are less obvious than they look:

* `ash` and `dust` are the *same four channels* (B15-B13, B14-B11, B13) and
  differ only in their enhancement stretch, so dust needs B14 as well.
* `true_color` reaches B02 and B04 only through the nested `hybrid_green`
  compositor, which names them by wavelength rather than by band.
"""


def bands_for(composites: Sequence[str]) -> list[str]:
    """Union of the bands a set of composites needs, sorted."""
    need: set[str] = set()
    for name in composites:
        try:
            need.update(COMPOSITE_REQUIREMENTS[name])
        except KeyError:
            raise KeyError(
                f"unknown composite {name!r}; known: "
                f"{sorted(COMPOSITE_REQUIREMENTS)}"
            ) from None
    return sorted(need)


BAND_SETS: dict[str, list[str]] = {
    # Everything the true colour, ash, dust, SO2 and night composites need.
    # ~192 MB per timeslot for segments 5-7, averaged over the diurnal cycle.
    "full": bands_for([
        "true_color",            # B01-B04, sharpened to 0.5 km
        "ash", "dust",           # B11, B13, B14, B15
        "night_microphysics",    # adds B07
        "volcanic_emissions",    # adds B09, B10
    ]),
    # Volcanic ash RGB only. Works day and night. ~33 MB per timeslot.
    "ash": COMPOSITE_REQUIREMENTS["ash"],
    # Dust RGB. Same channels as ash, different stretch. ~33 MB per timeslot.
    "dust": COMPOSITE_REQUIREMENTS["dust"],
    # Daylight true colour at native 0.5 km. ~138 MB per timeslot.
    "true_color": COMPOSITE_REQUIREMENTS["true_color"],
    # Single clean IR window, for a cheap end-to-end pilot. ~8.3 MB per slot.
    "pilot": ["B13"],
    # Literally every band, if you want to keep options open. ~214 MB per slot.
    "all": [f"B{n:02d}" for n in range(1, 17)],
}


# --- Time ------------------------------------------------------------------

def timeslots(
    start: datetime,
    end: datetime,
    step_minutes: int = 10,
    skip_housekeeping: bool = True,
) -> list[datetime]:
    """Every AHI full-disk observation time in [start, end], UTC, inclusive.

    Naive datetimes are assumed to already be UTC.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if step_minutes % 10:
        raise ValueError("AHI full-disk cadence is 10 minutes; step must be a multiple")

    out: list[datetime] = []
    t = start
    step = timedelta(minutes=step_minutes)
    while t <= end:
        if not (skip_housekeeping and t.strftime("%H%M") in HOUSEKEEPING_SLOTS):
            out.append(t)
        t += step
    return out


WIB = timezone(timedelta(hours=7))
"""Western Indonesian Time, the clock Krakatau runs on."""


def to_wib(t: datetime) -> datetime:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(WIB)


# --- Filenames and remote paths -------------------------------------------

def hsd_filename(t: datetime, band: str, segment: int, satellite: str = SATELLITE) -> str:
    """Canonical HSD filename, identical on AWS and on P-Tree."""
    band = band.upper()
    if band not in RESOLUTION_CODE:
        raise ValueError(f"unknown band {band!r}")
    if not 1 <= segment <= N_SEGMENTS:
        raise ValueError(f"segment must be 1..{N_SEGMENTS}, got {segment}")
    return (
        f"HS_{satellite}_{t:%Y%m%d}_{t:%H%M}_{band}_FLDK_"
        f"{RESOLUTION_CODE[band]}_S{segment:02d}{N_SEGMENTS:02d}.DAT.bz2"
    )


def aws_key(t: datetime, band: str, segment: int, satellite: str = SATELLITE) -> str:
    """Object key in the public noaa-himawari9 bucket."""
    return f"AHI-L1b-FLDK/{t:%Y/%m/%d/%H%M}/{hsd_filename(t, band, segment, satellite)}"


def aws_bucket(satellite: str = SATELLITE) -> str:
    return "noaa-himawari" + satellite[1:].lstrip("0")


def aws_url(t: datetime, band: str, segment: int, satellite: str = SATELLITE) -> str:
    return (
        f"https://{aws_bucket(satellite)}.s3.amazonaws.com/"
        f"{aws_key(t, band, segment, satellite)}"
    )


def ptree_path(t: datetime, band: str, segment: int, satellite: str = SATELLITE) -> str:
    """Absolute path on ftp.ptree.jaxa.jp under the /jma/ tree.

    P-Tree buckets HSD by year-month / day / hour; AWS uses one directory per
    10-minute slot. Same bytes, different shelving.
    """
    return f"/jma/hsd/{t:%Y%m}/{t:%d}/{t:%H}/{hsd_filename(t, band, segment, satellite)}"


def local_path(
    cache_dir, t: datetime, band: str, segment: int, satellite: str = SATELLITE
) -> Path:
    """Where a file lives in the local cache.

    One directory per timeslot, mirroring AWS, because satpy is happiest when a
    whole scene sits in a single folder it can glob.
    """
    return (
        Path(cache_dir)
        / f"{t:%Y%m%d}"
        / f"{t:%H%M}"
        / hsd_filename(t, band, segment, satellite)
    )


def scene_dir(cache_dir, t: datetime) -> Path:
    return Path(cache_dir) / f"{t:%Y%m%d}" / f"{t:%H%M}"


# --- Geometry --------------------------------------------------------------
#
# Forward geostationary projection from the CGMS LRIT/HRIT Global
# Specification, section 4.4.3.2, with the AHI full-disk scaling factors.

_FAC = {"R20": 20466275, "R10": 40932549, "R05": 81865099}
_OFF = {"R20": 2750.5, "R10": 5500.5, "R05": 11000.5}


def geos_line_col(
    lat: float,
    lon: float,
    resolution: str = "R20",
    subsat_lon: float = SUBSAT_LON,
) -> tuple[float, float]:
    """Full-disk (line, column) for a geographic point. NaN if off the disk."""
    fac = _FAC[resolution]
    off = _OFF[resolution]

    lat_r = math.radians(lat)
    dlon = math.radians(lon - subsat_lon)

    c_lat = math.atan(0.993305616 * math.tan(lat_r))
    r_l = 6356.7523 / math.sqrt(1.0 - 0.00669438444 * math.cos(c_lat) ** 2)

    r1 = 42164.0 - r_l * math.cos(c_lat) * math.cos(dlon)
    r2 = -r_l * math.cos(c_lat) * math.sin(dlon)
    r3 = r_l * math.sin(c_lat)

    # Visibility test: positive means the point is behind the limb.
    if r1 * (r1 - 42164.0) + r2**2 + 1.006803 * r3**2 > 0:
        return float("nan"), float("nan")

    rn = math.sqrt(r1**2 + r2**2 + r3**2)
    x = math.degrees(math.atan2(-r2, r1))
    y = math.degrees(math.asin(-r3 / rn))

    return off + y * fac / 65536.0, off + x * fac / 65536.0


def segment_of_line(line: float, resolution: str = "R20") -> int:
    """1-based segment index containing a full-disk line number."""
    per = PIXELS_PER_SEGMENT[resolution]
    return min(N_SEGMENTS, max(1, int(line // per) + 1))


def segments_for_extent(
    extent: Sequence[float],
    resolution: str = "R20",
    pad: int = 0,
    samples: int = 41,
) -> list[int]:
    """Segments covering a (lon_min, lat_min, lon_max, lat_max) box.

    Samples the whole box perimeter, not just the corners: lines of constant
    latitude bow across the disk, so the northernmost scan line of a box is not
    generally at one of its corners.

    `pad` widens the result by N segments on each side, cheap insurance if a
    plume drifts past the box you planned for.
    """
    lon_min, lat_min, lon_max, lat_max = extent
    lons = [lon_min + (lon_max - lon_min) * i / (samples - 1) for i in range(samples)]
    lats = [lat_min + (lat_max - lat_min) * i / (samples - 1) for i in range(samples)]

    lines: list[float] = []
    for lon in lons:
        for lat in (lat_min, lat_max):
            line, _ = geos_line_col(lat, lon, resolution)
            if not math.isnan(line):
                lines.append(line)
    for lat in lats:
        for lon in (lon_min, lon_max):
            line, _ = geos_line_col(lat, lon, resolution)
            if not math.isnan(line):
                lines.append(line)

    if not lines:
        raise ValueError(f"extent {tuple(extent)} does not intersect the Himawari disk")

    lo = segment_of_line(min(lines), resolution) - pad
    hi = segment_of_line(max(lines), resolution) + pad
    return list(range(max(1, lo), min(N_SEGMENTS, hi) + 1))


# --- Work list -------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrameTask:
    """One file to fetch."""

    time: datetime
    band: str
    segment: int
    satellite: str = SATELLITE

    @property
    def filename(self) -> str:
        return hsd_filename(self.time, self.band, self.segment, self.satellite)

    @property
    def url(self) -> str:
        return aws_url(self.time, self.band, self.segment, self.satellite)

    @property
    def key(self) -> str:
        return aws_key(self.time, self.band, self.segment, self.satellite)

    @property
    def ftp_path(self) -> str:
        return ptree_path(self.time, self.band, self.segment, self.satellite)

    def path(self, cache_dir) -> Path:
        return local_path(cache_dir, self.time, self.band, self.segment, self.satellite)


def frame_tasks(
    times: Iterable[datetime],
    bands: Sequence[str],
    segments: Sequence[int],
    satellite: str = SATELLITE,
) -> list[FrameTask]:
    """Cartesian product of times x bands x segments, in download order.

    Ordered time-major so an interrupted run leaves whole usable scenes behind
    rather than one band spread across the entire period.
    """
    return [
        FrameTask(t, b, s, satellite)
        for t in times
        for b in bands
        for s in segments
    ]
