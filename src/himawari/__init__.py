"""Himawari-9 AHI pipeline for the Anak Krakatau eruption, 4-6 September 2026.

Three stages, one per notebook:

  catalog + fetch  ->  notebook/01_download.ipynb
  render           ->  notebook/02_process.ipynb
  animate          ->  notebook/03_animate.ipynb

The heavy lifting lives here so the notebooks stay readable.
"""

__version__ = "0.1.0"

from .catalog import (
    BAND_SETS,
    HOUSEKEEPING_SLOTS,
    RESOLUTION_CODE,
    SATELLITE,
    aws_key,
    aws_url,
    frame_tasks,
    geos_line_col,
    hsd_filename,
    ptree_path,
    segments_for_extent,
    timeslots,
)

__all__ = [
    "BAND_SETS",
    "HOUSEKEEPING_SLOTS",
    "RESOLUTION_CODE",
    "SATELLITE",
    "aws_key",
    "aws_url",
    "frame_tasks",
    "geos_line_col",
    "hsd_filename",
    "ptree_path",
    "segments_for_extent",
    "timeslots",
]
