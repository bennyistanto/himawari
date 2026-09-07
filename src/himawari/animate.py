"""Frames to video.

MP4 is the primary output. A three-day run at 10-minute cadence is roughly 400
frames, and as an H.264 video that is a few tens of megabytes at full 5000 px
width, where the equivalent GIF would be gigabytes and dithered to 256 colours.
GIF is kept for the places that still demand one - chat apps, some CMSs - at a
reduced width.

Frame timing is handled by holding frames rather than by dropping them. Where a
timeslot is missing (AHI housekeeping at 02:40 and 14:40 UTC, or a night gap in
a daylight-only composite), `fill_gaps` repeats the previous frame so the
animation keeps a constant real-time-per-second rate instead of silently
jumping.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

_STAMP = re.compile(r"(\d{8})_(\d{4})\.png$")


def frame_time(path) -> datetime | None:
    """Recover the observation time a frame was rendered from."""
    m = _STAMP.search(Path(path).name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def fill_gaps(
    frames: Sequence[Path],
    step_minutes: int = 10,
    max_hold: int = 6,
) -> list[Path]:
    """Repeat frames across missing timeslots so playback runs at constant rate.

    Without this, a gap reads as an unexplained jump: the plume teleports. With
    it, the animation visibly pauses, which is an honest depiction of missing
    data rather than a hidden discontinuity.

    `max_hold` caps how many consecutive slots a single frame will cover, so a
    long night gap in a daylight-only composite does not freeze the video for
    minutes on end.
    """
    dated = [(frame_time(f), f) for f in frames]
    dated = sorted((t, f) for t, f in dated if t is not None)
    if not dated:
        return list(frames)

    step = timedelta(minutes=step_minutes)
    out: list[Path] = []
    prev_t: datetime | None = None
    for t, f in dated:
        if prev_t is not None:
            gap = int((t - prev_t) / step) - 1
            out.extend([out[-1]] * min(gap, max_hold))
        out.append(f)
        prev_t = t
    return out


def to_mp4(
    frames: Sequence[Path],
    out_path,
    fps: int = 12,
    quality: int = 7,
    max_width: int | None = None,
    macro_block_size: int = 2,
) -> Path:
    """Encode frames to H.264 MP4.

    `max_width` downscales on the way in - useful for turning a 5000 px
    true-colour series into something a browser will scrub through smoothly
    without re-rendering anything.

    `macro_block_size` defaults to 2, not imageio's 16. H.264 only requires
    even dimensions, and the regional grid is 1250x700, which 16 would pad to
    1264x704 - resampling every single frame to no benefit.
    """
    import imageio.v2 as imageio
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=quality,
        macro_block_size=macro_block_size, pixelformat="yuv420p",
    )
    try:
        for f in frames:
            img = Image.open(f).convert("RGB")
            if max_width and img.width > max_width:
                h = round(img.height * max_width / img.width)
                img = img.resize((max_width, h), Image.LANCZOS)
            writer.append_data(_as_array(img))
    finally:
        writer.close()
    return out_path


def to_gif(
    frames: Sequence[Path],
    out_path,
    fps: int = 10,
    max_width: int = 900,
    colors: int = 128,
    loop: int = 0,
) -> Path:
    """Encode frames to an animated GIF via PIL's own palette quantisation.

    Ash and dust RGBs survive a 128-colour palette well - they are already
    close to false-colour. True colour suffers visibly; prefer the MP4 there.
    """
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for f in frames:
        img = Image.open(f).convert("RGB")
        if img.width > max_width:
            h = round(img.height * max_width / img.width)
            img = img.resize((max_width, h), Image.LANCZOS)
        images.append(img.quantize(colors=colors, method=Image.MEDIANCUT))

    if not images:
        raise ValueError("no frames to write")

    images[0].save(
        out_path, save_all=True, append_images=images[1:],
        duration=round(1000 / fps), loop=loop, optimize=True, disposal=2,
    )
    return out_path


def side_by_side(
    left: Sequence[Path],
    right: Sequence[Path],
    out_dir,
    gap: int = 6,
    background: tuple[int, int, int] = (10, 10, 12),
) -> list[Path]:
    """Pair two composites of the same scene into one frame each.

    Matched on observation time, not on position in the list, so a
    daylight-only composite paired against a 24-hour one produces panels only
    where both exist rather than drifting out of sync partway through.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_time = {frame_time(f): f for f in right}
    made: list[Path] = []
    for lf in left:
        t = frame_time(lf)
        rf = by_time.get(t)
        if t is None or rf is None:
            continue
        a, b = Image.open(lf).convert("RGB"), Image.open(rf).convert("RGB")
        if b.height != a.height:
            b = b.resize((round(b.width * a.height / b.height), a.height), Image.LANCZOS)
        canvas = Image.new("RGB", (a.width + gap + b.width, a.height), background)
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + gap, 0))
        path = out_dir / f"pair_{t:%Y%m%d_%H%M}.png"
        canvas.save(path)
        made.append(path)
    return made


def _as_array(img):
    import numpy as np

    return np.asarray(img)


def describe(frames: Sequence[Path], fps: int = 12) -> str:
    """One-line summary of what a frame list will turn into."""
    times = sorted(t for t in (frame_time(f) for f in frames) if t is not None)
    if not times:
        return f"{len(frames)} frames"
    span = times[-1] - times[0]
    hours = span.total_seconds() / 3600
    seconds = len(frames) / fps
    return (
        f"{len(frames)} frames | {times[0]:%d %b %H:%M} to {times[-1]:%d %b %H:%M} UTC "
        f"({hours:.1f} h) | {seconds:.1f} s at {fps} fps "
        f"({hours * 3600 / max(seconds, 1e-9):.0f}x real time)"
    )
