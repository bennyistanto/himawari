"""Frames to video.

MP4 is the primary output. A three-day run at 10-minute cadence is roughly 400
frames, and as an H.264 video that is a few tens of megabytes at full 5000 px
width, where the equivalent GIF would be gigabytes and dithered to 256 colours.
GIF is kept for the places that still demand one - chat apps, some CMSs - at a
reduced width.

Frame timing is handled by filling gaps rather than by dropping them. Where a
timeslot is missing (AHI housekeeping at 02:40 and 14:40 UTC, or a night gap in
a daylight-only composite), `fill_gaps` inserts a frame so playback keeps a
constant amount of real time per second instead of silently jumping.

Short gaps are bridged by blending the two frames either side, which reads as
motion. Long ones repeat the previous frame, which reads as a pause and is the
honest depiction of data that is simply absent. Nothing synthesised is written
to disk: a bridged frame is a `Blend` describing how to make it, materialised
only as the encoder asks for it, so `frames/` stays one PNG per real
observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence, Union

_STAMP = re.compile(r"(\d{8})_(\d{4})\.png$")


def frame_time(path) -> datetime | None:
    """Recover the observation time a frame was rendered from."""
    m = _STAMP.search(Path(path).name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


@dataclass(frozen=True)
class Blend:
    """A frame that was never observed, to be synthesised at encode time.

    Holds paths rather than pixels so a three-day sequence costs nothing in
    memory, and so nothing synthetic is ever written beside the real frames.
    """

    a: Path
    b: Path
    weight: float                 # 0 gives `a`, 1 gives `b`
    when: datetime | None = None

    def load(self):
        from PIL import Image

        ia = Image.open(self.a).convert("RGB")
        ib = Image.open(self.b).convert("RGB")
        if ia.size != ib.size:
            ib = ib.resize(ia.size, Image.LANCZOS)
        return Image.blend(ia, ib, self.weight)


FrameSource = Union[Path, str, Blend]


def open_frame(item: FrameSource):
    """A PIL image for either a real frame or a blended one."""
    from PIL import Image

    if isinstance(item, Blend):
        return item.load()
    return Image.open(item).convert("RGB")


def _bridge(t0, a, t1, b, gap, step, mode, max_blend, max_hold, daylight_test):
    """The frames that belong between two observations `gap` slots apart."""
    if mode == "drop":
        return []

    usable = mode == "blend" and gap <= max_blend
    if usable and daylight_test is not None:
        # Never dissolve across the terminator. In a day/night blend the two
        # endpoints are different products, so a cross-fade would invent a
        # colour that no instrument returns. Hold instead.
        try:
            usable = daylight_test(t0) == daylight_test(t1)
        except Exception:  # noqa: BLE001 - a broken test must not stop encoding
            usable = False

    if usable:
        return [Blend(a, b, (i + 1) / (gap + 1), t0 + step * (i + 1))
                for i in range(gap)]
    return [a] * min(gap, max_hold)


def fill_gaps(
    frames: Sequence[Path],
    step_minutes: int = 10,
    mode: str = "blend",
    max_blend: int = 3,
    max_hold: int = 6,
    daylight_test: Callable[[datetime], bool] | None = None,
) -> list[FrameSource]:
    """Bridge missing timeslots so playback runs at a constant rate.

    Without this a gap reads as an unexplained jump: the plume teleports.

    `mode` chooses how a gap is bridged:

    * ``"blend"`` (default) interpolates linearly between the frames either
      side, so a gap of *n* slots yields *n* frames at weights
      ``i / (n + 1)``. AHI's two daily housekeeping slots are a single missing
      frame each, so this is a plain 50/50 average there.
    * ``"hold"`` repeats the previous frame, the older behaviour.
    * ``"drop"`` inserts nothing, which speeds the clock up across the gap.

    `max_blend` is why blending does not simply replace holding. Cross-fading
    two frames an hour apart does not depict the hour between them; it invents
    a plume in two places at once. Past this many missing slots the gap is held
    instead, so the animation visibly pauses, which is the honest depiction of
    absent data.

    `daylight_test` guards the day/night blends. Given a callable that reports
    whether the sun is up for a timestamp, a gap whose endpoints disagree is
    held rather than blended, because there the two frames are different
    products and a dissolve between them means nothing.

    Returns a mixed list of frame paths and `Blend` descriptors; pass each
    through `open_frame`.
    """
    if mode not in ("blend", "hold", "drop"):
        raise ValueError(f"mode must be blend, hold or drop, not {mode!r}")

    dated = [(frame_time(f), f) for f in frames]
    dated = sorted((t, f) for t, f in dated if t is not None)
    if not dated:
        return list(frames)

    step = timedelta(minutes=step_minutes)
    out: list[FrameSource] = []
    prev_t: datetime | None = None
    prev_f: Path | None = None
    for t, f in dated:
        if prev_t is not None:
            gap = int(round((t - prev_t) / step)) - 1
            if gap > 0:
                out.extend(_bridge(prev_t, prev_f, t, f, gap, step,
                                   mode, max_blend, max_hold, daylight_test))
        out.append(f)
        prev_t, prev_f = t, f
    return out


# Motion-compensated frame interpolation, as an ffmpeg filter.
#
# `mci` estimates motion between each pair of real frames and moves the
# pixels along it, rather than averaging the two (which on a plume drifting
# 4 to 7 px per step gives two plumes at half brightness, not one plume
# halfway between). `bidir` searches both directions, `aobmc` and `vsbmc`
# smooth the block edges that otherwise show as a quilt over fast motion.
INTERPOLATORS = {
    "minterpolate": "minterpolate=fps={out_fps}:mi_mode=mci"
                    ":mc_mode=aobmc:me_mode=bidir:vsbmc=1",
}


def to_mp4(
    frames: Sequence[Path],
    out_path,
    fps: int = 12,
    quality: int = 7,
    max_width: int | None = None,
    macro_block_size: int = 2,
    interpolate: str | None = None,
    out_fps: int | None = None,
) -> Path:
    """Encode frames to H.264 MP4.

    `max_width` downscales on the way in - useful for turning a 5000 px
    true-colour series into something a browser will scrub through smoothly
    without re-rendering anything.

    `macro_block_size` defaults to 2, not imageio's 16. H.264 only requires
    even dimensions, and the regional grid is 1250x700, which 16 would pad to
    1264x704 - resampling every single frame to no benefit.

    `interpolate` synthesises intermediate frames in the encoder, raising the
    played frame rate to `out_fps` (default twice `fps`) without changing the
    duration. `fps` therefore stays the rate of *real* observations and keeps
    setting how much satellite time passes per second of video, so a caption
    like "one second is one hour" survives interpolation unaltered. At the
    native 10-minute cadence, doubling gives an effective 5-minute step.

    Only ``"minterpolate"`` is available. The frames it adds are synthesised,
    not observed, and carry no information the real frames do not.
    """
    import imageio.v2 as imageio
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_params: list[str] = []
    if interpolate is not None:
        if interpolate not in INTERPOLATORS:
            raise ValueError(
                f"interpolate must be None or one of "
                f"{sorted(INTERPOLATORS)}, not {interpolate!r}")
        chain = INTERPOLATORS[interpolate].format(out_fps=out_fps or fps * 2)
        # Own the whole -vf chain. imageio appends its own `-vf scale=...`
        # when a frame is not divisible by macro_block_size, and ffmpeg keeps
        # only the last -vf, so letting both through would drop one silently.
        size = open_frame(frames[0]).size if frames else (0, 0)
        if max_width and size[0] > max_width:
            size = (max_width, round(size[1] * max_width / size[0]))
        if size[0] % 2 or size[1] % 2:
            chain = f"scale={size[0] // 2 * 2}:{size[1] // 2 * 2}," + chain
        macro_block_size = 1
        output_params = ["-vf", chain]
        # `mci` interpolates toward a following frame, so with nothing after
        # the last one it emits neither it nor its neighbourhood: measured as
        # the final real observation vanishing from the video. Repeating the
        # last frame gives the filter something to aim at. The frames this
        # adds past the end interpolate between a frame and its own copy, so
        # they hold rather than invent.
        if frames:
            frames = list(frames) + [frames[-1]]

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", quality=quality,
        macro_block_size=macro_block_size, pixelformat="yuv420p",
        output_params=output_params,
    )
    try:
        for f in frames:
            img = open_frame(f)
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
        img = open_frame(f)
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


def _any_time(item: FrameSource) -> datetime | None:
    return item.when if isinstance(item, Blend) else frame_time(item)


def describe(frames: Sequence[FrameSource], fps: int = 12) -> str:
    """One-line summary of what a frame list will turn into."""
    times = sorted(t for t in (_any_time(f) for f in frames) if t is not None)
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
