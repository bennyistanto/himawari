"""Run the whole pipeline against examples/tonga.yml, end to end.

This exists to test that nothing in the pipeline is specific to the Krakatau
example: a different satellite, hemisphere, dateline-crossing region and
timezone, with no code change.
"""
import sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from himawari import project, catalog
from himawari import render as R
from himawari import animate as A
from himawari.fetch import download, complete_times, estimate


# The __main__ guard is not optional on Windows: render_many(workers>1)
# uses ProcessPoolExecutor, which spawns fresh interpreters that re-import
# this module. Without the guard each worker re-runs the whole script and
# the pool dies with BrokenProcessPool.
def main():
    cfg = project.load(ROOT / "examples" / "tonga.yml")
    R.use_local_config(); cfg.make_dirs()
    print(cfg.summary()); print()

    tasks = cfg.tasks()
    est = estimate(tasks, cfg.backend(), sample_slots=3)
    print(f"estimate: {est['files']} files, {est['mb_per_slot']:.0f} MB/slot, "
          f"{est['total_gb']:.2f} GB total")

    t0 = time.time()
    rep = download(tasks, cfg.cache_dir, cfg.backend(), workers=cfg.download_workers,
                   progress=False)
    print("download:", rep.summary())

    good, bad = complete_times(tasks, cfg.cache_dir)
    print(f"complete timeslots: {len(good)}, incomplete: {len(bad)}")
    if not good:
        sys.exit("nothing to render")

    for p in cfg.products:
        ov = R.coastline_overlay(p.grid, cfg.overlay_path(p.grid))
        R.render_many(good[:1], cfg.cache_dir, cfg.frames_dir, [p.composite], p.grid,
                      overlay_path=ov, marker=cfg.marker, tz=cfg.tz, tz_label=cfg.tz_label,
                      resample_cache=cfg.resample_cache, workers=1, progress=False)
        R.render_many(good, cfg.cache_dir, cfg.frames_dir, [p.composite], p.grid,
                      overlay_path=ov, marker=cfg.marker, tz=cfg.tz, tz_label=cfg.tz_label,
                      resample_cache=cfg.resample_cache,
                      workers=cfg.render_workers, skip_dark=cfg.skip_dark, progress=False)
        frames = R.rendered_frames(cfg.frames_dir, p.composite, p.grid)
        # Coverage check. The failure this guards against is silent: a frame
        # that is 90% black still renders, still animates, and looks like a
        # night scene until you notice the straight diagonal edge.
        import numpy as np
        from PIL import Image
        if frames:
            mid = np.asarray(Image.open(frames[len(frames)//2]).convert("L"))
            lit = float((mid > 8).mean()) * 100
            flag = "" if lit > 60 else "   <<< CHECK: mostly empty"
            print(f"rendered {p.composite:6s}: {len(frames)} frames, "
                  f"{lit:.0f}% of the frame carries data{flag}")
        else:
            print(f"rendered {p.composite:6s}: no frames")
        if frames:
            out = Path(cfg.output_dir) / f"tonga_{p.composite}.gif"
            A.to_gif(A.fill_gaps(frames, step_minutes=cfg.step_minutes), out,
                     fps=cfg.animation.get("gif_fps", 6),
                     max_width=cfg.animation.get("gif_max_width", 800))
            print(f"   {out.name}  {out.stat().st_size/2**20:.1f} MB")
    print(f"\ntotal wall clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
