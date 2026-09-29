"""Build a narrated cut (title cards + showcase clips) from rendered showcases.

Run from an episode folder with that episode's venv, e.g.
    cd 005-sc2-stutter-step && uv run python ../tools/build_cut.py <out_dir>
Edit `segments` below: ("card", [title, line, ...]) or ("clip", <showcase.mp4>).
Clips drop their own 3 s title and results cards. Cards use arena.overlay.card.
Card text must match measured results.
"""
import subprocess, sys
from pathlib import Path
from arena.overlay import card

OUT = Path(sys.argv[1]); RUNS = Path("runs"); R4 = Path("../004-sc2-squad-commander/runs")
CARD_S = 4.5
segments = [
    ("card", ["Stutter-step", "Episode 005: 12 Marines vs 10 Banelings + 10 Zerglings",
              "Shoot when the rifle is ready, step back while it reloads."]),
    ("card", ["Before: episode 004 commander", "Jev plans, Marines shoot and split, no stutter-step",
              "8 of 20 wins. Survivors average 18 HP."]),
    ("clip", R4 / "round4g/jev_commander-20260928-183614-s14/showcase.mp4"),
    ("card", ["Stutter-step, no AI at all", "Every Marine stutter-steps the whole fight",
              "20 of 20 wins. 8.9 Marines alive, 51 HP each."]),
    ("clip", RUNS / "stutter_all-20260928-203159-s0/showcase.mp4"),
    ("card", ["Jev commander + stutter-step", "Jev plans; Marines it is unsure about fall back to stutter-step",
              "20 of 20 wins, fastest fights (9.4 s). Jev picked stutter itself only 4% of the time."]),
    ("clip", RUNS / "jev_commander_stutter-20260928-211554-s0/showcase.mp4"),
    ("card", ["The lesson", "The technique mattered more than the brain",
              "Random 14/20 · Jev commander 8/20 · stutter_all 20/20 · Jev + stutter 20/20",
              "Fast reflexes in front, a slower planner behind: how game AI is built."]),
]
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
       "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
parts = []
for i, (kind, val) in enumerate(segments):
    seg = OUT / f"seg{i:02d}.mp4"
    if kind == "card":
        png = OUT / f"card{i:02d}.png"; card(val).save(png)
        cmd = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(CARD_S), "-i", str(png),
               "-f", "lavfi", "-t", str(CARD_S), "-i", "anullsrc=r=48000:cl=stereo", "-shortest", *ENC, str(seg)]
    else:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(val)],
                                   capture_output=True, text=True, check=True).stdout)
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", "3", "-t", f"{dur - 6:.2f}", "-i", str(val), *ENC, str(seg)]
    subprocess.run(cmd, check=True); parts.append(seg)
lst = OUT / "list.txt"; lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
final = OUT / "ep005-stutter-step.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(final)], check=True)
print(final)
