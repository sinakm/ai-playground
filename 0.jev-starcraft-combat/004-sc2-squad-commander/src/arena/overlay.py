"""Render the Jev decision panel and composite it with the captured game video."""

from __future__ import annotations

import bisect
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from arena import config
from arena.evaluate import cost_usd
from arena.log import read_jsonl

PANEL_SIZE = (480, 1080)
GAME_SIZE = (1440, 1080)
FPS = 30
CARD_SECONDS = 3
BG = (18, 20, 26)
FG = (235, 235, 240)
MUTED = (140, 145, 160)
BAR = (76, 114, 176)
CHOSEN = (85, 168, 104)

POLICY_LABELS = {
    "jev_commander": (
        "JEV COMMANDER",
        "Jev vs Banelings, round 4: a squad with a commander",
        "a Jev commander plans, every Marine decides with a shared blackboard",
    ),
    "jev_marine": (
        "JEV DECIDES PER MARINE",
        "Jev vs Banelings, round 3: every Marine decides",
        "Jev told the goal: one decision for each Marine",
    ),
    "random": (
        "RANDOM PICKS PER MARINE",
        "Random vs Banelings",
        "baseline: a random pick of 8 actions per marine each step",
    ),
    "attack_move": ("ATTACK-MOVE", "Attack-move vs Banelings", "baseline: no AI, just charge"),
}
PER_MARINE_POLICIES = ("random", "jev_marine", "jev_commander")


def _font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def decision_index_for_frames(wall_times: list[float], start_wall: float, n_frames: int, fps: int) -> list[int]:
    return [bisect.bisect_right(wall_times, start_wall + i / fps) - 1 for i in range(n_frames)]


def trim_offset(records: list[dict], record_start_wall: float, lead_s: float = 1.0) -> float:
    """Seconds to trim off the start of the capture so playback begins `lead_s` before the first decision."""
    if not records:
        return 0.0
    return max(0.0, records[0]["wall_time"] - record_start_wall - lead_s)


def cumulative(records: list[dict]) -> list[dict]:
    out, tin, tout = [], 0, 0
    for n, r in enumerate(records, start=1):
        tin += r["input_tokens"]
        tout += r["output_tokens"]
        out.append({"decisions": n, "cost_usd": cost_usd(tin, tout)})
    return out


def _action_list(record: dict) -> dict[str, str]:
    if record.get("marine_actions") is None:
        return config.ACTIONS
    if record.get("policy") == "jev_marine":
        return config.JEV_MARINE_ACTIONS
    if record.get("policy") == "jev_commander":
        # Stim is the commander's call (shown in the header); retreat_to_squad is a reflex.
        return {**config.COMMANDER_MARINE_ACTIONS, **{a: "" for a in config.EXECUTED_ONLY_ACTIONS}}
    return config.MARINE_ACTIONS


def _shown_actions(record: dict) -> dict:
    """Per-Marine actions the panel shows: executed (after roles and caps) when logged, else Jev's."""
    executed = record.get("executed_actions")
    return executed if executed else record["marine_actions"]


def panel_actions(record: dict) -> tuple[dict[str, float], str]:
    """Share of Marines per action and the most common action (ties: first in action order)."""
    shown = _shown_actions(record)
    order = list(_action_list(record))
    n = len(shown)
    counts = {a: sum(1 for v in shown.values() if v == a) for a in order}
    top = max(order, key=lambda a: (counts[a], -order.index(a)))
    return {a: counts[a] / n for a in order}, top


def marine_headline(record: dict) -> str:
    """Most common Marine action and how many living Marines do it, e.g. "KITE 7/12"."""
    shown = _shown_actions(record)
    _, action = panel_actions(record)
    count = sum(1 for a in shown.values() if a == action)
    return f"{action.upper()} {count}/{len(shown)}"


def commander_header(record: dict) -> str | None:
    """Panel header for a commander record, e.g. "COMMANDER: FOCUS_BANES"."""
    plan = record.get("squad_plan")
    if not plan:
        return None
    return f"COMMANDER: {plan.upper()}" + (" \u00b7 STIM" if record.get("stim_now") else "")


def _fit_font(d: ImageDraw.ImageDraw, text: str, size: int, max_width: int, min_size: int = 24):
    while size > min_size and d.textlength(text, font=_font(size)) > max_width:
        size -= 2 if size <= 30 else 4
    return _font(size)


def panel_frame(record: dict | None, totals: dict | None, header: str = "JEV DECIDES") -> Image.Image:
    img = Image.new("RGB", PANEL_SIZE, BG)
    d = ImageDraw.Draw(img)
    if record is not None:
        header = commander_header(record) or header
    d.text((32, 40), header, font=_fit_font(d, header, 28, PANEL_SIZE[0] - 64), fill=MUTED)
    if record is None:
        d.text((32, 100), "waiting...", font=_font(56), fill=FG)
        return img
    per_marine = record.get("marine_actions") is not None
    big = marine_headline(record) if per_marine and record["marine_actions"] else record["action"].upper()
    d.text((32, 90), big, font=_fit_font(d, big, 72, PANEL_SIZE[0] - 64), fill=CHOSEN)
    confidence = record["confidence"]
    label = "mean confidence" if per_marine else "confidence"
    conf_text = f"{label} {confidence:.2f}" if confidence is not None else f"{label} -"
    d.text((32, 190), conf_text, font=_font(30), fill=FG)
    y = 270
    probabilities = record["probabilities"]
    actions = _action_list(record)
    row = 90 if len(actions) <= 5 else 58
    top = record["action"]
    if per_marine and record["marine_actions"]:
        probabilities, top = panel_actions(record)
    for action in actions:
        p = (probabilities or {}).get(action, 0.0)
        color = CHOSEN if action == top else BAR
        compact = row < 90
        d.text((32, y), action, font=_font(22 if compact else 26), fill=FG)
        bar_top, bar_bottom = (y + 28, y + 46) if compact else (y + 36, y + 60)
        d.rectangle([32, bar_top, 32 + int(400 * p), bar_bottom], fill=color)
        p_text = f"{p:.2f}" if probabilities is not None else "-"
        d.text((440 - 70, y), p_text, font=_font(22 if compact else 26), fill=MUTED)
        y += row
    latency = f"latency   {record['latency_ms']:.0f} ms"
    if record.get("commander_latency_ms") is not None:
        latency += f" ({record['commander_latency_ms']:.0f}+{record['soldier_latency_ms']:.0f})"
    stats = [
        latency + ("  LATE" if record["late"] else ""),
        f"decisions {totals['decisions']}",
        f"cost      ${totals['cost_usd']:.4f}",
        f"marines   {record['marines_alive']}",
        f"enemies   {record['enemies_alive']}",
    ]
    for i, line in enumerate(stats):
        d.text((32, 760 + i * 56), line, font=_fit_font(d, line, 30, PANEL_SIZE[0] - 64, min_size=16), fill=FG)
    return img


def card(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(img)
    y = 380
    for i, line in enumerate(lines):
        d.text((160, y), line, font=_fit_font(d, line, 84 if i == 0 else 44, 1600), fill=FG if i == 0 else MUTED)
        y += 130 if i == 0 else 70
    return img


def _probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _results_lines(results: dict) -> list[str]:
    lines = ["Results: 10 fights per policy"]
    for policy, m in results.items():
        lines.append(
            f"{policy}: {m['wins']}/{m['runs']} wins, {m['mean_marines_alive']:.1f} marines alive, "
            f"{m['mean_enemies_killed']:.1f} Zerg killed, score {m['mean_score']:.1f}"
        )
    return lines


def render(run_dir: Path, results_summary: Path) -> Path:
    capture = run_dir / "capture.mp4"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = read_jsonl(run_dir / "decisions.jsonl")
    totals = cumulative(records)
    offset = trim_offset(records, summary["record_start_wall"])
    n_frames = int((_probe_duration(capture) - offset) * FPS)
    start_wall = summary["record_start_wall"] + offset
    idx = decision_index_for_frames([r["wall_time"] for r in records], start_wall, n_frames, FPS)

    header, card_title, subtitle = POLICY_LABELS.get(summary["policy"], POLICY_LABELS["jev_marine"])

    title = run_dir / "title.png"
    ending = run_dir / "results.png"
    per_marine = summary["policy"] in PER_MARINE_POLICIES
    unit = "one decision per Marine" if per_marine else "one squad decision"
    if summary["policy"] == "jev_commander":
        unit = "a commander plan then one decision per Marine"
    interval_line = f"12 Marines, {unit} every {config.DECISION_INTERVAL_LOOPS / config.LOOPS_PER_SECOND:.2f} s"
    card([card_title, interval_line, subtitle]).save(title)
    results = json.loads(results_summary.read_text(encoding="utf-8")) if results_summary.exists() else {}
    card(_results_lines(results)).save(ending)

    out = run_dir / "showcase.mp4"
    has_audio = "audio" in subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(capture)],
        capture_output=True, text=True, check=True,
    ).stdout
    audio_in = (
        ["-ss", f"{offset:.3f}", "-i", str(capture)]
        if has_audio
        else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    )
    body_seconds = n_frames / FPS
    gw, gh = GAME_SIZE
    pw, ph = PANEL_SIZE
    filt = (
        f"[0:v]scale={gw}:{gh}:force_original_aspect_ratio=decrease,pad={gw}:{gh}:(ow-iw)/2:(oh-ih)/2,setsar=1[g];"
        f"[g][1:v]hstack=inputs=2:shortest=1,fps={FPS},format=yuv420p,setsar=1[body];"
        f"[5:a]apad,atrim=0:{body_seconds:.3f},asetpts=N/SR/TB,aformat=sample_rates=48000:channel_layouts=stereo[ba];"
        f"[4:a]aformat=sample_rates=48000:channel_layouts=stereo[ta];"
        f"[6:a]aformat=sample_rates=48000:channel_layouts=stereo[ea];"
        f"[2:v]fps={FPS},format=yuv420p,setsar=1[t];[3:v]fps={FPS},format=yuv420p,setsar=1[e];"
        f"[t][ta][body][ba][e][ea]concat=n=3:v=1:a=1[v][a]"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset:.3f}", "-i", str(capture),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{pw}x{ph}", "-r", str(FPS), "-i", "-",
        "-loop", "1", "-t", str(CARD_SECONDS), "-i", str(title),
        "-loop", "1", "-t", str(CARD_SECONDS), "-i", str(ending),
        "-f", "lavfi", "-t", str(CARD_SECONDS), "-i", "anullsrc=r=48000:cl=stereo",
        *audio_in,
        "-f", "lavfi", "-t", str(CARD_SECONDS), "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cache: dict[int, bytes] = {}
    try:
        for i in idx:
            if i not in cache:
                cache[i] = panel_frame(
                    records[i] if i >= 0 else None, totals[i] if i >= 0 else None, header=header
                ).tobytes()
            proc.stdin.write(cache[i])
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg composite failed with code {rc}")
    return out
