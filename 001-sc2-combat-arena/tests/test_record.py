from pathlib import Path

from arena.record import ffmpeg_capture_cmd, ffmpeg_mux_cmd, make_recorder_factory


def test_capture_cmd_targets_window():
    cmd = ffmpeg_capture_cmd(Path("v.mp4"))
    assert cmd[0] == "ffmpeg"
    assert "gdigrab" in cmd
    assert "title=StarCraft II" in cmd
    assert cmd[-1] == "v.mp4"


def test_mux_cmd_copies_video_encodes_aac():
    cmd = ffmpeg_mux_cmd(Path("v.mp4"), Path("a.wav"), Path("o.mp4"))
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[-1] == "o.mp4"


def test_factory_builds_recorder(tmp_path):
    rec = make_recorder_factory(audio=False)(tmp_path)
    assert rec.start_wall is None
