from pathlib import Path
from unittest.mock import MagicMock

from arena.record import ffmpeg_capture_cmd, ffmpeg_mux_cmd, make_recorder_factory, Recorder


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


def test_stop_handles_ffmpeg_failure_no_video(tmp_path):
    """Recorder.stop() does not raise when ffmpeg exited with error and no video.mp4 exists."""
    rec = Recorder(tmp_path, audio=False)
    fake_ffmpeg = MagicMock()
    fake_ffmpeg.poll.return_value = 1
    fake_ffmpeg.returncode = 1
    rec._ffmpeg = fake_ffmpeg

    rec.stop()

    error_file = tmp_path / "recording_error.txt"
    assert error_file.exists()
    content = error_file.read_text()
    assert "code 1" in content


def test_stop_replaces_video_when_exists(tmp_path):
    """Recorder.stop() replaces video.mp4 with capture.mp4 when video exists and no audio."""
    rec = Recorder(tmp_path, audio=False)
    fake_ffmpeg = MagicMock()
    fake_ffmpeg.poll.return_value = 0
    fake_ffmpeg.returncode = 0
    rec._ffmpeg = fake_ffmpeg

    video = tmp_path / "video.mp4"
    video.write_text("test video data")

    rec.stop()

    assert (tmp_path / "capture.mp4").exists()
    assert not video.exists()
