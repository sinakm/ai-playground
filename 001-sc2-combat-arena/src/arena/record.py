"""Capture the SC2 window with ffmpeg and speaker audio with soundcard loopback."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK = 1024


def ffmpeg_capture_cmd(out: Path, window_title: str = "StarCraft II", fps: int = 30) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "gdigrab", "-framerate", str(fps), "-i", f"title={window_title}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(out),
    ]


def ffmpeg_mux_cmd(video: Path, audio: Path, out: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out),
    ]


class Recorder:
    def __init__(self, run_dir: Path, audio: bool):
        self.run_dir = run_dir
        self.audio = audio
        self.start_wall: float | None = None
        self._ffmpeg: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._blocks: list[np.ndarray] = []
        self._audio_error: str | None = None

    def start(self) -> None:
        if self.audio:
            self._thread = threading.Thread(target=self._record_audio, daemon=True)
            self._thread.start()
        self._ffmpeg = subprocess.Popen(
            ffmpeg_capture_cmd(self.run_dir / "video.mp4"), stdin=subprocess.PIPE
        )
        self.start_wall = time.time()

    def _record_audio(self) -> None:
        try:
            import soundcard as sc

            mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
            with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as rec:
                while not self._stop.is_set():
                    self._blocks.append(rec.record(numframes=BLOCK))
        except Exception as exc:
            self._audio_error = repr(exc)

    def _write_wav(self, path: Path) -> None:
        data = np.concatenate(self._blocks) if self._blocks else np.zeros((0, CHANNELS))
        pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())

    def stop(self) -> None:
        ffmpeg_code = None
        if self._ffmpeg is not None:
            if self._ffmpeg.poll() is None:
                try:
                    self._ffmpeg.communicate(input=b"q", timeout=30)
                except subprocess.TimeoutExpired:
                    self._ffmpeg.kill()
                    self._ffmpeg.wait()
            ffmpeg_code = self._ffmpeg.returncode

        video = self.run_dir / "video.mp4"
        final = self.run_dir / "capture.mp4"
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            audio = self.run_dir / "audio.wav"
            self._write_wav(audio)
            if self._blocks:
                try:
                    subprocess.run(ffmpeg_mux_cmd(video, audio, final), check=True)
                    # Write audio errors before returning, independent of video success
                    if self._audio_error:
                        error_file = self.run_dir / "recording_error.txt"
                        with error_file.open("a") as f:
                            f.write(f"audio: {self._audio_error}\n")
                    return
                except subprocess.CalledProcessError as e:
                    error_file = self.run_dir / "recording_error.txt"
                    with error_file.open("a") as f:
                        f.write(f"mux failed: {e}\n")
                    # Write audio errors before returning
                    if self._audio_error:
                        with error_file.open("a") as f:
                            f.write(f"audio: {self._audio_error}\n")
                    video.replace(final)
                    return
        if not video.exists() or video.stat().st_size == 0:
            error_file = self.run_dir / "recording_error.txt"
            with error_file.open("w") as f:
                f.write(f"ffmpeg exited with code {ffmpeg_code}; no video captured\n")
            if self._audio_error:
                with error_file.open("a") as f:
                    f.write(f"audio: {self._audio_error}\n")
            return
        # Write audio errors before final video.replace, independent of video success
        if self._audio_error:
            error_file = self.run_dir / "recording_error.txt"
            with error_file.open("a") as f:
                f.write(f"audio: {self._audio_error}\n")
        video.replace(final)


def make_recorder_factory(audio: bool):
    use_audio = audio and sys.platform == "win32"
    return lambda run_dir: Recorder(run_dir, use_audio)
