"""
Mémoire tampon vidéo : garde les dernières minutes d'images pour accompagner une alerte.

Les images sont gardées en JPEG réduit (≈ 40 Mo pour 3 min à 5 images/s en 640 px).
Le clip est encodé en MP4 H.264 avec ffmpeg, le seul format lisible à la fois sur
iPhone, Android et navigateur. Sans ffmpeg, repli sur OpenCV (format lisible avec VLC).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger("safenest.clip")


class Clip:
    """Copie figée des images au moment de l'incident, encodée à la demande."""

    def __init__(self, frames: list[tuple[float, bytes]]):
        self.frames = frames

    @property
    def duration(self) -> float:
        return self.frames[-1][0] - self.frames[0][0]

    def _fps(self) -> float:
        # Cadence réelle mesurée, pour que la durée du clip corresponde à la réalité
        if self.duration <= 0:
            return 5.0
        return max(1.0, min(30.0, (len(self.frames) - 1) / self.duration))

    def export_mp4(self) -> str:
        """Encode le clip dans un fichier temporaire et renvoie son chemin (à supprimer après usage)."""
        fd, path = tempfile.mkstemp(prefix="safenest-clip-", suffix=".mp4")
        os.close(fd)
        if shutil.which("ffmpeg"):
            self._export_ffmpeg(path)
        else:
            log.warning("ffmpeg absent : clip encodé par OpenCV, illisible sur iPhone et navigateur")
            self._export_opencv(path)
        return path

    def _export_ffmpeg(self, path: str) -> None:
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-framerate", f"{self._fps():.3f}", "-c:v", "mjpeg", "-i", "-",
            # Dimensions paires et couleurs « plage vidéo » (le JPEG est en plage complète,
            # mal gérée par certains lecteurs) : H.264 standard, lisible partout
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            path,
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _, jpeg in self.frames:
                process.stdin.write(jpeg)
        except BrokenPipeError:
            pass  # ffmpeg s'est arrêté : l'erreur est lue ci-dessous
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=120) != 0:
            raise RuntimeError(f"ffmpeg a échoué : {stderr.strip()[:300]}")

    def _export_opencv(self, path: str) -> None:
        first = cv2.imdecode(np.frombuffer(self.frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps(), (width, height))
        try:
            for _, jpeg in self.frames:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()


class ClipBuffer:
    def __init__(self, seconds: float = 180, fps: float = 5, max_width: int = 640,
                 jpeg_quality: int = 70):
        self.seconds = seconds
        self.interval = 1.0 / fps
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        self._frames: deque[tuple[float, bytes]] = deque()
        self._last_added = 0.0
        self._lock = threading.Lock()

    def add(self, frame) -> None:
        """Ajoute une image (BGR OpenCV). Les images en trop sont ignorées pour tenir la cadence."""
        now = time.time()
        if now - self._last_added < self.interval:
            return
        self._last_added = now

        height, width = frame.shape[:2]
        if width > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(height * self.max_width / width)))
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return

        with self._lock:
            self._frames.append((now, jpeg.tobytes()))
            while self._frames and self._frames[0][0] < now - self.seconds:
                self._frames.popleft()

    def snapshot(self) -> Clip | None:
        """Images actuellement en mémoire (au moins 2), ou None."""
        with self._lock:
            frames = list(self._frames)
        return Clip(frames) if len(frames) >= 2 else None

    def clear(self) -> None:
        """À appeler quand la source vidéo change (les dimensions peuvent changer)."""
        with self._lock:
            self._frames.clear()
