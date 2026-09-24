#!/usr/bin/env python3
"""
Diffuse la webcam (Mac ou Linux) en MJPEG pour le Prototype SafeNest.

Usage : lancer via ./lancer_camera.sh (installe tout au premier lancement)
    ./lancer_camera.sh                        # caméra 0, port 8081
    ./lancer_camera.sh --camera 1             # autre caméra (ex. iPhone via Continuity Camera)

Le Raspberry Pi le détecte tout seul : dans l'interface SafeNest,
Paramètres > Source vidéo > 🔄 Scanner.
(URL manuelle si besoin : http://<IP-du-Mac>:8081/video)
"""

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

latest_jpeg = None
lock = threading.Lock()
INFO = {"service": "safenest-cam", "name": socket.gethostname().split(".")[0], "camera": 0}


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        if sys.platform == "darwin":
            hint = ("Vérifie l'autorisation Caméra du Terminal dans Réglages Système > "
                    "Confidentialité et sécurité > Caméra.")
        else:
            hint = "Vérifie qu'une webcam est branchée (ls /dev/video*) et que tu es dans le groupe 'video'."
        raise SystemExit(f"[ERREUR] Impossible d'ouvrir la caméra {camera_index}. {hint}")
    print(f"[OK] Caméra {camera_index} ouverte")
    return cap


def capture_loop(cap, width, quality, fps):
    global latest_jpeg
    delay = 1.0 / fps
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()
        time.sleep(delay)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpeg = latest_jpeg
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/info":
            body = json.dumps(INFO).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            body = b'<html><body style="margin:0;background:#000"><img src="/video" style="width:100%"></body></html>'
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description="Diffuse la webcam en MJPEG pour SafeNest")
    p.add_argument("--camera", type=int, default=0, help="index de la caméra (défaut 0)")
    p.add_argument("--port", type=int, default=8081, help="port HTTP (défaut 8081)")
    p.add_argument("--width", type=int, default=960, help="largeur max des images")
    p.add_argument("--quality", type=int, default=70, help="qualité JPEG 0-100")
    p.add_argument("--fps", type=int, default=15, help="images par seconde")
    p.add_argument("--name", default=INFO["name"], help="nom affiché dans SafeNest")
    a = p.parse_args()
    INFO.update(name=a.name, camera=a.camera)

    cap = open_camera(a.camera)
    threading.Thread(
        target=capture_loop, args=(cap, a.width, a.quality, a.fps), daemon=True
    ).start()

    print(f"[STREAM] http://0.0.0.0:{a.port}/video  (Ctrl+C pour arrêter)")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
