#!/usr/bin/env python3
"""
Tests de la file d'attente des alertes sur la carte SD (safenest_cloud.py), sans réseau
externe : une fausse API SafeNest locale imite POST /device/alerts (idempotente par clé),
l'envoi du clip sur l'URL signée et sa confirmation.

    python -m unittest test_outbox -v
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import stat
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import safenest_cloud
from safenest_cloud import Outbox, SafeNestCloud

TOKEN = "jeton-de-test"
DEVICE_ID = "11111111-1111-4111-8111-111111111111"
CLIP_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"clip-de-test" * 100

# Les avertissements attendus (API coupée, refus…) ne doivent pas encombrer la sortie
logging.getLogger("safenest.cloud").setLevel(logging.CRITICAL)


def wait_until(predicate, timeout: float = 10, message: str = "délai dépassé"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)


def closed_port_url() -> str:
    """Adresse d'une API coupée (port sans serveur : connexion refusée)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


class FakeClip:
    """Imite Clip de clip_buffer.py : export_mp4() écrit un fichier temporaire."""

    def __init__(self, fail: bool = False, delay: float = 0):
        self.fail = fail
        self.delay = delay

    def export_mp4(self) -> str:
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("ffmpeg absent (test)")
        fd, path = tempfile.mkstemp(prefix="safenest-test-clip-", suffix=".mp4")
        with os.fdopen(fd, "wb") as f:
            f.write(CLIP_BYTES)
        return path


class FakeApi:
    """Fausse API SafeNest : routes /device/* utilisées par le boîtier."""

    def __init__(self):
        self.alerts: dict[str, dict] = {}   # idempotency_key → {"id", "body", "clip_confirmed"}
        self.uploads: dict[str, bytes] = {}  # alert_id → clip reçu
        self.created = 0                     # alertes créées (= proches prévenus)
        self.alert_posts: list[dict] = []    # corps de chaque POST /device/alerts
        self.alert_status: int | None = None  # force une erreur sur POST /device/alerts
        self.lock = threading.Lock()
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status: int, body=None):
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                if data:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _read(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def do_POST(self):
                raw = self._read()
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._reply(401, {"message": "Jeton refusé"})
                if self.path == "/v1/device/heartbeat":
                    return self._reply(200, {"cameras": []})
                if self.path == "/v1/device/alerts":
                    if api.alert_status:
                        return self._reply(api.alert_status, {"message": "Refus de test"})
                    body = json.loads(raw)
                    with api.lock:
                        api.alert_posts.append(body)
                        alert = api.alerts.get(body["idempotency_key"])
                        if alert is None:
                            alert = {"id": str(uuid.uuid4()), "body": body, "clip_confirmed": False}
                            api.alerts[body["idempotency_key"]] = alert
                            api.created += 1
                    upload = None
                    if body.get("with_clip") and not alert["clip_confirmed"]:
                        upload = {
                            "path": f"clips/{alert['id']}.mp4",
                            "signed_url": f"{api.base}/upload/{alert['id']}",
                            "token": "jeton-envoi",
                        }
                    return self._reply(201, {"alert_id": alert["id"], "clip_upload": upload})
                parts = self.path.split("/")  # ["", "v1", "device", "alerts", <id>, "clip"]
                if len(parts) == 6 and parts[3] == "alerts" and parts[5] == "clip":
                    with api.lock:
                        alert = next((a for a in api.alerts.values() if a["id"] == parts[4]), None)
                        if alert is None:
                            return self._reply(400, {"message": "Alerte inconnue pour ce boîtier"})
                        if parts[4] not in api.uploads:
                            return self._reply(409, {"message": "Le clip n'a pas encore été envoyé"})
                        alert["clip_confirmed"] = True
                    return self._reply(204)
                return self._reply(404, {"message": "Route inconnue"})

            def do_PUT(self):
                data = self._read()
                if not self.path.startswith("/upload/"):
                    return self._reply(404, {"message": "Route inconnue"})
                with api.lock:
                    api.uploads[self.path.split("/")[2]] = data
                return self._reply(200, {"Key": "ok"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.url = f"{self.base}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def only_alert(self) -> dict:
        assert len(self.alerts) == 1, f"{len(self.alerts)} alertes reçues au lieu d'une"
        return next(iter(self.alerts.values()))


class SafeNestTestBase(unittest.TestCase):
    """Boîtier appairé, état dans un dossier temporaire, fausse API locale."""

    def setUp(self):
        self.state_dir = tempfile.mkdtemp(prefix="safenest-test-")
        self.addCleanup(shutil.rmtree, self.state_dir, True)
        for patcher in (
            mock.patch.dict(os.environ, {"SAFENEST_STATE_DIR": self.state_dir}),
            # Essais immédiats sans attente : les tests restent rapides
            mock.patch.object(safenest_cloud, "RETRY_DELAYS_S", (0, 0)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Boîtier déjà appairé
        with open(os.path.join(self.state_dir, "device.json"), "w", encoding="utf-8") as f:
            json.dump({"device_id": DEVICE_ID, "device_token": TOKEN}, f)
        self.api = FakeApi()
        self.addCleanup(self.api.close)
        self.outbox_dir = os.path.join(self.state_dir, "outbox")

    def make_cloud(self, api_url: str) -> SafeNestCloud:
        cloud = SafeNestCloud(api_url=api_url)
        self.addCleanup(cloud.stop)
        return cloud

    def settle(self, cloud: SafeNestCloud):
        """Attend la fin des fils d'envoi et d'encodage du boîtier."""
        wait_until(lambda: not cloud._sending and not cloud._encoding,
                   message="les fils d'envoi ne se terminent pas")
        time.sleep(0.2)  # dernier essai du fil d'encodage, lancé juste après
        for key in cloud.outbox.keys():
            with cloud._entry_lock(key):
                pass

    def add_entry(self, outbox: Outbox, created_at: float | None = None,
                  device_id: str = DEVICE_ID, with_clip: bool = False) -> str:
        key = str(uuid.uuid4())
        outbox.add({
            "type": "fall", "occurred_at": "2026-11-21T10:00:00+00:00", "confidence": 0.9,
            "camera_id": None, "idempotency_key": key, "with_clip": with_clip,
            "device_id": device_id, "created_at": created_at or time.time(),
            "alert_id": None, "clip_sent": False,
        })
        return key


class OutboxTestCase(SafeNestTestBase):
    # ---------------------------------------------------------------- coupure réseau

    def test_api_coupee_alerte_et_clip_gardes_sur_la_carte_sd(self):
        cloud = self.make_cloud(closed_port_url())
        cloud.report_incident("fire", 0.8764, clip=FakeClip())
        self.settle(cloud)

        keys = cloud.outbox.keys()
        self.assertEqual(len(keys), 1)
        entry = os.path.join(self.outbox_dir, keys[0])
        with open(os.path.join(entry, "alert.json"), encoding="utf-8") as f:
            record = json.load(f)
        self.assertEqual(record["type"], "fire")
        self.assertEqual(record["confidence"], 0.876)
        self.assertEqual(record["idempotency_key"], keys[0])
        self.assertTrue(record["with_clip"])
        self.assertIsNone(record["alert_id"])
        with open(os.path.join(entry, "clip.mp4"), "rb") as f:
            self.assertEqual(f.read(), CLIP_BYTES)
        # Vidéo du logement : lisible par le boîtier seul
        self.assertEqual(stat.S_IMODE(os.stat(entry).st_mode), 0o700)
        for name in ("alert.json", "clip.mp4"):
            self.assertEqual(stat.S_IMODE(os.stat(os.path.join(entry, name)).st_mode), 0o600)
        self.assertEqual(cloud.status()["pending_alerts"], 1)
        self.assertEqual(self.api.created, 0)

    def test_api_retablie_une_seule_alerte_et_le_clip(self):
        cloud = self.make_cloud(closed_port_url())
        cloud.report_incident("fire", 0.9, clip=FakeClip())
        self.settle(cloud)
        self.assertEqual(len(cloud.outbox.keys()), 1)

        cloud.api_url = self.api.url
        self.assertEqual(cloud.flush_outbox(), 0)
        alert = self.api.only_alert()
        self.assertEqual(self.api.created, 1)
        self.assertEqual(alert["body"]["type"], "fire")
        self.assertEqual(self.api.uploads[alert["id"]], CLIP_BYTES)
        self.assertTrue(alert["clip_confirmed"])
        self.assertEqual(os.listdir(self.outbox_dir), [])
        self.assertEqual(cloud.status()["pending_alerts"], 0)
        self.assertEqual(cloud.last_alert["id"], alert["id"])
        self.assertTrue(cloud.last_alert["clip"])

        # Rien de plus à envoyer : pas de seconde alerte
        cloud.flush_outbox()
        self.assertEqual(self.api.created, 1)

    def test_reponse_perdue_pas_de_seconde_alerte(self):
        # L'alerte était arrivée, mais la réponse s'est perdue : même clé → même alerte
        cloud = self.make_cloud(self.api.url)
        key = self.add_entry(cloud.outbox)
        self.api.alerts[key] = {"id": str(uuid.uuid4()), "body": {}, "clip_confirmed": False}
        self.api.created = 1

        self.assertEqual(cloud.flush_outbox(), 0)
        self.assertEqual(self.api.created, 1)

    def test_redemarrage_reprend_l_alerte_en_attente(self):
        before = self.make_cloud(closed_port_url())
        before.report_incident("fall", 0.92, clip=FakeClip())
        self.settle(before)
        before.stop()
        self.assertEqual(len(before.outbox.keys()), 1)

        # Nouveau processus (redémarrage du Pi) : la file est reprise au démarrage
        after = self.make_cloud(self.api.url)
        after.start()
        wait_until(lambda: not after.outbox.keys(), message="file d'attente non vidée")
        alert = self.api.only_alert()
        self.assertEqual(alert["body"]["type"], "fall")
        self.assertEqual(self.api.uploads[alert["id"]], CLIP_BYTES)
        self.assertTrue(alert["clip_confirmed"])

    # ---------------------------------------------------------------- ordre d'envoi

    def test_alerte_envoyee_avant_la_fin_de_l_encodage_du_clip(self):
        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9, clip=FakeClip(delay=0.5))
        wait_until(lambda: self.api.created == 1, message="alerte non envoyée")
        # Les proches sont prévenus sans attendre la vidéo
        self.assertNotIn(self.api.only_alert()["id"], self.api.uploads)

        wait_until(lambda: self.api.only_alert()["clip_confirmed"], message="clip non envoyé")
        self.settle(cloud)
        self.assertEqual(self.api.created, 1)
        self.assertFalse(self.api.alert_posts[0]["with_clip"])
        self.assertTrue(self.api.alert_posts[-1]["with_clip"])
        self.assertEqual(cloud.outbox.keys(), [])

    def test_clip_non_encode_l_alerte_part_sans_video(self):
        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9, clip=FakeClip(fail=True))
        self.settle(cloud)
        self.assertEqual(self.api.created, 1)
        self.assertEqual(self.api.uploads, {})
        self.assertEqual(cloud.outbox.keys(), [])

    # ---------------------------------------------------------------- refus

    def test_refus_definitif_entree_abandonnee(self):
        self.api.alert_status = 400
        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9, clip=FakeClip())
        self.settle(cloud)
        self.assertEqual(cloud.outbox.keys(), [])
        self.assertEqual(self.api.created, 0)

    def test_erreur_passagere_entree_gardee(self):
        self.api.alert_status = 503
        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9)
        self.settle(cloud)
        self.assertEqual(len(cloud.outbox.keys()), 1)

        self.api.alert_status = None
        self.assertEqual(cloud.flush_outbox(), 0)
        self.assertEqual(self.api.created, 1)

    def test_boitier_dissocie_entree_gardee(self):
        self.api.alert_status = 401
        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9)
        self.settle(cloud)
        self.assertFalse(cloud.paired)
        self.assertEqual(len(cloud.outbox.keys()), 1)

    def test_autre_appairage_alerte_supprimee_sans_envoi(self):
        # Vidéo filmée sous un appairage précédent (autre foyer) : jamais transmise
        cloud = self.make_cloud(self.api.url)
        self.add_entry(cloud.outbox, device_id="22222222-2222-4222-8222-222222222222", with_clip=True)
        self.assertEqual(cloud.flush_outbox(), 0)
        self.assertEqual(self.api.created, 0)

    # ---------------------------------------------------------------- taille de la file

    def test_entree_trop_vieille_supprimee(self):
        cloud = self.make_cloud(self.api.url)
        old = self.add_entry(cloud.outbox, created_at=time.time() - 15 * 86400)
        self.add_entry(cloud.outbox)

        self.assertEqual(cloud.flush_outbox(), 0)
        self.assertEqual(self.api.created, 1)
        self.assertNotIn(old, self.api.alerts)

    def test_duree_de_conservation_reglable(self):
        with mock.patch.dict(os.environ, {"SAFENEST_OUTBOX_MAX_DAYS": "2"}):
            outbox = Outbox(self.outbox_dir)
        self.assertEqual(outbox.max_age_s, 2 * 86400)
        self.add_entry(outbox, created_at=time.time() - 3 * 86400)
        kept = self.add_entry(outbox, created_at=time.time() - 1 * 86400)
        outbox.prune()
        self.assertEqual(outbox.keys(), [kept])

    def test_file_pleine_les_plus_anciennes_supprimees(self):
        outbox = Outbox(self.outbox_dir, max_entries=3)
        now = time.time()
        keys = [self.add_entry(outbox, created_at=now - 50 + i) for i in range(5)]
        self.assertEqual(outbox.keys(), keys[2:])


try:
    import cv2
    import numpy as np
except ImportError:  # environnement sans OpenCV : le test du vrai clip est ignoré
    cv2 = None


@unittest.skipUnless(cv2 is not None and shutil.which("ffmpeg"), "OpenCV et ffmpeg nécessaires")
class RealClipTest(SafeNestTestBase):
    def test_vrai_clip_mp4_envoye(self):
        from clip_buffer import Clip

        frames = []
        for i in range(10):
            image = np.full((240, 320, 3), (0, 80, 20 * i), np.uint8)
            ok, jpeg = cv2.imencode(".jpg", image)
            self.assertTrue(ok)
            frames.append((1000.0 + i * 0.2, jpeg.tobytes()))

        cloud = self.make_cloud(self.api.url)
        cloud.report_incident("fire", 0.9, clip=Clip(frames))
        wait_until(lambda: self.api.alerts and self.api.only_alert()["clip_confirmed"],
                   timeout=30, message="clip non envoyé")
        data = self.api.uploads[self.api.only_alert()["id"]]
        self.assertEqual(data[4:8], b"ftyp")  # MP4


if __name__ == "__main__":
    unittest.main(verbosity=2)
