"""
Connexion du boîtier SafeNest à l'API cloud.

Contrat d'API : docs/api.md du dépôt EIP-SafeNest/SafeNest (routes /device/*).

- appairage avec le code à 6 caractères affiché dans la web app ;
- signe de vie toutes les 30 s, qui renvoie la configuration des caméras ;
- envoi des alertes (chute, feu) avec le clip vidéo des minutes précédentes.

Le jeton du boîtier est gardé dans ~/.config/safenest/device.json (lisible par
l'utilisateur seulement). Point d'entrée pour les modèles de détection :

    cloud.report_incident("fall", confidence=0.92, clip=clip_buffer.snapshot())
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import requests

log = logging.getLogger("safenest.cloud")

FIRMWARE_VERSION = "prototype-0.2"
HEARTBEAT_INTERVAL_S = 30
INCIDENT_TYPES = ("fall", "fire")
# Nouvelles tentatives si l'API est injoignable au moment d'une alerte
RETRY_DELAYS_S = (0, 2, 5, 10, 20, 40)


class CloudError(Exception):
    """Erreur de communication avec l'API (message lisible pour l'utilisateur)."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        # Vrai pour les erreurs passagères (réseau, 5xx, 429) : un nouvel essai peut réussir
        self.retryable = retryable


def _state_file() -> str:
    base = os.environ.get("SAFENEST_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "safenest"
    )
    return os.path.join(base, "device.json")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    message = body.get("message") if isinstance(body, dict) else None
    if isinstance(message, list):
        message = " ; ".join(str(m) for m in message)
    return message or f"Erreur {response.status_code} de l'API"


class SafeNestCloud:
    def __init__(
        self,
        api_url: str | None = None,
        camera_status_provider: Callable[[], list[dict]] | None = None,
    ):
        self.api_url = (
            api_url or os.environ.get("SAFENEST_API_URL") or "http://localhost:3333/v1"
        ).rstrip("/")
        # Renvoie l'état des caméras pour le signe de vie : [{"id": ..., "status": "ok"|"error"}]
        self.camera_status_provider = camera_status_provider

        self.cameras: list[dict] = []
        self.last_contact: float | None = None
        self.last_error: str | None = None
        self.last_alert: dict | None = None

        self._lock = threading.Lock()
        self._state = self._load_state()
        self._camera_listeners: list[Callable[[list[dict]], None]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._http = requests.Session()

    # ------------------------------------------------------------------ état local

    @property
    def paired(self) -> bool:
        return bool(self._state.get("device_token"))

    @property
    def device_id(self) -> str | None:
        return self._state.get("device_id")

    def _load_state(self) -> dict:
        try:
            with open(_state_file(), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        path = _state_file()
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        tmp = f"{path}.tmp"
        # Fichier créé directement en 600 : le jeton n'est jamais lisible par les autres
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._state, f)
        os.replace(tmp, path)

    def _forget(self, reason: str | None, only_token: str | None = None) -> None:
        """
        Oublie le jeton. Avec only_token, seulement s'il n'a pas changé entre-temps :
        un refus arrivé en retard ne doit pas effacer un nouvel appairage.
        """
        with self._lock:
            if only_token is not None and self._state.get("device_token") != only_token:
                return
            self._state = {}
            self._save_state()
        self._set_cameras([])
        self.last_error = reason

    # ------------------------------------------------------------------ HTTP

    def _request(self, method: str, path: str, body: dict | None = None,
                 auth: bool = True, timeout: float = 10) -> dict | list | None:
        headers = {}
        token = None
        if auth:
            token = self._state.get("device_token")
            if not token:
                raise CloudError("Boîtier non appairé")
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = self._http.request(
                method, f"{self.api_url}{path}", json=body, headers=headers, timeout=timeout
            )
        except requests.RequestException as error:
            self.last_error = f"API SafeNest injoignable ({self.api_url})"
            raise CloudError(self.last_error, retryable=True) from error

        if response.status_code == 401 and auth:
            # Boîtier supprimé dans la web app, ou nouveau code d'appairage généré
            self._forget(
                "Le boîtier a été dissocié depuis la web app : appairez-le à nouveau.",
                only_token=token,
            )
            raise CloudError("Jeton du boîtier refusé", status=401)
        if not response.ok:
            self.last_error = _error_message(response)
            raise CloudError(
                self.last_error,
                status=response.status_code,
                retryable=response.status_code >= 500 or response.status_code == 429,
            )

        try:
            data = response.json() if response.content else None
        except ValueError as error:
            # Proxy, portail captif… : une page HTML au lieu de la réponse de l'API
            self.last_error = "Réponse inattendue de l'API SafeNest"
            raise CloudError(self.last_error, status=response.status_code, retryable=True) from error

        self.last_contact = time.time()
        self.last_error = None
        return data

    # ------------------------------------------------------------------ appairage

    def pair(self, code: str) -> str:
        """Échange le code affiché dans la web app contre le jeton du boîtier."""
        data = self._request(
            "POST",
            "/device/pair",
            {"pairing_code": code.strip(), "firmware_version": FIRMWARE_VERSION},
            auth=False,
        )
        if not isinstance(data, dict) or not data.get("device_id") or not data.get("device_token"):
            raise CloudError("Réponse inattendue de l'API SafeNest (appairage)")
        with self._lock:
            self._state = {
                "device_id": data["device_id"],
                "device_token": data["device_token"],
                "paired_at": _iso(time.time()),
            }
            self._save_state()
        log.info("Boîtier appairé (%s)", data["device_id"])
        self._wake.set()  # signe de vie immédiat pour récupérer les caméras
        return data["device_id"]

    def unpair(self) -> None:
        """Oublie le jeton localement (le boîtier reste visible dans la web app)."""
        self._forget(None)

    # ------------------------------------------------------------------ signe de vie

    def start(self) -> None:
        threading.Thread(target=self._heartbeat_loop, name="safenest-heartbeat", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def heartbeat(self) -> list[dict]:
        statuses = self.camera_status_provider() if self.camera_status_provider else []
        data = self._request(
            "POST",
            "/device/heartbeat",
            {"firmware_version": FIRMWARE_VERSION, "cameras": statuses},
        )
        cameras = data.get("cameras") if isinstance(data, dict) else None
        if not isinstance(cameras, list):
            raise CloudError("Réponse inattendue de l'API SafeNest (caméras)", retryable=True)
        self._set_cameras(cameras)
        return self.cameras

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if self.paired:
                try:
                    self.heartbeat()
                except CloudError as error:
                    log.warning("Signe de vie : %s", error)
                except Exception:
                    # Jamais d'arrêt du signe de vie : sans lui, le boîtier paraît hors ligne
                    log.exception("Signe de vie : erreur inattendue")
            self._wake.wait(HEARTBEAT_INTERVAL_S)
            self._wake.clear()

    # ------------------------------------------------------------------ caméras

    def on_cameras_changed(self, listener: Callable[[list[dict]], None]) -> None:
        self._camera_listeners.append(listener)

    def _set_cameras(self, cameras: list[dict]) -> None:
        changed = cameras != self.cameras
        self.cameras = cameras
        if changed:
            for listener in self._camera_listeners:
                try:
                    listener(cameras)
                except Exception:  # un écouteur défaillant ne doit pas bloquer les autres
                    log.exception("Écouteur de caméras en échec")

    def active_camera(self) -> dict | None:
        """Première caméra activée dans la web app (le prototype n'en analyse qu'une)."""
        return next((c for c in self.cameras if c.get("enabled")), None)

    # ------------------------------------------------------------------ alertes

    def report_incident(self, incident_type: str, confidence: float | None = None,
                        camera_id: str | None = None, clip=None,
                        occurred_at: float | None = None) -> None:
        """
        Signale un incident à l'API, sans bloquer l'appelant.

        incident_type : "fall" (chute) ou "fire" (départ de feu)
        confidence    : score du modèle entre 0 et 1
        clip          : objet renvoyé par ClipBuffer.snapshot() (ou None)
        """
        if incident_type not in INCIDENT_TYPES:
            raise ValueError(f"Type d'incident inconnu : {incident_type}")
        if not self.paired:
            log.warning("Incident %s non transmis : boîtier non appairé", incident_type)
            return
        threading.Thread(
            target=self._send_incident,
            args=(incident_type, confidence, camera_id, clip, occurred_at or time.time()),
            name=f"safenest-alert-{incident_type}",
            daemon=True,
        ).start()

    def _send_incident(self, incident_type, confidence, camera_id, clip, occurred_at) -> None:
        body = {
            "type": incident_type,
            "occurred_at": _iso(occurred_at),
            "with_clip": clip is not None,
            # Même clé à chaque essai : si une requête a abouti sans que la réponse arrive,
            # l'API ne crée pas une seconde alerte (et ne prévient pas les contacts deux fois)
            "idempotency_key": str(uuid.uuid4()),
        }
        if confidence is not None:
            body["confidence"] = round(max(0.0, min(1.0, float(confidence))), 3)
        if camera_id:
            body["camera_id"] = camera_id

        # 1. Créer l'alerte : c'est elle qui prévient les proches, le clip vient ensuite
        result = None
        for delay in RETRY_DELAYS_S:
            time.sleep(delay)
            try:
                result = self._request("POST", "/device/alerts", body)
                break
            except CloudError as error:
                if not self.paired or not error.retryable:
                    # Refus définitif (données invalides, boîtier révoqué) : réessayer ne sert à rien
                    log.error("Alerte %s refusée : %s", incident_type, error)
                    return
                log.warning("Alerte %s non envoyée (%s), nouvel essai…", incident_type, error)
            except Exception:
                log.exception("Alerte %s : erreur inattendue, nouvel essai…", incident_type)
        if result is None:
            log.error("Alerte %s abandonnée après %d essais", incident_type, len(RETRY_DELAYS_S))
            return

        alert_id = result["alert_id"]
        self.last_alert = {"id": alert_id, "type": incident_type, "at": time.time(), "clip": False}
        log.info("Alerte %s envoyée (%s)", incident_type, alert_id)

        # 2. Envoyer le clip vidéo
        upload = result.get("clip_upload")
        if clip is None or not upload:
            return
        path = None
        try:
            path = clip.export_mp4()
            with open(path, "rb") as f:
                response = self._http.put(
                    upload["signed_url"],
                    data=f,
                    headers={"Content-Type": "video/mp4"},
                    timeout=120,
                )
            if not response.ok:
                raise CloudError(f"Envoi du clip refusé ({response.status_code})")
            self._request("POST", f"/device/alerts/{alert_id}/clip")
            self.last_alert["clip"] = True
            log.info("Clip de l'alerte %s envoyé", alert_id)
        except Exception as error:
            log.error("Clip de l'alerte %s non envoyé : %s", alert_id, error)
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ interface locale

    def status(self) -> dict:
        camera = self.active_camera()
        return {
            "paired": self.paired,
            "device_id": self.device_id,
            "api_url": self.api_url,
            "seconds_since_contact": (
                round(time.time() - self.last_contact) if self.last_contact else None
            ),
            "error": self.last_error,
            "camera": {"id": camera["id"], "name": camera["name"]} if camera else None,
            "camera_count": len(self.cameras),
            "last_alert": (
                {**self.last_alert, "seconds_ago": round(time.time() - self.last_alert["at"])}
                if self.last_alert
                else None
            ),
        }
