"""
Connexion du boîtier SafeNest à l'API cloud.

Contrat d'API : docs/api.md du dépôt EIP-SafeNest/SafeNest (routes /device/*).

- appairage avec le code à 6 caractères affiché dans la web app ;
- signe de vie toutes les 30 s, qui renvoie la configuration des caméras ;
- envoi des alertes (chute, feu) avec le clip vidéo des minutes précédentes ;
- file d'attente sur la carte SD : une alerte et son clip survivent à une coupure
  réseau ou à un redémarrage du boîtier, et repartent dès que l'API répond.

Le jeton du boîtier est gardé dans ~/.config/safenest/device.json et les alertes en
attente dans ~/.config/safenest/outbox/ (lisibles par l'utilisateur seulement).
Point d'entrée pour les modèles de détection :

    cloud.report_incident("fall", confidence=0.92, clip=clip_buffer.snapshot())
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import requests

log = logging.getLogger("safenest.cloud")

FIRMWARE_VERSION = "prototype-0.3"
HEARTBEAT_INTERVAL_S = 30
INCIDENT_TYPES = ("fall", "fire")
# Essais immédiats à l'envoi d'une alerte ; ensuite, elle attend le prochain signe de vie réussi
RETRY_DELAYS_S = (0, 2, 5, 10, 20, 40)
# File d'attente : 50 alertes au plus, gardées 14 jours (durée de conservation des clips)
OUTBOX_MAX_ENTRIES = 50
OUTBOX_DEFAULT_MAX_DAYS = 14

# Résultat d'un essai d'envoi d'une alerte de la file d'attente
DONE = "done"                  # envoyée (avec son clip s'il y en a un), entrée supprimée
WAITING_CLIP = "waiting_clip"  # alerte envoyée, clip encore en cours d'encodage
RETRY = "retry"                # API injoignable ou boîtier non appairé : l'entrée attend
DROPPED = "dropped"            # refus définitif : entrée supprimée
BUSY = "busy"                  # déjà en cours d'envoi par un autre fil


class CloudError(Exception):
    """Erreur de communication avec l'API (message lisible pour l'utilisateur)."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        # Vrai pour les erreurs passagères (réseau, 5xx, 429) : un nouvel essai peut réussir
        self.retryable = retryable


def _state_dir() -> str:
    return os.environ.get("SAFENEST_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "safenest"
    )


def _state_file() -> str:
    return os.path.join(_state_dir(), "device.json")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _write_json_private(path: str, data) -> None:
    """Écriture atomique, fichier créé directement en 600, forcée sur la carte SD."""
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        # Une coupure de courant juste après ne doit pas laisser un fichier vide
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _env_days(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        log.warning("%s invalide : %s jours par défaut", name, default)
        return float(default)


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    message = body.get("message") if isinstance(body, dict) else None
    if isinstance(message, list):
        message = " ; ".join(str(m) for m in message)
    return message or f"Erreur {response.status_code} de l'API"


class Outbox:
    """
    File d'attente des alertes sur la carte SD, une entrée par alerte :

        <dossier>/<idempotency_key>/alert.json   description de l'alerte et état de l'envoi
        <dossier>/<idempotency_key>/clip.mp4     clip encodé (s'il y en a un)

    Dossiers en 700 et fichiers en 600 : ils contiennent la vidéo du logement.
    """

    RECORD = "alert.json"
    CLIP = "clip.mp4"
    # Dossier sans alert.json valide (écriture interrompue) : supprimé passé ce délai
    ORPHAN_GRACE_S = 600

    def __init__(self, directory: str, max_entries: int = OUTBOX_MAX_ENTRIES,
                 max_age_s: float | None = None):
        self.directory = directory
        self.max_entries = max_entries
        if max_age_s is None:
            max_age_s = _env_days("SAFENEST_OUTBOX_MAX_DAYS", OUTBOX_DEFAULT_MAX_DAYS) * 86400
        self.max_age_s = max_age_s

    def _entry_dir(self, key: str) -> str:
        return os.path.join(self.directory, key)

    def add(self, record: dict) -> None:
        """Enregistre une nouvelle alerte (avant tout envoi)."""
        key = record["idempotency_key"]
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        os.makedirs(self._entry_dir(key), mode=0o700, exist_ok=True)
        self.save(key, record)
        self.prune()

    def save(self, key: str, record: dict) -> None:
        _write_json_private(os.path.join(self._entry_dir(key), self.RECORD), record)

    def load(self, key: str) -> dict | None:
        try:
            with open(os.path.join(self._entry_dir(key), self.RECORD), encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        return record if isinstance(record, dict) else None

    def attach_clip(self, key: str, source: str) -> None:
        """Déplace le clip encodé dans l'entrée (copie si /tmp est en mémoire vive)."""
        target = os.path.join(self._entry_dir(key), self.CLIP)
        tmp = f"{target}.tmp"
        shutil.move(source, tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)

    def clip_path(self, key: str) -> str | None:
        path = os.path.join(self._entry_dir(key), self.CLIP)
        return path if os.path.isfile(path) else None

    def remove(self, key: str) -> None:
        shutil.rmtree(self._entry_dir(key), ignore_errors=True)

    def _entries(self) -> list[tuple[float, str, dict | None]]:
        try:
            names = os.listdir(self.directory)
        except OSError:
            return []
        entries = []
        for name in names:
            if _is_uuid(name) and os.path.isdir(self._entry_dir(name)):
                record = self.load(name)
                created = float(record.get("created_at") or 0) if record else 0.0
                entries.append((created, name, record))
        return entries

    def keys(self) -> list[str]:
        """Alertes en attente, de la plus ancienne à la plus récente."""
        return [name for _, name, record in sorted(self._entries()) if record is not None]

    def prune(self, now: float | None = None) -> list[str]:
        """Supprime les entrées trop anciennes, puis les plus anciennes au-delà du maximum."""
        now = time.time() if now is None else now
        removed, kept = [], []
        for created, name, record in self._entries():
            if record is None:
                try:
                    orphan = now - os.path.getmtime(self._entry_dir(name)) > self.ORPHAN_GRACE_S
                except OSError:
                    orphan = False
                if orphan:
                    self.remove(name)
                    removed.append(name)
            elif now - created > self.max_age_s:
                log.warning(
                    "Alerte %s en attente depuis plus de %d jours : supprimée du boîtier",
                    record.get("type"), round(self.max_age_s / 86400),
                )
                self.remove(name)
                removed.append(name)
            else:
                kept.append((created, name))
        kept.sort()
        for _, name in kept[: max(0, len(kept) - self.max_entries)]:
            log.warning("File d'attente pleine (%d alertes) : la plus ancienne est supprimée",
                        self.max_entries)
            self.remove(name)
            removed.append(name)
        return removed


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
        self.outbox = Outbox(os.path.join(_state_dir(), "outbox"))

        self._lock = threading.Lock()
        self._state = self._load_state()
        self._camera_listeners: list[Callable[[list[dict]], None]] = []
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._outbox_wake = threading.Event()
        self._flush_lock = threading.Lock()
        # Un seul envoi à la fois par alerte (fil de l'alerte, fil du clip, reprise)
        self._entry_locks: dict[str, threading.Lock] = {}
        # Alertes suivies par leur propre fil (envoi immédiat) ou dont le clip s'encode
        self._sending: set[str] = set()
        self._encoding: set[str] = set()
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
        # Fichier créé directement en 600 : le jeton n'est jamais lisible par les autres
        _write_json_private(path, self._state)

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
        threading.Thread(target=self._outbox_loop, name="safenest-outbox", daemon=True).start()
        # Au démarrage : reprendre les alertes restées sur la carte SD
        self._outbox_wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._outbox_wake.set()

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
                    # L'API répond : c'est le moment de vider la file d'attente
                    self._outbox_wake.set()
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

        L'alerte est d'abord écrite sur la carte SD, puis envoyée ; sans réseau, elle
        repart au prochain signe de vie réussi, même après un redémarrage du boîtier.
        """
        if incident_type not in INCIDENT_TYPES:
            raise ValueError(f"Type d'incident inconnu : {incident_type}")
        if not self.paired:
            log.warning("Incident %s non transmis : boîtier non appairé", incident_type)
            return
        record = {
            "type": incident_type,
            "occurred_at": _iso(occurred_at or time.time()),
            "confidence": (
                round(max(0.0, min(1.0, float(confidence))), 3) if confidence is not None else None
            ),
            "camera_id": camera_id,
            # Même clé à chaque essai, même après un redémarrage : si une requête a abouti sans
            # que la réponse arrive, l'API ne crée pas une seconde alerte (ni de second envoi)
            "idempotency_key": str(uuid.uuid4()),
            "with_clip": clip is not None,
            # Boîtier qui a filmé : l'alerte n'est jamais transmise à un autre compte
            "device_id": self.device_id,
            "created_at": time.time(),
            # État de l'envoi
            "alert_id": None,
            "clip_sent": False,
        }
        threading.Thread(
            target=self._process_incident,
            args=(record, clip),
            name=f"safenest-alert-{incident_type}",
            daemon=True,
        ).start()

    def _process_incident(self, record: dict, clip) -> None:
        key = record["idempotency_key"]
        with self._lock:
            self._sending.add(key)
            if clip is not None:
                self._encoding.add(key)
        try:
            # 1. Sur la carte SD avant tout envoi
            try:
                self.outbox.add(record)
            except OSError as error:
                # Carte SD pleine ou passée en lecture seule : prévenir quand même, sans clip
                log.error("File d'attente indisponible (%s) : alerte %s envoyée sans clip",
                          error, record["type"])
                with self._lock:
                    self._encoding.discard(key)
                self._send_unsaved(record)
                return

            # 2. Clip encodé en parallèle : l'alerte, qui prévient les proches, n'attend pas la vidéo
            if clip is not None:
                threading.Thread(
                    target=self._encode_clip, args=(key, clip),
                    name=f"safenest-clip-{record['type']}", daemon=True,
                ).start()

            # 3. Envoi immédiat, avec quelques nouveaux essais
            for delay in RETRY_DELAYS_S:
                if self._stop.wait(delay):
                    return
                if self._deliver(key) != RETRY:
                    break
            else:
                log.warning("Alerte %s gardée sur le boîtier : nouvel essai au prochain signe de vie",
                            record["type"])
        finally:
            with self._lock:
                self._sending.discard(key)

    def _encode_clip(self, key: str, clip) -> None:
        path = None
        try:
            path = clip.export_mp4()
            self.outbox.attach_clip(key, path)
            path = None
        except Exception as error:
            log.error("Clip de l'alerte non encodé (%s) : l'alerte part sans vidéo", error)
            try:
                with self._entry_lock(key):
                    record = self.outbox.load(key)
                    if record is not None:
                        record["with_clip"] = False
                        self.outbox.save(key, record)
            except OSError:
                log.exception("File d'attente : mise à jour impossible")
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            with self._lock:
                self._encoding.discard(key)
        # L'alerte est peut-être déjà partie : envoyer le clip maintenant (ou terminer l'entrée)
        self._deliver(key, wait=True)

    def flush_outbox(self) -> int:
        """Reprend les alertes en attente, de la plus ancienne à la plus récente."""
        if not self._flush_lock.acquire(blocking=False):
            return len(self.outbox.keys())
        try:
            self.outbox.prune()
            if self.paired:
                for key in self.outbox.keys():
                    if self._stop.is_set():
                        break
                    with self._lock:
                        handled_elsewhere = key in self._sending or key in self._encoding
                    if handled_elsewhere:
                        continue
                    if self._deliver(key) == RETRY:
                        break  # API injoignable : inutile d'insister avec les suivantes
            return len(self.outbox.keys())
        finally:
            self._flush_lock.release()

    def _outbox_loop(self) -> None:
        while not self._stop.is_set():
            self._outbox_wake.wait()
            self._outbox_wake.clear()
            if self._stop.is_set():
                break
            try:
                self.flush_outbox()
            except Exception:
                log.exception("File d'attente : erreur inattendue")

    def _entry_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._entry_locks.setdefault(key, threading.Lock())

    def _deliver(self, key: str, wait: bool = False) -> str:
        """Un essai d'envoi d'une alerte de la file d'attente (voir DONE, RETRY…)."""
        lock = self._entry_lock(key)
        if not lock.acquire(blocking=wait):
            return BUSY
        try:
            return self._deliver_locked(key)
        except Exception:
            log.exception("Alerte en attente : erreur inattendue, nouvel essai plus tard")
            return RETRY
        finally:
            lock.release()

    def _deliver_locked(self, key: str) -> str:
        record = self.outbox.load(key)
        if record is None:
            return DONE  # déjà envoyée
        if not self.paired:
            return RETRY  # en attente d'un nouvel appairage
        if record.get("device_id") != self.device_id:
            # Boîtier appairé entre-temps à un autre boîtier de la web app (autre foyer) :
            # la vidéo de l'ancien logement ne doit pas partir chez le nouveau propriétaire
            log.warning("Alerte %s enregistrée sous un autre appairage : supprimée sans envoi",
                        record.get("type"))
            self._finish(key, record)
            return DROPPED

        with self._lock:
            encoding = key in self._encoding
        clip_file = self.outbox.clip_path(key)
        wants_clip = bool(record.get("with_clip")) and not record.get("clip_sent")
        if wants_clip and not clip_file and not encoding:
            log.warning("Clip de l'alerte %s perdu (redémarrage pendant l'encodage)", record["type"])
            wants_clip = False

        try:
            result = None
            if not record.get("alert_id") or (wants_clip and clip_file):
                # Création de l'alerte, ou nouvel essai (même clé) pour obtenir une URL d'envoi du clip
                result = self._request(
                    "POST", "/device/alerts", self._alert_body(record, with_clip=bool(clip_file))
                )
                if not isinstance(result, dict) or not result.get("alert_id"):
                    raise CloudError("Réponse inattendue de l'API SafeNest (alerte)", retryable=True)
                if not record.get("alert_id"):
                    record["alert_id"] = result["alert_id"]
                    self.outbox.save(key, record)
                    self.last_alert = {"id": record["alert_id"], "type": record["type"],
                                       "at": time.time(), "clip": False}
                    log.info("Alerte %s envoyée (%s)", record["type"], record["alert_id"])

            if not wants_clip:
                self._finish(key, record)
                return DONE
            if not clip_file:
                return WAITING_CLIP  # le fil d'encodage enverra le clip

            confirm = f"/device/alerts/{record['alert_id']}/clip"
            upload = result.get("clip_upload")
            if upload:
                self._upload_clip(clip_file, upload["signed_url"])
                self._request("POST", confirm)
                log.info("Clip de l'alerte %s envoyé", record["alert_id"])
            else:
                # Pas d'URL d'envoi : clip déjà confirmé lors d'un essai dont la réponse s'est
                # perdue, ou URL que l'API n'a pas pu créer. La confirmation (idempotente) tranche.
                try:
                    self._request("POST", confirm)
                except CloudError as error:
                    if error.status == 409:
                        raise CloudError("URL d'envoi du clip indisponible", retryable=True) from error
                    raise
            self._finish(key, record, clip_sent=True)
            return DONE
        except CloudError as error:
            if error.status == 401 or not self.paired:
                return RETRY  # boîtier dissocié : l'alerte attend un nouvel appairage
            if error.retryable:
                log.warning("Alerte %s en attente : %s", record["type"], error)
                return RETRY
            # Refus définitif (données invalides…) : réessayer ne sert à rien
            log.error("Alerte %s abandonnée : %s", record["type"], error)
            self._finish(key, record)
            return DROPPED

    def _alert_body(self, record: dict, with_clip: bool) -> dict:
        body = {
            "type": record["type"],
            "occurred_at": record["occurred_at"],
            "with_clip": with_clip,
            "idempotency_key": record["idempotency_key"],
        }
        if record.get("confidence") is not None:
            body["confidence"] = record["confidence"]
        # Alerte déjà créée : l'API la retrouve par sa clé. Sans camera_id, une caméra supprimée
        # entre-temps dans la web app n'empêche pas l'envoi du clip.
        if record.get("camera_id") and not record.get("alert_id"):
            body["camera_id"] = record["camera_id"]
        return body

    def _upload_clip(self, path: str, signed_url: str) -> None:
        try:
            with open(path, "rb") as f:
                response = self._http.put(
                    signed_url, data=f, headers={"Content-Type": "video/mp4"}, timeout=120
                )
        except requests.RequestException as error:
            raise CloudError("Envoi du clip interrompu (réseau)", retryable=True) from error
        if not response.ok:
            raise CloudError(
                f"Envoi du clip refusé ({response.status_code})",
                status=response.status_code,
                retryable=response.status_code >= 500 or response.status_code in (408, 429),
            )

    def _finish(self, key: str, record: dict, clip_sent: bool = False) -> None:
        if clip_sent and self.last_alert and self.last_alert.get("id") == record.get("alert_id"):
            self.last_alert["clip"] = True
        self.outbox.remove(key)
        with self._lock:
            self._entry_locks.pop(key, None)

    def _send_unsaved(self, record: dict) -> None:
        """Repli sans carte SD : l'alerte seule, avec les essais immédiats."""
        for delay in RETRY_DELAYS_S:
            if self._stop.wait(delay):
                return
            try:
                result = self._request("POST", "/device/alerts",
                                       self._alert_body(record, with_clip=False))
            except CloudError as error:
                if not error.retryable:
                    log.error("Alerte %s refusée : %s", record["type"], error)
                    return
                continue
            if isinstance(result, dict) and result.get("alert_id"):
                self.last_alert = {"id": result["alert_id"], "type": record["type"],
                                   "at": time.time(), "clip": False}
                log.info("Alerte %s envoyée (%s)", record["type"], result["alert_id"])
                return
        log.error("Alerte %s abandonnée après %d essais", record["type"], len(RETRY_DELAYS_S))

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
            # Alertes gardées sur la carte SD en attendant l'API
            "pending_alerts": len(self.outbox.keys()),
        }
