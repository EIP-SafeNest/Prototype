"""
Mot de passe de l'interface locale du boîtier (http://safenest.local).

- Premier accès : la personne qui installe le boîtier choisit le mot de passe (/setup).
- Ensuite, les pages, le flux vidéo, l'API locale et le temps réel (Socket.IO)
  exigent d'être connecté. La session dure 30 jours.
- 5 essais au plus par adresse IP toutes les 5 minutes.
- Formulaires refusés s'ils sont envoyés depuis un autre site (même origine obligatoire).

Mot de passe oublié : sur le boîtier (SSH ou clavier), supprimer
~/.config/safenest/admin.json puis recharger la page pour en choisir un nouveau.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from datetime import timedelta
from urllib.parse import urlparse

from flask import jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

MIN_LENGTH = 8
MAX_FAILURES = 5
FAILURE_WINDOW_S = 300
# Pages accessibles sans être connecté
PUBLIC_ENDPOINTS = {"login", "setup"}

_failures: dict[str, list[float]] = {}


def _config_dir() -> str:
    return os.environ.get("SAFENEST_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".config", "safenest"
    )


def _admin_file() -> str:
    return os.path.join(_config_dir(), "admin.json")


def _write_private(path: str, content: str) -> None:
    """Écrit un fichier lisible par l'utilisateur seulement (600)."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _session_secret() -> str:
    """Clé de signature des cookies de session, créée au premier démarrage."""
    path = os.path.join(_config_dir(), "session_secret")
    try:
        with open(path, encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    _write_private(path, key)
    return key


def _admin() -> dict | None:
    try:
        with open(_admin_file(), encoding="utf-8") as f:
            data = json.load(f)
        return data if data.get("password_hash") else None
    except (OSError, ValueError):
        return None


def is_logged_in() -> bool:
    """
    Vrai si la session correspond au mot de passe actuel : changer ou réinitialiser
    le mot de passe déconnecte toutes les sessions ouvertes.
    """
    admin = _admin()
    return bool(admin) and session.get("auth") == admin.get("session_version")


def _same_origin() -> bool:
    origin = request.headers.get("Origin")
    return not origin or urlparse(origin).netloc == request.host


def _start_session(version: str) -> None:
    session.clear()
    session.permanent = True
    session["auth"] = version


def _too_many_failures(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _failures.get(ip, []) if now - t < FAILURE_WINDOW_S]
    _failures[ip] = recent
    return len(recent) >= MAX_FAILURES


def init_auth(app) -> None:
    app.secret_key = _session_secret()
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        # Le cookie n'accompagne pas les requêtes envoyées par un autre site
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    )

    @app.before_request
    def require_login():
        if request.endpoint in PUBLIC_ENDPOINTS or is_logged_in():
            return None
        if request.path.startswith("/api/") or request.path == "/video_feed":
            return jsonify({"error": "Connexion requise"}), 401
        return redirect(url_for("login" if _admin() else "setup"))

    @app.route("/setup", methods=["GET", "POST"])
    def setup():
        if _admin():
            return redirect(url_for("login"))
        error = None
        if request.method == "POST":
            if not _same_origin():
                return "Action refusée", 403
            password = request.form.get("password", "")
            if len(password) < MIN_LENGTH:
                error = f"Le mot de passe doit contenir au moins {MIN_LENGTH} caractères."
            elif password != request.form.get("confirm", ""):
                error = "Les deux mots de passe ne correspondent pas."
            else:
                version = secrets.token_hex(8)
                _write_private(_admin_file(), json.dumps({
                    "password_hash": generate_password_hash(password),
                    "session_version": version,
                }))
                _start_session(version)
                return redirect("/")
        return render_template("login.html", mode="setup", error=error, min_length=MIN_LENGTH), (
            400 if error else 200
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        admin = _admin()
        if not admin:
            return redirect(url_for("setup"))
        if is_logged_in():
            return redirect("/")
        error, status = None, 200
        if request.method == "POST":
            if not _same_origin():
                return "Action refusée", 403
            ip = request.remote_addr or "inconnue"
            if _too_many_failures(ip):
                error, status = "Trop d'essais. Patientez 5 minutes puis réessayez.", 429
            elif check_password_hash(admin["password_hash"], request.form.get("password", "")):
                _failures.pop(ip, None)
                _start_session(admin["session_version"])
                return redirect("/")
            else:
                _failures.setdefault(ip, []).append(time.time())
                error, status = "Mot de passe incorrect.", 401
        return render_template("login.html", mode="login", error=error, min_length=MIN_LENGTH), status

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))
