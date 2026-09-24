"""
Détection automatique des caméras disponibles pour SafeNest.

- Caméras locales : webcams USB branchées sur la machine (/dev/video*).
- Caméras réseau : flux MJPEG trouvés sur le réseau local (/24) et sur les
  appareils Tailscale connectés :
    * mac_camera_stream.py (port 8081)
    * IP Webcam, Android (port 8080)
    * DroidCam (port 4747)

Uniquement la bibliothèque standard (cv2 est optionnel, pour macOS/Windows).
"""

import ipaddress
import json
import os
import platform
import re
import socket
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KNOWN_PORTS = {
    8081: "Flux MJPEG",
    8080: "IP Webcam (Android)",
    4747: "DroidCam",
}
VIDEO_PATHS = ["/video", "/mjpegfeed"]

# Pas de proxy HTTP pour parler aux caméras du réseau local
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Nœuds vidéo internes du Raspberry Pi 5 qui ne sont pas des caméras
_IGNORED_V4L2 = re.compile(r"pispbe|rp1-cfe|hevc|bcm2835|codec|isp|unicam", re.I)


# --------------------------------------------------------------------------
# Caméras locales
# --------------------------------------------------------------------------
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def local_cameras(base="/sys/class/video4linux"):
    cams = []
    if platform.system() == "Linux":
        if not os.path.isdir(base):
            return cams
        for dev in sorted(os.listdir(base), key=lambda d: int(re.sub(r"\D", "", d) or 0)):
            m = re.fullmatch(r"video(\d+)", dev)
            if not m:
                continue
            name = _read(os.path.join(base, dev, "name")) or dev
            # Une webcam USB crée plusieurs nœuds : seul l'index 0 capture l'image
            if _read(os.path.join(base, dev, "index")) not in ("", "0"):
                continue
            if _IGNORED_V4L2.search(name):
                continue
            idx = int(m.group(1))
            cams.append({
                "id": f"local:{idx}",
                "type": "local",
                "label": f"{name} (/dev/video{idx})",
                "camera_index": idx,
                "video_url": "",
            })
    else:
        try:
            import cv2
        except ImportError:
            return cams
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cams.append({
                    "id": f"local:{idx}",
                    "type": "local",
                    "label": f"Webcam {idx}",
                    "camera_index": idx,
                    "video_url": "",
                })
            cap.release()
    return cams


# --------------------------------------------------------------------------
# Hôtes à scanner
# --------------------------------------------------------------------------
def _tailscale_peers():
    """[(ip, nom)] des appareils Tailscale en ligne, et IPs de la machine."""
    peers, own = [], set()
    try:
        out = subprocess.run(["tailscale", "status", "--json"],
                             capture_output=True, text=True, timeout=4)
        data = json.loads(out.stdout)
    except Exception:
        return peers, own
    for ip in (data.get("Self") or {}).get("TailscaleIPs") or []:
        own.add(ip)
    for peer in (data.get("Peer") or {}).values():
        if not peer.get("Online"):
            continue
        ips = [ip for ip in peer.get("TailscaleIPs") or [] if ":" not in ip]
        if ips:
            name = peer.get("HostName") or (peer.get("DNSName") or "").split(".")[0]
            peers.append((ips[0], name))
    return peers, own


def _lan_networks():
    """[(ip_locale, réseau /24)] des interfaces réseau locales."""
    nets = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            ifname, cidr = parts[1], parts[3]
            if ifname == "lo" or ifname.startswith(("tailscale", "docker", "br-", "veth", "virbr")):
                continue
            iface = ipaddress.ip_interface(cidr)
            net = iface.network if iface.network.prefixlen >= 24 else \
                ipaddress.ip_network(f"{iface.ip}/24", strict=False)
            nets.append((str(iface.ip), net))
    except Exception:
        pass
    if not nets:  # repli (macOS, etc.)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127."):
                nets.append((ip, ipaddress.ip_network(f"{ip}/24", strict=False)))
        except Exception:
            pass
    return nets


# --------------------------------------------------------------------------
# Détection des flux
# --------------------------------------------------------------------------
def _port_open(ip, port, timeout):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _identify(ip, port, host_name):
    """Renvoie la description de la caméra si (ip, port) diffuse un flux vidéo."""
    base = f"http://{ip}:{port}"

    # Script mac_camera_stream.py : il s'annonce sur /info
    try:
        with _opener.open(base + "/info", timeout=1.5) as r:
            info = json.loads(r.read(2048).decode())
        if info.get("service") == "safenest-cam":
            name = info.get("name") or host_name or ip
            return {
                "id": base + "/video",
                "type": "network",
                "label": f"{name} — caméra {info.get('camera', 0)} ({ip})",
                "camera_index": None,
                "video_url": base + "/video",
            }
    except Exception:
        pass

    # Autres applis : on vérifie qu'un flux MJPEG répond
    for path in VIDEO_PATHS:
        try:
            with _opener.open(base + path, timeout=1.5) as r:
                ctype = r.headers.get("Content-Type", "")
            if ctype.startswith(("multipart/x-mixed-replace", "image/")):
                who = host_name or ip
                return {
                    "id": base + path,
                    "type": "network",
                    "label": f"{KNOWN_PORTS.get(port, 'Flux MJPEG')} — {who} ({ip})",
                    "camera_index": None,
                    "video_url": base + path,
                }
        except Exception:
            continue
    return None


def network_cameras(connect_timeout=0.4):
    peers, own_ips = _tailscale_peers()
    nets = _lan_networks()
    own_ips.update(ip for ip, _ in nets)

    hosts = {ip: name for ip, name in peers}
    for _, net in nets:
        for h in net.hosts():
            hosts.setdefault(str(h), "")
    for ip in own_ips:
        hosts.pop(ip, None)

    targets = [(ip, port) for ip in hosts for port in KNOWN_PORTS]
    with ThreadPoolExecutor(max_workers=128) as pool:
        opened = [t for t, ok in zip(targets, pool.map(
            lambda t: _port_open(t[0], t[1], connect_timeout), targets)) if ok]
        found = list(pool.map(lambda t: _identify(t[0], t[1], hosts[t[0]]), opened))
    return [c for c in found if c]


def scan_cameras():
    return local_cameras() + network_cameras()


if __name__ == "__main__":
    for cam in scan_cameras():
        print(f"- {cam['label']}  ->  {cam['video_url'] or 'index ' + str(cam['camera_index'])}")
