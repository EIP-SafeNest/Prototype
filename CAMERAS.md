# 📡 Caméras : utiliser un Mac, un PC Linux ou un téléphone

Le Raspberry Pi n'a pas besoin d'avoir une caméra branchée : il peut lire le flux vidéo
d'un autre appareil du réseau et le **détecte tout seul**.

```
 Mac / Linux / téléphone ──(flux MJPEG, port 8081/8080/4747)──▶ Raspberry Pi (web_app_rpi.py)
                                                                 └─ interface http://<pi>:5001
```

## 1. Partager la caméra d'un Mac ou d'un PC Linux

Depuis le dossier `Prototype` :

```bash
chmod +x lancer_camera.sh   # une seule fois
./lancer_camera.sh
```

Au premier lancement, le script installe OpenCV dans `.venv-camera/` (≈ 1 min).
Ensuite, il démarre directement. Ctrl+C pour arrêter.

Options utiles :

| Option | Rôle | Défaut |
|---|---|---|
| `--camera 1` | Choisir une autre caméra (ex. iPhone via Continuity Camera sur Mac, souvent 1) | `0` |
| `--name "Salon"` | Nom affiché dans l'interface SafeNest | nom de l'ordi |
| `--port 8082` | Port HTTP | `8081` |
| `--fps 10` / `--width 640` / `--quality 60` | Alléger le flux | `15` / `960` / `70` |

Pour vérifier que ça marche : ouvre `http://localhost:8081` sur l'ordi, tu dois voir ta caméra.

**Premier lancement**

- **Mac** : autoriser l'accès Caméra pour le Terminal, puis cliquer sur *Autoriser* si macOS
  demande d'accepter les connexions entrantes pour Python.
- **Linux** : il faut `python3-venv` (`sudo apt install python3-venv`) et être dans le groupe `video`.

## 2. Partager la caméra d'un téléphone

- **Android** : appli *IP Webcam* (port 8080) ou *DroidCam* (port 4747), puis « Démarrer le serveur ».
- **iPhone** : le plus simple est de passer par un Mac (Continuity Camera) avec
  `./lancer_camera.sh --camera 1`.

## 3. Choisir la caméra sur le Raspberry Pi

1. Sur le Pi : `./run.sh`
2. Ouvrir l'interface `http://<IP-du-Pi>:5001`
3. Cliquer sur ⚙️ : le scan se lance tout seul (quelques secondes). 🔄 pour rescanner.
4. Choisir la caméra dans **Source vidéo**. Si la détection tournait, elle redémarre automatiquement sur la nouvelle caméra.

Le champ **Réglage manuel** permet toujours de saisir une URL à la main
(ex. `http://192.168.1.20:8081/video`).

## Comment fonctionne le scan

`camera_scanner.py` (appelé par la route `GET /api/cameras`) cherche :

- **Webcams USB du Pi** : `/dev/video*` (les nœuds internes du Pi 5 sont ignorés) ;
- **Flux réseau** sur les ports 8081, 8080 et 4747, sur :
  - le réseau local du Pi (sous-réseau /24),
  - les appareils **Tailscale** en ligne (`tailscale status`), donc ça marche aussi à distance.

Un appareil lancé avec `lancer_camera.sh` s'annonce sur `/info`, ce qui permet d'afficher son nom.
Les autres sont reconnus s'ils répondent un flux MJPEG sur `/video` ou `/mjpegfeed`.

Si le flux se coupe plus de 5 s, le Pi se reconnecte tout seul.

## Dépannage

| Problème | Piste |
|---|---|
| Aucune caméra trouvée | Le script tourne-t-il sur l'ordi ? Pi et ordi sur le même Wi-Fi ou sur Tailscale ? Pare-feu de l'ordi ? |
| `Impossible d'ouvrir la caméra 0` | Autorisation Caméra (Mac) ou webcam non branchée (Linux). Essayer `--camera 1`. |
| Image saccadée | Réduire avec `--fps 8 --width 640`. |
| Le scan ne voit pas les appareils Tailscale | Vérifier `tailscale status` sur le Pi. |

## Fichiers

| Fichier | Rôle |
|---|---|
| `lancer_camera.sh` | Lanceur Mac/Linux (installe et démarre le partage) |
| `mac_camera_stream.py` | Serveur qui diffuse la caméra (`/video`, `/info`) |
| `camera_scanner.py` | Scan des caméras locales et réseau |
| `web_app_rpi.py` | Route `/api/cameras` + boucle vidéo qui se reconnecte |
| `templates/index.html` | Liste « Source vidéo » dans les paramètres |
