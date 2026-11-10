# ☁️ Connexion du boîtier à SafeNest

Le Raspberry Pi (`web_app_rpi.py`) envoie ses alertes à l'API SafeNest
(dépôt [EIP-SafeNest/SafeNest](https://github.com/EIP-SafeNest/SafeNest), contrat dans `docs/api.md`).
L'API prévient ensuite les contacts d'urgence (SMS, email) et l'app mobile (push).

```
 caméra ──► Raspberry Pi ──(alerte + clip des 3 dernières minutes)──► API SafeNest ──► SMS / email / push
              │  analyse locale, aucune vidéo envoyée en continu
              └─ interface locale http://safenest.local (appairage, réglages)
```

## Parcours d'installation

1. Dans la web app SafeNest : **Boîtiers → Ajouter un boîtier** → un code à 6 caractères s'affiche.
2. Ouvrir `http://safenest.local` (ou l'adresse IP du Pi) depuis un appareil du même réseau.
3. Carte **Compte SafeNest** : saisir le code → « Associer ».
4. Ajouter les caméras dans la web app (fiche du boîtier) : le Pi analyse la première caméra activée.
   Sans caméra dans la web app, il utilise la source réglée localement (⚙️ ou `SAFENEST_VIDEO_URL`).

Régénérer le code ou supprimer le boîtier dans la web app le dissocie : l'interface locale l'indique.

## Installer le service sur le Raspberry Pi

```bash
cd ~/Safenest/Prototype
./install.sh                      # environnement Python (une seule fois)
./deploy/install_service.sh       # ffmpeg, avahi, service « safenest » au démarrage
nano ~/.config/safenest/env       # SAFENEST_API_URL = adresse de l'API
sudo systemctl restart safenest
journalctl -u safenest -f         # journal en direct
```

Toutes les variables sont décrites dans [deploy/safenest.env.example](deploy/safenest.env.example).
Le jeton du boîtier est gardé dans `~/.config/safenest/device.json` (lisible par l'utilisateur seulement).

## Brancher un modèle de détection (ex. chute)

Dans la boucle de détection de `web_app_rpi.py`, appeler :

```python
report_incident("fall", confidence)   # chute
report_incident("fire", confidence)   # départ de feu (déjà branché)
```

La fonction gère le reste : délai minimal entre deux alertes du même type
(`SAFENEST_ALERT_COOLDOWN`), clip des dernières minutes, envoi en arrière-plan avec
nouvelles tentatives si le réseau est coupé. Elle ne bloque jamais la boucle d'analyse.

## Clips vidéo

Les images sont gardées en mémoire (JPEG 640 px, 5 images/s, `SAFENEST_CLIP_SECONDS`
secondes) et encodées en MP4 H.264 au moment de l'alerte. **ffmpeg est nécessaire** pour un
clip lisible sur iPhone et navigateur (installé par `deploy/install_service.sh`).

## Développement sans API en ligne

L'API tourne sur un Mac (`npm run dev` dans le dépôt SafeNest). Le Pi doit pouvoir la
joindre : même réseau local (`SAFENEST_API_URL=http://<ip-du-mac>:3333/v1`) ou Tailscale,
si les règles d'accès du tailnet autorisent le Pi à contacter le Mac.

Pour tester sans Pi, le simulateur du dépôt SafeNest joue le rôle du boîtier :
`npm run simulate -- pair <CODE>` puis `npm run simulate -- alert fire`.
