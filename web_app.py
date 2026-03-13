#!/usr/bin/env python3
"""
Web Interface for Fire Detection System
- Real-time video streaming
- Fire detection with YOLOv8
- WhatsApp notifications via Twilio
- Multithreaded for maximum performance
"""

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from datetime import datetime
from twilio.rest import Client
from ultralytics import YOLO
import os
import traceback
import requests as http_requests  # for image upload

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration (modifiable via interface)
class Config:
    video_url = "http://admin:admin@10.74.255.99:8081/video"
    camera_index = 0
    analysis_fps = 10
    fire_sensitivity = 70  # Plus haut = détecte plus facilement (0-100)
    # Filtres d'image
    auto_exposure = True      # Correction automatique exposition
    reduce_glare = True       # Réduire éblouissement soleil
    gamma_correction = 0.8    # < 1 = assombrir, > 1 = éclaircir

config = Config()

# Twilio WhatsApp
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "08fcabd66aa4d6370f48b70abd9a1f40"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Sandbox
DESTINATION_WHATSAPP = "whatsapp:+33684466232"

NOTIFICATION_COOLDOWN = 3

# URL publique du serveur (ngrok, etc.) — nécessaire pour envoyer les images via Twilio
# Ex: "https://abc123.ngrok.io"  — laisser vide si pas de tunnel public
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")

# Dossier captures
CAPTURES_DIR = os.path.join(os.path.dirname(__file__), "captures")

# Load YOLO model (forced: YOLO11s fire model)
print("[INFO] Chargement du modèle YOLO...")
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "runs",
    "detect",
    "runs",
    "safenest-fire11",
    "weights",
    "best.pt",
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[FATAL] Modèle requis introuvable: {MODEL_PATH}")

yolo_model = YOLO(MODEL_PATH)
FIRE_CLASS_NAMES = ["fire", "flame", "smoke"]
print(f"[INFO] Modèle forcé chargé: {MODEL_PATH}")

# Temporal smoothing: require N consecutive positive frames before alerting
FIRE_CONFIRM_FRAMES = 3   # Must detect fire N frames in a row
FIRE_CLEAR_FRAMES   = 5   # Must be clear N frames in a row to reset

# Minimum bbox area as % of frame — 0.5% filters noise but allows small flames
MIN_BBOX_AREA_RATIO = 0.005  # 0.5% of total frame area

# Global state
class DetectionState:
    def __init__(self):
        self.running = False
        self.fire_detected = False
        self.fire_bbox = None
        self.last_notification_time = 0
        self.frame_count = 0
        self.detection_count = 0
        self.current_frame = None
        self.current_frame_raw = None  # Raw CV2 frame for captures
        self.status = "Arrêté"
        self.cap = None
        # Temporal smoothing counters
        self.consecutive_fire = 0      # consecutive frames WITH fire
        self.consecutive_clear = 0     # consecutive frames WITHOUT fire
        # Multithreading
        self.frame_queue = Queue(maxsize=2)  # Buffer minimal pour réduire latence
        self.analysis_executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.last_frame_time = 0
        
state = DetectionState()
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Taille d'analyse - plus grand = meilleure détection
ANALYSIS_SIZE = (960, 720)  # Bon compromis vitesse/précision

# CLAHE pour correction d'exposition
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_frame(frame):
    """Retourne le frame sans modification."""
    return frame


# ── Motion Analyzer ──────────────────────────────────────────
class MotionAnalyzer:
    """
    Analyse les différences entre frames pour distinguer le vrai feu
    (qui scintille / bouge / grandit) des objets statiques (briquet, vêtement).
    
    Principe :
    1. On maintient un arrière-plan (background) mis à jour lentement
    2. Chaque frame est comparé au background → masque de mouvement
    3. Une détection YOLO n'est validée que si sa bbox contient du mouvement
    """
    
    def __init__(self):
        self.background = None
        self.prev_frame = None
        self.alpha = 0.02  # Vitesse d'adaptation du background (lent = stable)
        self.frame_count = 0
    
    def update(self, frame):
        """Met à jour le modèle de fond avec le nouveau frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            self.background = gray.astype(float)
            self.prev_frame = gray
            self.frame_count = 0
            return
        
        # Mise à jour progressive du background (exponential moving average)
        cv2.accumulateWeighted(gray, self.background, self.alpha)
        self.prev_frame = gray
        self.frame_count += 1
    
    def get_motion_in_bbox(self, frame, bbox):
        """
        Calcule le score de mouvement dans une bbox.
        Retourne (motion_ratio, flicker_score) :
        - motion_ratio : % de pixels qui ont bougé vs le background
        - flicker_score : intensité du scintillement (changement frame-à-frame)
        """
        if self.background is None or self.frame_count < 5:
            # Pas encore assez de frames pour comparer → on accepte
            return 1.0, 1.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        x, y, w, h = bbox
        fh, fw = gray.shape[:2]
        # Clamp & expand bbox slightly for motion context
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0, 0.0
        
        # 1) Motion vs background (long-term change)
        bg_roi = self.background[y1:y2, x1:x2].astype(np.uint8)
        curr_roi = gray[y1:y2, x1:x2]
        diff_bg = cv2.absdiff(curr_roi, bg_roi)
        _, thresh_bg = cv2.threshold(diff_bg, 30, 255, cv2.THRESH_BINARY)
        motion_ratio = np.sum(thresh_bg > 0) / thresh_bg.size
        
        # 2) Flicker vs previous frame (short-term change = scintillement)
        prev_roi = self.prev_frame[y1:y2, x1:x2]
        diff_prev = cv2.absdiff(curr_roi, prev_roi)
        _, thresh_prev = cv2.threshold(diff_prev, 15, 255, cv2.THRESH_BINARY)
        flicker_score = np.sum(thresh_prev > 0) / thresh_prev.size
        
        return motion_ratio, flicker_score
    
    def is_dynamic(self, frame, bbox):
        """
        Vérifie si la zone de la bbox contient du mouvement.
        Le vrai feu scintille TOUJOURS, un objet statique ne bouge pas.
        
        Returns: (is_moving: bool, detail: str)
        """
        motion_ratio, flicker = self.get_motion_in_bbox(frame, bbox)
        
        # Scoring
        reasons = []
        is_moving = False
        
        if motion_ratio > 0.15:
            reasons.append(f"mouvement={motion_ratio:.0%}")
        if flicker > 0.05:
            reasons.append(f"scintillement={flicker:.0%}")
        
        # Feu = mouvement ET/OU scintillement
        if motion_ratio > 0.10 or flicker > 0.03:
            is_moving = True
        
        detail = ", ".join(reasons) if reasons else f"statique(mvt={motion_ratio:.0%}, flk={flicker:.0%})"
        return is_moving, detail

# Instance globale
motion_analyzer = MotionAnalyzer()


def detect_fire_yolo(frame):
    """
    Détection de feu hybride : YOLO + analyse de mouvement.
    
    Pipeline :
    1. YOLO détecte les candidats "feu" dans le frame
    2. Pour chaque candidat, on vérifie le MOUVEMENT dans la bbox
       (comparaison avec le background et le frame précédent)
    3. Pas de mouvement = objet statique = ignoré
    4. Mouvement/scintillement = potentiellement du vrai feu = validé
    """
    original_h, original_w = frame.shape[:2]
    
    processed_frame = preprocess_frame(frame)
    
    # Resize pour accélérer l'analyse YOLO
    if original_w > ANALYSIS_SIZE[0] or original_h > ANALYSIS_SIZE[1]:
        scale_x = ANALYSIS_SIZE[0] / original_w
        scale_y = ANALYSIS_SIZE[1] / original_h
        scale = min(scale_x, scale_y)
        small_frame = cv2.resize(processed_frame, None, fx=scale, fy=scale)
    else:
        small_frame = processed_frame
        scale = 1.0
    
    # Sensibilité: 100% = seuil bas (0.15), 0% = seuil haut (0.60)
    confidence_threshold = 0.60 - (config.fire_sensitivity / 100.0 * 0.45)
    
    # YOLO inference
    results = yolo_model(small_frame, verbose=False, conf=0.12, iou=0.4, imgsz=640)
    
    if FIRE_CLASS_NAMES:
        best_conf = 0.0
        best_bbox = None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id].lower()
                
                if any(fw in cls_name for fw in FIRE_CLASS_NAMES):
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold and conf > best_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = int(x1 / scale), int(y1 / scale)
                        x2, y2 = int(x2 / scale), int(y2 / scale)
                        bbox_w, bbox_h = x2 - x1, y2 - y1
                        bbox_area = bbox_w * bbox_h
                        frame_area = original_w * original_h
                        
                        if bbox_area / frame_area < MIN_BBOX_AREA_RATIO:
                            continue
                        
                        # ── MOTION CHECK ──
                        # Est-ce que cette zone BOUGE / SCINTILLE ?
                        is_moving, motion_detail = motion_analyzer.is_dynamic(frame, (x1, y1, bbox_w, bbox_h))
                        
                        if not is_moving:
                            print(f"[MOTION] YOLO=feu (conf={conf:.0%}) mais STATIQUE → {motion_detail}")
                            continue
                        
                        print(f"[MOTION] ✅ Mouvement confirmé: {motion_detail} (conf={conf:.0%})")
                        best_conf = conf
                        best_bbox = (x1, y1, bbox_w, bbox_h)
        
        if best_bbox is not None:
            return True, best_bbox, best_conf
        return False, None, 0.0
    
    # Fallback: Color-based detection ONLY when no custom fire model
    return detect_fire_color(frame)


def detect_fire_color(frame):
    """
    Fallback: Detect fire using color analysis (HSV).
    Returns (detected, bbox, confidence).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Fire colors: bright orange/red/yellow with HIGH saturation and brightness
    lower1 = np.array([0, 150, 200])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([15, 150, 200])
    upper2 = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, None, 0.0
    
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    
    # Min area based on sensitivity
    min_area = 5000 - (config.fire_sensitivity * 45)
    
    if area > min_area:
        x, y, w, h = cv2.boundingRect(largest)
        if h / max(w, 1) >= 0.3:  # Aspect ratio check
            confidence = min(1.0, area / 10000)  # Pseudo-confidence based on size
            return True, (x, y, w, h), confidence
    
    return False, None, 0.0


def upload_image(image_bytes):
    """Upload image to a public HTTPS host for Twilio media_url."""
    
    # 1) catbox.moe — reliable, HTTPS, no API key needed
    try:
        r = http_requests.post(
            'https://catbox.moe/user/api.php',
            data={'reqtype': 'fileupload'},
            files={'fileToUpload': ('fire_alert.jpg', image_bytes, 'image/jpeg')},
            timeout=6
        )
        if r.ok and r.text.startswith('https://'):
            url = r.text.strip()
            print(f"[UPLOAD] ✅ Image catbox: {url}")
            return url
        else:
            print(f"[UPLOAD] catbox erreur: {r.status_code} — {r.text[:150]}")
    except Exception as e:
        print(f"[UPLOAD] catbox échoué: {e}")
    
    # 2) litterbox.catbox.moe — temporary (24h), same reliability
    try:
        r = http_requests.post(
            'https://litterbox.catbox.moe/resources/internals/api.php',
            data={'reqtype': 'fileupload', 'time': '24h'},
            files={'fileToUpload': ('fire_alert.jpg', image_bytes, 'image/jpeg')},
            timeout=6
        )
        if r.ok and r.text.startswith('https://'):
            url = r.text.strip()
            print(f"[UPLOAD] ✅ Image litterbox: {url}")
            return url
        else:
            print(f"[UPLOAD] litterbox erreur: {r.status_code} — {r.text[:150]}")
    except Exception as e:
        print(f"[UPLOAD] litterbox échoué: {e}")
    
    # 3) imgbb.com — free, no key needed for anonymous
    try:
        import base64
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        r = http_requests.post(
            'https://api.imgbb.com/1/upload',
            data={'key': 'a]d5765e5b4e37a2e28b92f4afbb6c3b', 'image': b64},
            timeout=6
        )
        if r.ok:
            link = r.json().get('data', {}).get('url')
            if link:
                print(f"[UPLOAD] ✅ Image imgbb: {link}")
                return link
        else:
            print(f"[UPLOAD] imgbb erreur: {r.status_code}")
    except Exception as e:
        print(f"[UPLOAD] imgbb échoué: {e}")
    
    print("[UPLOAD] ❌ Tous les services d'upload ont échoué")
    return None


def send_whatsapp_notification(frame=None):
    """Send WhatsApp notification quickly, then send image in follow-up."""
    current_time = time.time()
    if current_time - state.last_notification_time < NOTIFICATION_COOLDOWN:
        print(f"[WHATSAPP] Cooldown actif, notification ignorée")
        return False

    # Réserver immédiatement le cooldown pour éviter les doublons concurrentiels
    state.last_notification_time = current_time

    # 1) Envoyer d'abord le texte (rapide)
    try:
        text_message = twilio_client.messages.create(
            body=f"🔥 ALERTE FEU: Flammes détectées à {datetime.now().strftime('%H:%M:%S')}! Vérifiez immédiatement!",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATION_WHATSAPP
        )
        print(f"[WHATSAPP] ✅ Alerte texte envoyée: {text_message.sid}")
    except Exception:
        print(f"[ERROR] Twilio WhatsApp (texte) échoué:")
        traceback.print_exc()
        # Autoriser une nouvelle tentative rapidement si le texte n'est pas parti
        state.last_notification_time = 0
        return False

    # 2) Envoyer la photo en asynchrone (ne bloque plus l'alerte)
    def _send_image_followup(frame_for_upload):
        if frame_for_upload is None:
            return
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"fire_{timestamp}.jpg"
            image_path = os.path.join(CAPTURES_DIR, image_filename)
            cv2.imwrite(image_path, frame_for_upload)
            print(f"[CAPTURE] Image sauvegardée: {image_path}")

            _, jpg_buffer = cv2.imencode('.jpg', frame_for_upload, [cv2.IMWRITE_JPEG_QUALITY, 85])
            media_url = upload_image(jpg_buffer.tobytes())
            if not media_url:
                return

            img_message = twilio_client.messages.create(
                body=f"📸 Capture feu ({datetime.now().strftime('%H:%M:%S')})",
                from_=TWILIO_WHATSAPP_NUMBER,
                to=DESTINATION_WHATSAPP,
                media_url=[media_url]
            )
            print(f"[WHATSAPP] ✅ Image envoyée: {img_message.sid}")
        except Exception:
            print(f"[ERROR] Envoi image WhatsApp échoué:")
            traceback.print_exc()

    threading.Thread(target=_send_image_followup, args=(frame.copy() if frame is not None else None,), daemon=True).start()
    return True


def draw_detection_box(frame, detected, bbox=None):
    """Draw red/orange box around detected fire."""
    if detected and bbox:
        x, y, w, h = bbox
        # Draw box around fire position (bbox already in pixels)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 3)  # Orange
        cv2.putText(frame, "FEU!", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    elif detected:
        # Fallback: orange border around frame
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 100, 255), 4)
        cv2.putText(frame, "FEU DETECTE!", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    return frame


def analyze_frame_async(frame, frame_id):
    """Analyse un frame dans un thread séparé."""
    try:
        # Mettre à jour le modèle de fond à chaque frame
        motion_analyzer.update(frame)
        fire_detected, bbox, confidence = detect_fire_yolo(frame)
        return (fire_detected, bbox, confidence, frame_id)
    except Exception as e:
        print(f"[ERROR] Analyse frame: {e}")
        return (False, None, 0.0, frame_id)


def capture_thread():
    """Thread dédié à la capture vidéo - priorité à la faible latence."""
    video_source = config.video_url.strip() if config.video_url else None
    
    if video_source:
        print(f"[CAPTURE] Connexion à {video_source}")
        state.status = f"Connexion..."
        state.cap = cv2.VideoCapture(video_source)
        if not state.cap.isOpened():
            time.sleep(2)
            state.cap = cv2.VideoCapture(video_source)
    else:
        print(f"[CAPTURE] Webcam index {config.camera_index}")
        state.cap = cv2.VideoCapture(config.camera_index)
    
    if not state.cap.isOpened():
        for idx in [0, 1, 2]:
            state.cap = cv2.VideoCapture(idx)
            if state.cap.isOpened():
                print(f"[CAPTURE] Caméra trouvée à l'index {idx}")
                break
    
    if not state.cap.isOpened():
        state.status = "Erreur: Connexion impossible"
        state.running = False
        socketio.emit('error', {'message': 'Impossible de se connecter'})
        return
    
    # Optimisations pour réduire la latence
    state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
    state.cap.set(cv2.CAP_PROP_FPS, 30)  # Limiter FPS source
    
    print("[CAPTURE] ✅ Flux vidéo connecté!")
    state.status = "En cours..."
    
    frame_skip = 0
    
    while state.running:
        ret, frame = state.cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        
        # Skip frames si la queue est pleine (réduire latence)
        if state.frame_queue.full():
            frame_skip += 1
            if frame_skip < 3:  # Skip jusqu'à 3 frames
                continue
            # Vider la queue si trop en retard
            while not state.frame_queue.empty():
                try:
                    state.frame_queue.get_nowait()
                except:
                    break
        frame_skip = 0
        
        # Mettre le frame dans la queue (non-bloquant)
        try:
            state.frame_queue.put_nowait(frame)
        except:
            # Queue pleine, on skip ce frame
            try:
                state.frame_queue.get_nowait()  # Vider le plus ancien
                state.frame_queue.put_nowait(frame)
            except:
                pass
        
        # Afficher le frame avec filtres appliqués
        with state.lock:
            state.current_frame_raw = frame.copy()
            # Appliquer les mêmes filtres pour l'affichage
            filtered_frame = preprocess_frame(frame)
            display_frame = draw_detection_box(filtered_frame, state.fire_detected, state.fire_bbox)
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state.current_frame = buffer.tobytes()
    
    if state.cap:
        state.cap.release()


def analysis_thread():
    """Thread dédié à l'analyse YOLO - traite les frames en parallèle."""
    analysis_interval = 1.0 / config.analysis_fps
    last_analysis_time = 0
    pending_futures = []
    
    print("[ANALYSE] ✅ Thread d'analyse YOLO démarré")
    
    while state.running:
        try:
            frame = state.frame_queue.get(timeout=0.1)
        except Empty:
            continue
        
        current_time = time.time()
        
        # Limiter le FPS d'analyse
        if current_time - last_analysis_time < analysis_interval:
            continue
        
        last_analysis_time = current_time
        state.frame_count += 1
        
        # Soumettre l'analyse au pool de threads
        future = state.analysis_executor.submit(analyze_frame_async, frame.copy(), state.frame_count)
        pending_futures.append(future)
        
        # Vérifier les résultats terminés
        completed = [f for f in pending_futures if f.done()]
        for future in completed:
            pending_futures.remove(future)
            try:
                fire_detected, bbox, confidence, frame_id = future.result()
                
                with state.lock:
                    if fire_detected:
                        state.consecutive_fire += 1
                        state.consecutive_clear = 0
                        
                        # Only confirm fire after N consecutive positive frames
                        if state.consecutive_fire >= FIRE_CONFIRM_FRAMES:
                            was_already_detected = state.fire_detected
                            state.fire_detected = True
                            state.fire_bbox = bbox
                            
                            # Emit notification ONLY on the transition (clear → fire)
                            if not was_already_detected:
                                state.detection_count += 1
                                
                                # Capture the current display frame for the notification
                                notif_frame = state.current_frame_raw.copy() if state.current_frame_raw is not None else None
                                
                                # WhatsApp notification (with cooldown)
                                if current_time - state.last_notification_time > NOTIFICATION_COOLDOWN:
                                    threading.Thread(target=send_whatsapp_notification, args=(notif_frame,), daemon=True).start()
                                
                                socketio.emit('detection', {
                                    'detected': True,
                                    'message': f'🔥 FEU DÉTECTÉ! (conf: {confidence:.0%})',
                                    'count': state.detection_count,
                                    'bbox': bbox
                                })
                                print(f"[ALERT] Feu confirmé! Frame {frame_id} (conf: {confidence:.0%}, streak: {state.consecutive_fire})")
                    else:
                        state.consecutive_clear += 1
                        state.consecutive_fire = 0
                        
                        # Only clear after N consecutive negative frames
                        if state.consecutive_clear >= FIRE_CLEAR_FRAMES:
                            state.fire_detected = False
                            state.fire_bbox = None
            except Exception as e:
                print(f"[ERROR] Résultat analyse: {e}")
    
    print("[ANALYSE] Thread d'analyse arrêté")


def detection_loop():
    """Lance les threads de capture et d'analyse."""
    # Thread de capture (priorité haute - ne doit jamais bloquer)
    capture = threading.Thread(target=capture_thread, daemon=True)
    capture.start()
    
    # Thread d'analyse (peut prendre du temps)
    analysis = threading.Thread(target=analysis_thread, daemon=True)
    analysis.start()
    
    # Attendre que les threads se terminent
    capture.join()
    analysis.join()
    
    state.status = "Arrêté"
    print("[INFO] Détection arrêtée")


    if state.cap:
        state.cap.release()
    state.status = "Arrêté"


def generate_frames():
    """Generator for video streaming."""
    while True:
        if state.current_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.current_frame + b'\r\n')
        else:
            # Placeholder image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "En attente...", (200, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.json
        if 'video_url' in data:
            config.video_url = data['video_url']
        if 'camera_index' in data:
            config.camera_index = int(data['camera_index'])
        if 'analysis_fps' in data:
            config.analysis_fps = max(1, min(30, int(data['analysis_fps'])))
        if 'fire_sensitivity' in data:
            config.fire_sensitivity = max(0, min(100, int(data['fire_sensitivity'])))
        # Filtres anti-éblouissement
        if 'auto_exposure' in data:
            config.auto_exposure = bool(data['auto_exposure'])
        if 'reduce_glare' in data:
            config.reduce_glare = bool(data['reduce_glare'])
        if 'gamma_correction' in data:
            config.gamma_correction = max(0.3, min(1.5, float(data['gamma_correction'])))
        return jsonify({'success': True})
    
    return jsonify({
        'video_url': config.video_url,
        'camera_index': config.camera_index,
        'analysis_fps': config.analysis_fps,
        'fire_sensitivity': config.fire_sensitivity,
        'auto_exposure': config.auto_exposure,
        'reduce_glare': config.reduce_glare,
        'gamma_correction': config.gamma_correction
    })


@app.route('/api/stats')
def api_stats():
    return jsonify({
        'frame_count': state.frame_count,
        'detection_count': state.detection_count,
        'fire_detected': state.fire_detected,
        'status': state.status,
        'running': state.running
    })


# SocketIO events
@socketio.on('connect')
def on_connect():
    emit('status_update', {'status': state.status, 'running': state.running})


@socketio.on('start_detection')
def on_start():
    if not state.running:
        state.running = True
        state.frame_count = 0
        state.detection_count = 0
        state.fire_detected = False
        state.consecutive_fire = 0
        state.consecutive_clear = 0
        # Reset motion analyzer pour repartir d'un fond propre
        motion_analyzer.background = None
        motion_analyzer.prev_frame = None
        motion_analyzer.frame_count = 0
        thread = threading.Thread(target=detection_loop)
        thread.daemon = True
        thread.start()
        emit('status_update', {'status': 'Démarrage...', 'running': True})


@socketio.on('stop_detection')
def on_stop():
    state.running = False
    state.current_frame = None
    emit('status_update', {'status': 'Arrêté', 'running': False})


@socketio.on('test_notification')
def on_test():
    try:
        message = twilio_client.messages.create(
            body=f"🧪 TEST: Système actif - {datetime.now().strftime('%H:%M:%S')}",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATION_WHATSAPP
        )
        emit('notification_result', {'success': True, 'sid': message.sid})
    except Exception as e:
        emit('notification_result', {'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("🔥  Fire Detection Web Interface (Local YOLO)")
    print("=" * 60)
    print("Ouvrez http://localhost:5001 dans votre navigateur")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
