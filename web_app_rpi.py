#!/usr/bin/env python3
"""
SafeNest - application du boîtier (Raspberry Pi).

- analyse le flux vidéo en local (YOLO via OpenCV DNN, sans PyTorch) ;
- interface locale : http://safenest.local (appairage, réglages, image en direct) ;
- envoie les alertes à l'API SafeNest avec le clip des dernières minutes
  (voir safenest_cloud.py et CLOUD.md).

Pour brancher un nouveau modèle (ex. détection de chute) : appeler
report_incident("fall", confidence) depuis la boucle de détection.
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import logging
import numpy as np
from datetime import datetime
import os
import urllib.request
from urllib.parse import urlparse

from safenest_cloud import SafeNestCloud, CloudError
from clip_buffer import ClipBuffer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s : %(message)s")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration (surchargeable par variables d'environnement, cf. deploy/safenest.env.example)
class Config:
    # Source utilisée tant que le boîtier n'a pas de caméra configurée dans la web app
    video_url = os.environ.get("SAFENEST_VIDEO_URL", "http://admin:admin@10.74.255.99:8081/video")
    camera_index = 0
    analysis_fps = 5
    fire_sensitivity = 50

config = Config()

# Délai minimal entre deux alertes du même type (évite d'envoyer 10 SMS pour le même feu)
NOTIFICATION_COOLDOWN = int(os.environ.get("SAFENEST_ALERT_COOLDOWN", "120"))
# Durée de vidéo gardée en mémoire et envoyée avec l'alerte (CdC : 3 minutes)
CLIP_SECONDS = int(os.environ.get("SAFENEST_CLIP_SECONDS", "180"))

# YOLO model paths
MODEL_DIR = os.path.dirname(__file__)
YOLO_CFG_PATH = os.path.join(MODEL_DIR, "yolov4-tiny-fire.cfg")
YOLO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov4-tiny-fire.weights")

# Global state
class DetectionState:
    def __init__(self):
        self.running = False
        self.fire_detected = False
        self.fire_bbox = None
        self.frame_count = 0
        self.detection_count = 0
        self.current_frame = None
        self.status = "Arrêté"
        self.cap = None
        self.net = None
        self.output_layers = None
        self.classes = ["fire", "smoke"]
        self.use_yolo = False
        self.run_id = 0
        # Caméra de la web app en cours d'analyse (None = source locale)
        self.active_camera_id = None
        self.source_signature = None
        self.last_frame_time = 0
        self.source_changed = False
        self.mode = ""
        self.last_incident_time = {}
        self.last_unpaired_warning = 0

state = DetectionState()


def camera_statuses():
    """État de la caméra analysée, transmis à l'API à chaque signe de vie."""
    if not state.running or not state.active_camera_id:
        return []
    ok = time.time() - state.last_frame_time < 10
    return [{"id": state.active_camera_id, "status": "ok" if ok else "error"}]


cloud = SafeNestCloud(camera_status_provider=camera_statuses)
clips = ClipBuffer(seconds=CLIP_SECONDS)


def on_cloud_cameras_changed(cameras):
    """
    Caméras modifiées dans la web app : la boucle de détection rouvre la source,
    mais seulement si la caméra analysée ou son URL a changé (le statut, lui,
    change à chaque signe de vie).
    """
    url, camera_id, _ = current_source()
    if (url, camera_id) != state.source_signature:
        state.source_changed = True


cloud.on_cameras_changed(on_cloud_cameras_changed)


def current_source():
    """
    Source à analyser : la caméra configurée dans la web app, sinon la configuration
    locale (URL réseau, ou index de webcam). Renvoie (source, id caméra web app, nom).
    """
    camera = cloud.active_camera()
    if camera:
        return camera["stream_url"], camera["id"], camera["name"]
    if config.video_url and config.video_url.strip():
        return config.video_url.strip(), None, "Configuration locale"
    return int(config.camera_index), None, f"Webcam {config.camera_index}"


def download_model():
    """Download YOLOv4-tiny fire detection model for OpenCV DNN."""
    # Fire detection model trained on fire/smoke dataset
    cfg_url = "https://raw.githubusercontent.com/gengyanlei/fire-smoke-detect-yolov4/master/cfg/yolov4-tiny-obj.cfg"
    weights_url = "https://github.com/gengyanlei/fire-smoke-detect-yolov4/releases/download/v1.0/yolov4-tiny-obj_last.weights"
    
    if not os.path.exists(YOLO_CFG_PATH):
        print(f"[DOWNLOAD] Configuration YOLO...")
        urllib.request.urlretrieve(cfg_url, YOLO_CFG_PATH)
        print(f"[OK] {YOLO_CFG_PATH}")
    
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print(f"[DOWNLOAD] Poids YOLO (~23MB, patientez)...")
        urllib.request.urlretrieve(weights_url, YOLO_WEIGHTS_PATH)
        print(f"[OK] {YOLO_WEIGHTS_PATH}")


def load_yolo_model():
    """Load YOLO model using OpenCV DNN (no PyTorch needed!)."""
    print("[INFO] Chargement YOLO avec OpenCV DNN...")
    
    # Download if needed
    if not os.path.exists(YOLO_WEIGHTS_PATH) or not os.path.exists(YOLO_CFG_PATH):
        try:
            download_model()
        except Exception as e:
            print(f"[WARNING] Download failed: {e}")
            return None, None
    
    try:
        # Load with OpenCV DNN - works on ARM!
        net = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
        
        # CPU backend for Raspberry Pi
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layers
        layer_names = net.getLayerNames()
        unconnected = net.getUnconnectedOutLayers()
        # Handle both OpenCV 4.x formats
        if isinstance(unconnected[0], np.ndarray):
            output_layers = [layer_names[i[0] - 1] for i in unconnected]
        else:
            output_layers = [layer_names[i - 1] for i in unconnected]
        
        print("[OK] YOLO chargé avec OpenCV DNN!")
        return net, output_layers
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None


def detect_fire_yolo_dnn(frame, net, output_layers):
    """
    Detect fire using YOLO with OpenCV DNN.
    Real neural network inference!
    """
    height, width = frame.shape[:2]
    
    # Confidence threshold
    conf_threshold = 0.8 - (config.fire_sensitivity / 100.0 * 0.6)
    
    # Create blob (416x416 for YOLOv4-tiny)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Forward pass
    outputs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                # Scale back to frame size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-Maximum Suppression
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
        if len(indices) > 0:
            idx = indices[0] if isinstance(indices[0], int) else indices[0][0]
            x, y, w, h = boxes[idx]
            label = state.classes[class_ids[idx]] if class_ids[idx] < len(state.classes) else "fire"
            return True, (x, y, w, h), confidences[idx], label
    
    return False, None, 0.0, None


def detect_fire_color(frame):
    """Fallback: HSV color-based detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower1 = np.array([0, 120, 180])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([15, 120, 180])
    upper2 = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = 3000 - (config.fire_sensitivity * 28)
        
        if area > max(min_area, 200):
            x, y, w, h = cv2.boundingRect(largest)
            confidence = min(1.0, area / 8000)
            return True, (x, y, w, h), confidence, "fire"
    
    return False, None, 0.0, None


def report_incident(incident_type, confidence=None):
    """
    Signale un incident au cloud SafeNest, qui prévient les contacts d'urgence
    (SMS, email, push) et reçoit le clip des dernières minutes.

    Point d'entrée pour tous les modèles de détection :
        report_incident("fire", 0.87)   # départ de feu
        report_incident("fall", 0.92)   # chute (modèle de l'équipe IA)

    Renvoie True si l'alerte part, False si elle est ignorée (délai minimal ou boîtier non appairé).
    """
    now = time.time()
    if now - state.last_incident_time.get(incident_type, 0) < NOTIFICATION_COOLDOWN:
        return False
    if not cloud.paired:
        # Un message par minute au plus (la détection tourne plusieurs fois par seconde)
        if now - state.last_unpaired_warning > 60:
            state.last_unpaired_warning = now
            print(f"[CLOUD] {incident_type} détecté mais boîtier non appairé : aucune alerte envoyée")
        return False

    state.last_incident_time[incident_type] = now
    cloud.report_incident(
        incident_type,
        confidence=confidence,
        camera_id=state.active_camera_id,
        clip=clips.snapshot(),
        occurred_at=now,
    )
    socketio.emit('cloud_alert', {'type': incident_type, 'timestamp': datetime.now().strftime('%H:%M:%S')})
    return True


def draw_detection_box(frame, detected, bbox=None, label="FEU"):
    """Draw bounding box."""
    if detected and bbox:
        x, y, w, h = bbox
        color = (0, 100, 255) if "fire" in label.lower() else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label.upper(), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


STREAM_TIMEOUT_S = 5


def open_capture():
    """Ouvre la source vidéo (caméra de la web app, URL réseau locale ou webcam)."""
    source, camera_id, camera_name = current_source()
    signature = (source, camera_id)
    if signature != state.source_signature:
        # Autre caméra (autres dimensions) : ses images d'avant ne vont pas dans le clip.
        # Une simple reconnexion à la même caméra garde, elle, les minutes déjà en mémoire.
        clips.clear()
    state.active_camera_id = camera_id
    state.source_signature = signature
    state.source_changed = False
    print(f"[VIDEO] {camera_name}")
    if isinstance(source, str):
        try:
            # Délais max pour ne jamais rester bloqué sur un flux réseau
            return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
        except Exception:
            return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def reopen_capture(cap):
    """Ferme le flux et rouvre la source courante (reconnexion ou changement de caméra)."""
    cap.release()
    cap = open_capture()
    state.cap = cap
    if not cap.isOpened():
        set_video_status(False)
    return cap


def set_video_status(receiving):
    """Statut affiché : « Actif » si la caméra envoie des images, sinon l'erreur."""
    status = f"Actif ({state.mode})" if receiving else "Caméra injoignable, nouvel essai…"
    if status != state.status:
        state.status = status
        socketio.emit('status_update', {'status': status, 'running': True})


def detection_loop():
    """Main detection loop."""
    # Load YOLO
    state.net, state.output_layers = load_yolo_model()
    state.use_yolo = state.net is not None
    
    state.mode = "YOLO DNN" if state.use_yolo else "Couleur HSV"
    print(f"[MODE] Détection: {state.mode}")
    
    # Chaque démarrage a son propre identifiant : une ancienne boucle encore
    # bloquée sur une lecture s'arrête d'elle-même au lieu de partager la caméra.
    my_run = state.run_id

    def still_mine():
        return state.running and state.run_id == my_run

    # Le boîtier peut démarrer avant la caméra : en cas d'échec on réessaie
    # indéfiniment au lieu d'abandonner (la boucle tourne en service au démarrage).
    cap = open_capture()
    state.cap = cap
    if not cap.isOpened():
        print("[VIDEO] Caméra injoignable, nouvel essai dans quelques secondes...")
        set_video_status(False)

    analysis_interval = 1.0 / config.analysis_fps
    last_analysis_time = 0
    last_label = "fire"
    last_frame_time = time.time()

    while still_mine():
        # Caméra changée dans la web app (ou appairage) : on bascule tout de suite
        if state.source_changed:
            print("[VIDEO] Changement de caméra")
            cap = reopen_capture(cap)
            last_frame_time = time.time()

        ret, frame = cap.read() if cap.isOpened() else (False, None)
        if not still_mine():
            break
        if not ret:
            # Flux coupé, figé ou jamais ouvert : on se reconnecte
            if time.time() - last_frame_time > STREAM_TIMEOUT_S:
                print("[VIDEO] Plus d'image, reconnexion...")
                set_video_status(False)
                time.sleep(1)
                cap = reopen_capture(cap)
                last_frame_time = time.time()
            else:
                time.sleep(0.1)
            continue
        last_frame_time = time.time()
        state.last_frame_time = last_frame_time
        set_video_status(True)
        clips.add(frame)

        state.frame_count += 1
        current_time = time.time()

        if current_time - last_analysis_time >= analysis_interval:
            last_analysis_time = current_time
            
            # Run detection
            if state.use_yolo:
                detected, bbox, conf, label = detect_fire_yolo_dnn(
                    frame, state.net, state.output_layers)
            else:
                detected, bbox, conf, label = detect_fire_color(frame)
            
            state.fire_detected = detected
            state.fire_bbox = bbox
            if label:
                last_label = label
            
            if detected:
                state.detection_count += 1
                socketio.emit('detection', {
                    'detected': True,
                    'confidence': float(conf),
                    'label': label,
                    'method': 'YOLO' if state.use_yolo else 'Color',
                    'count': state.detection_count,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                # Flammes et fumée sont toutes deux des alertes « départ de feu »
                report_incident("fire", float(conf))
        
        display_frame = draw_detection_box(
            frame.copy(), state.fire_detected, state.fire_bbox, last_label)
        state.current_frame = display_frame
        time.sleep(0.01)
    
    cap.release()
    if state.run_id == my_run:
        state.status = "Arrêté"
        socketio.emit('status_update', {'status': state.status, 'running': False})


def generate_frames():
    """Video stream generator."""
    while True:
        if state.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', state.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "En attente...", (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
def api_cameras():
    """Scanne les caméras locales et réseau (Tailscale + réseau local)."""
    try:
        from camera_scanner import scan_cameras
        cameras = scan_cameras()
    except Exception as e:
        return jsonify({'cameras': [], 'error': str(e)}), 500
    return jsonify({
        'cameras': cameras,
        'current': {'video_url': config.video_url or '', 'camera_index': config.camera_index},
    })

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
        # Nouvelle source locale : prise en compte tout de suite si la détection tourne
        if current_source()[:2] != state.source_signature:
            state.source_changed = True
        return jsonify({'success': True})
    
    return jsonify({
        'video_url': config.video_url,
        'camera_index': config.camera_index,
        'analysis_fps': config.analysis_fps,
        'fire_sensitivity': config.fire_sensitivity
    })

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'frame_count': state.frame_count,
        'detection_count': state.detection_count,
        'fire_detected': state.fire_detected,
        'status': state.status,
        'running': state.running,
        'model': 'YOLO DNN' if state.use_yolo else 'Color'
    })


# --- Compte SafeNest (appairage avec la web app) ------------------------------

def is_local_page_request():
    """
    Les actions du compte (associer, dissocier) ne sont acceptées que depuis
    l'interface du boîtier elle-même : JSON obligatoire (une autre page web ne
    peut pas en envoyer sans autorisation du navigateur) et même origine.
    """
    if not request.is_json:
        return False
    origin = request.headers.get('Origin')
    return not origin or urlparse(origin).netloc == request.host


def forbidden_origin():
    return jsonify({'error': "Action refusée : utilisez l'interface du boîtier."}), 403


@app.route('/api/cloud')
def api_cloud_status():
    status = cloud.status()
    _, camera_id, camera_name = current_source()
    status['source'] = {
        'name': camera_name,
        'from_web_app': camera_id is not None,
        # Faux si la surveillance tourne mais qu'aucune image n'arrive depuis 10 s
        'receiving': (not state.running) or time.time() - state.last_frame_time < 10,
    }
    return jsonify(status)

@app.route('/api/cloud/pair', methods=['POST'])
def api_cloud_pair():
    if not is_local_page_request():
        return forbidden_origin()
    if cloud.paired:
        return jsonify({'error': "Ce boîtier est déjà associé à un compte : dissociez-le d'abord."}), 409
    code = ((request.get_json(silent=True) or {}).get('code') or '').strip()
    if not code:
        return jsonify({'error': "Saisissez le code affiché dans l'application SafeNest"}), 400
    try:
        cloud.pair(code)
    except CloudError as error:
        return jsonify({'error': str(error)}), 400
    return jsonify(cloud.status())

@app.route('/api/cloud/unpair', methods=['POST'])
def api_cloud_unpair():
    if not is_local_page_request():
        return forbidden_origin()
    cloud.unpair()
    state.source_changed = True
    return jsonify(cloud.status())

@app.route('/api/cloud/test', methods=['POST'])
def api_cloud_test():
    """Vérifie la connexion à l'API (signe de vie immédiat)."""
    if not is_local_page_request():
        return forbidden_origin()
    try:
        cloud.heartbeat()
    except CloudError as error:
        return jsonify({'ok': False, 'error': str(error)}), 502
    except Exception:
        return jsonify({'ok': False, 'error': "Erreur inattendue, voir le journal du boîtier"}), 500
    return jsonify({'ok': True, **cloud.status()})


# SocketIO
@socketio.on('connect')
def on_connect():
    emit('status_update', {'status': state.status, 'running': state.running})

def start_detection():
    if not state.running:
        state.running = True
        state.run_id += 1
        state.frame_count = 0
        state.detection_count = 0
        threading.Thread(target=detection_loop, daemon=True).start()

@socketio.on('start_detection')
def on_start():
    start_detection()

@socketio.on('stop_detection')
def on_stop():
    state.running = False


if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5001"))
    print("=" * 50)
    print("SafeNest - boîtier (OpenCV DNN, sans PyTorch)")
    print(f"Interface locale : http://0.0.0.0:{port}")
    print(f"API SafeNest     : {cloud.api_url} ({'appairé' if cloud.paired else 'non appairé'})")
    print("=" * 50)
    cloud.start()
    # En service, la surveillance démarre seule (SAFENEST_AUTOSTART=0 pour la désactiver)
    if os.environ.get("SAFENEST_AUTOSTART", "1") != "0":
        start_detection()
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
