#!/usr/bin/env python3
"""
Web Interface for Fire Detection System
- Real-time video streaming
- Fire detection with YOLOv8
- WhatsApp notifications via Twilio
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from twilio.rest import Client
from ultralytics import YOLO
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration (modifiable via interface)
class Config:
    video_url = "http://admin:admin@10.74.255.99:8081/video"
    camera_index = 0
    analysis_fps = 10
    fire_sensitivity = 50  # Confidence threshold (0-100) -> 0.0-1.0

config = Config()

# Twilio WhatsApp
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "08fcabd66aa4d6370f48b70abd9a1f40"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Sandbox
DESTINATION_WHATSAPP = "whatsapp:+33684466232"

NOTIFICATION_COOLDOWN = 30

# Load YOLO model - will download fire detection model
print("[INFO] Chargement du modèle YOLO...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_yolov8.pt")

# Si pas de modèle fire spécifique, on utilise YOLOv8n standard
# et on détecte les objets lumineux/orange (fallback)
if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
    FIRE_CLASS_NAMES = ["fire", "flame", "smoke"]
    print(f"[INFO] Modèle fire custom chargé: {MODEL_PATH}")
else:
    # Utiliser YOLOv8n standard - on détectera via couleur + forme
    yolo_model = YOLO("yolov8n.pt")
    FIRE_CLASS_NAMES = []  # Pas de classe fire dans le modèle standard
    print("[INFO] Modèle YOLOv8n standard chargé (détection par couleur activée)")

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
        self.status = "Arrêté"
        self.cap = None
        
state = DetectionState()
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


def detect_fire_yolo(frame):
    """
    Detect fire using YOLO model.
    If fire-specific model exists, use it directly.
    Otherwise, use color-based detection as fallback.
    Returns (detected, bbox, confidence) where bbox is (x, y, w, h) in pixels.
    """
    # Sensibilité inversée: 100% = seuil bas (0.1), 0% = seuil haut (0.8)
    # Plus la sensibilité est haute, plus on détecte facilement (de loin)
    confidence_threshold = 0.8 - (config.fire_sensitivity / 100.0 * 0.7)  # 0.1 à 0.8
    
    # Run YOLO inference avec seuil bas pour ne rien rater
    results = yolo_model(frame, verbose=False, conf=0.05, iou=0.5)
    
    # If we have a fire-specific model
    if FIRE_CLASS_NAMES:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id].lower()
                
                # Check if it's fire/flame/smoke
                if any(fire_word in cls_name for fire_word in FIRE_CLASS_NAMES):
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        return True, (x1, y1, x2 - x1, y2 - y1), conf
    
    # Fallback: Color-based detection (for standard YOLO model)
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


def send_whatsapp_notification():
    """Send WhatsApp notification."""
    current_time = time.time()
    if current_time - state.last_notification_time < NOTIFICATION_COOLDOWN:
        return False
    
    try:
        message = twilio_client.messages.create(
            body=f"� ALERTE FEU: Flammes détectées à {datetime.now().strftime('%H:%M:%S')}! Vérifiez immédiatement!",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATION_WHATSAPP
        )
        state.last_notification_time = current_time
        print(f"[WHATSAPP] Notification envoyée: {message.sid}")
        return True
    except Exception as e:
        print(f"[ERROR] Twilio: {e}")
        return False


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


def detection_loop():
    """Main detection loop running in background."""
    analysis_interval = 1.0 / config.analysis_fps
    last_analysis_time = 0
    
    # Open camera
    video_source = config.video_url.strip() if config.video_url else None
    
    if video_source:
        print(f"[INFO] Connexion à {video_source}")
        state.status = f"Connexion à {video_source}..."
        state.cap = cv2.VideoCapture(video_source)
        
        # Attendre la connexion
        if not state.cap.isOpened():
            time.sleep(2)
            state.cap = cv2.VideoCapture(video_source)
    else:
        print(f"[INFO] Utilisation webcam index {config.camera_index}")
        state.cap = cv2.VideoCapture(config.camera_index)
    
    if not state.cap.isOpened() and not video_source:
        # Try other indices only for webcam
        for idx in [0, 1, 2]:
            state.cap = cv2.VideoCapture(idx)
            if state.cap.isOpened():
                print(f"[INFO] Caméra trouvée à l'index {idx}")
                break
    
    if not state.cap.isOpened():
        state.status = "Erreur: Impossible de se connecter"
        state.running = False
        socketio.emit('error', {'message': 'Impossible de se connecter à la caméra'})
        return
    
    state.status = "En cours..."
    print("[SUCCESS] Flux vidéo connecté!")
    
    while state.running:
        ret, frame = state.cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        current_time = time.time()
        
        # Détection YOLO (rapide en local)
        if current_time - last_analysis_time >= analysis_interval:
            last_analysis_time = current_time
            state.frame_count += 1
            
            # Détection YOLO
            fire_detected, bbox, confidence = detect_fire_yolo(frame)
            
            if fire_detected:
                state.fire_detected = True
                state.fire_bbox = bbox
                state.detection_count += 1
                
                # Send notification (première détection ou après cooldown)
                if state.detection_count == 1 or (current_time - state.last_notification_time > 30):
                    if send_whatsapp_notification():
                        print("[WHATSAPP] ✅ Notification envoyée!")
                
                socketio.emit('detection', {
                    'detected': True,
                    'message': f'🔥 FEU DÉTECTÉ! (conf: {confidence:.0%})',
                    'count': state.detection_count,
                    'bbox': bbox
                })
                print(f"[ALERT] Feu détecté! Frame {state.frame_count} (conf: {confidence:.0%})")
            else:
                state.fire_detected = False
                state.fire_bbox = None
        
        # Draw box if detected (avec bbox si disponible)
        display_frame = draw_detection_box(frame, state.fire_detected, state.fire_bbox)
        
        # Encode frame for streaming (plein framerate de la caméra)
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state.current_frame = buffer.tobytes()
    
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


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_detection():
    if not state.running:
        state.running = True
        state.frame_count = 0
        state.detection_count = 0
        state.fire_detected = False
        thread = threading.Thread(target=detection_loop)
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})


@app.route('/stop', methods=['POST'])
def stop_detection():
    state.running = False
    state.current_frame = None
    return jsonify({'status': 'stopped'})


@app.route('/status')
def get_status():
    return jsonify({
        'running': state.running,
        'status': state.status,
        'frame_count': state.frame_count,
        'detection_count': state.detection_count,
        'fire_detected': state.fire_detected,
        'config': {
            'video_url': config.video_url,
            'camera_index': config.camera_index,
            'analysis_fps': config.analysis_fps,
            'fire_sensitivity': config.fire_sensitivity
        }
    })


@app.route('/config', methods=['GET', 'POST'])
def handle_config():
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
        return jsonify({'status': 'updated', 'config': {
            'video_url': config.video_url,
            'camera_index': config.camera_index,
            'analysis_fps': config.analysis_fps,
            'fire_sensitivity': config.fire_sensitivity
        }})
    return jsonify({
        'video_url': config.video_url,
        'camera_index': config.camera_index,
        'analysis_fps': config.analysis_fps,
        'fire_sensitivity': config.fire_sensitivity
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🔥  Fire Detection Web Interface (Local YOLO)")
    print("=" * 60)
    print("Ouvrez http://localhost:5001 dans votre navigateur")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
