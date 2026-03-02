#!/usr/bin/env python3
"""
Web Interface for Hand Detection System
- Real-time video streaming
- Hand detection with red bounding box
- WhatsApp notifications via Twilio
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import base64
import time
import threading
import numpy as np
from datetime import datetime
from openai import OpenAI
from twilio.rest import Client

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration (modifiable via interface)
class Config:
    video_url = "http://admin:admin@10.74.255.99:8081/video"  # Avec authentification
    camera_index = 0  # 0 = défaut, 1 = caméra avant
    analysis_fps = 3  # FPS pour l'analyse IA (économie API)
    compression_quality = 50
    resize_scale = 0.7

config = Config()

# OpenAI
OPENAI_API_KEY = "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"

# Twilio WhatsApp
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "67907e241d8371c65d25fd1d167b77be"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
DESTINATION_WHATSAPP = "whatsapp:+33684466232"

NOTIFICATION_COOLDOWN = 30

# Global state
class DetectionState:
    def __init__(self):
        self.running = False
        self.hand_detected = False
        self.hand_bbox = None  # (x, y, w, h) en pourcentages
        self.last_notification_time = 0
        self.frame_count = 0
        self.detection_count = 0
        self.current_frame = None
        self.status = "Arrêté"
        self.cap = None
        
state = DetectionState()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


def analyze_frame_for_fire(base64_image):
    """Analyze image with OpenAI Vision API. Returns (detected, bbox) where bbox is (x, y, w, h) in percentages."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Look at this image carefully. Is there FIRE or FLAMES visible?
Look for: flames, fire, burning objects, visible combustion with orange/yellow/red flames.

If YES, respond with: YES x y w h
Where x,y is the top-left corner and w,h is width/height of the fire area, all as percentages (0-100) of the image.
Example: YES 20 30 25 35

If NO fire/flames visible, respond with: NO"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=30
        )
        answer = response.choices[0].message.content.strip()
        print(f"[AI] Response: {answer}")
        
        if answer.upper().startswith("YES"):
            parts = answer.split()
            if len(parts) >= 5:
                try:
                    x = float(parts[1]) / 100
                    y = float(parts[2]) / 100
                    w = float(parts[3]) / 100
                    h = float(parts[4]) / 100
                    return True, (x, y, w, h)
                except:
                    return True, None
            return True, None
        return False, None
    except Exception as e:
        print(f"[ERROR] OpenAI: {e}")
        return False, None


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
    if detected:
        h, w = frame.shape[:2]
        
        if bbox:
            # Draw box around fire position
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int((bbox[0] + bbox[2]) * w)
            y2 = int((bbox[1] + bbox[3]) * h)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 4)  # Orange
            cv2.putText(frame, "FEU!", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        else:
            # Fallback: orange border around frame
            cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 100, 255), 4)
            cv2.putText(frame, "FEU DETECTE!", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    return frame


def analyze_async(base64_image):
    """Run analysis in background thread."""
    detected, bbox = analyze_frame_for_fire(base64_image)
    state.hand_detected = detected
    state.hand_bbox = bbox
    
    if detected:
        state.detection_count += 1
        send_whatsapp_notification()
        socketio.emit('detection', {
            'detected': True,
            'count': state.detection_count,
            'time': datetime.now().strftime('%H:%M:%S')
        })


def detection_loop():
    """Main detection loop running in background."""
    analysis_interval = 1.0 / config.analysis_fps  # Interval pour analyse IA
    last_analysis_time = 0
    analysis_in_progress = False
    
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
        
        # Analyse IA en arrière-plan (non bloquante)
        if current_time - last_analysis_time >= analysis_interval and not analysis_in_progress:
            last_analysis_time = current_time
            state.frame_count += 1
            
            # Compress for API
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * config.resize_scale), int(h * config.resize_scale)))
            _, encoded = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, config.compression_quality])
            base64_image = base64.b64encode(encoded).decode('utf-8')
            
            # Lancer l'analyse en arrière-plan
            analysis_thread = threading.Thread(target=analyze_async, args=(base64_image,))
            analysis_thread.daemon = True
            analysis_thread.start()
        
        # Draw box if detected (avec bbox si disponible)
        display_frame = draw_detection_box(frame, state.hand_detected, state.hand_bbox)
        
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
        state.hand_detected = False
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
        'hand_detected': state.hand_detected,
        'config': {
            'video_url': config.video_url,
            'camera_index': config.camera_index,
            'analysis_fps': config.analysis_fps
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
            config.analysis_fps = max(1, min(10, int(data['analysis_fps'])))  # 1-10 fps
        return jsonify({'status': 'updated', 'config': {
            'video_url': config.video_url,
            'camera_index': config.camera_index,
            'analysis_fps': config.analysis_fps
        }})
    return jsonify({
        'video_url': config.video_url,
        'camera_index': config.camera_index,
        'analysis_fps': config.analysis_fps
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🖐️  Hand Detection Web Interface")
    print("=" * 60)
    print("Ouvrez http://localhost:5001 dans votre navigateur")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
