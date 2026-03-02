#!/usr/bin/env python3
"""
Web Interface for Fire Detection System - Raspberry Pi Version
Uses OpenCV DNN to run YOLO models WITHOUT PyTorch!
Real neural network inference on ARM.
"""

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from twilio.rest import Client
import os
import urllib.request

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
class Config:
    video_url = "http://admin:admin@10.74.255.99:8081/video"
    camera_index = 0
    analysis_fps = 5
    fire_sensitivity = 50

config = Config()

# Twilio WhatsApp
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "08fcabd66aa4d6370f48b70abd9a1f40"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
DESTINATION_WHATSAPP = "whatsapp:+33684466232"

NOTIFICATION_COOLDOWN = 30

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
        self.last_notification_time = 0
        self.frame_count = 0
        self.detection_count = 0
        self.current_frame = None
        self.status = "Arrêté"
        self.cap = None
        self.net = None
        self.output_layers = None
        self.classes = ["fire", "smoke"]
        self.use_yolo = False
        
state = DetectionState()
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


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


def send_whatsapp_notification():
    """Send WhatsApp alert."""
    current_time = time.time()
    if current_time - state.last_notification_time < NOTIFICATION_COOLDOWN:
        return False
    
    try:
        message = twilio_client.messages.create(
            body=f"🔥 ALERTE FEU: Détecté à {datetime.now().strftime('%H:%M:%S')}!",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATION_WHATSAPP
        )
        state.last_notification_time = current_time
        print(f"[WHATSAPP] Envoyé: {message.sid}")
        return True
    except Exception as e:
        print(f"[ERROR] Twilio: {e}")
        return False


def draw_detection_box(frame, detected, bbox=None, label="FEU"):
    """Draw bounding box."""
    if detected and bbox:
        x, y, w, h = bbox
        color = (0, 100, 255) if "fire" in label.lower() else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label.upper(), (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def detection_loop():
    """Main detection loop."""
    # Load YOLO
    state.net, state.output_layers = load_yolo_model()
    state.use_yolo = state.net is not None
    
    mode = "YOLO DNN" if state.use_yolo else "Couleur HSV"
    print(f"[MODE] Détection: {mode}")
    state.status = f"Actif ({mode})"
    socketio.emit('status_update', {'status': state.status, 'running': True})
    
    # Open video
    video_source = config.video_url.strip() if config.video_url else None
    
    if video_source:
        print(f"[VIDEO] {video_source}")
        state.cap = cv2.VideoCapture(video_source)
    else:
        print(f"[VIDEO] Webcam {config.camera_index}")
        state.cap = cv2.VideoCapture(config.camera_index)
    
    if not state.cap.isOpened():
        state.status = "Erreur vidéo"
        socketio.emit('status_update', {'status': state.status, 'running': False})
        state.running = False
        return
    
    analysis_interval = 1.0 / config.analysis_fps
    last_analysis_time = 0
    last_label = "fire"
    
    while state.running:
        ret, frame = state.cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
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
                    'confidence': conf,
                    'label': label,
                    'method': 'YOLO' if state.use_yolo else 'Color',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
                send_whatsapp_notification()
        
        display_frame = draw_detection_box(
            frame.copy(), state.fire_detected, state.fire_bbox, last_label)
        state.current_frame = display_frame
        time.sleep(0.01)
    
    if state.cap:
        state.cap.release()
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


# SocketIO
@socketio.on('connect')
def on_connect():
    emit('status_update', {'status': state.status, 'running': state.running})

@socketio.on('start_detection')
def on_start():
    if not state.running:
        state.running = True
        state.frame_count = 0
        state.detection_count = 0
        threading.Thread(target=detection_loop, daemon=True).start()

@socketio.on('stop_detection')
def on_stop():
    state.running = False

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
    print("=" * 50)
    print("🔥 Fire Detection - OpenCV DNN (No PyTorch!)")
    print("=" * 50)
    print("Utilise YOLO via OpenCV DNN")
    print("Compatible Raspberry Pi ARM")
    print("Interface: http://0.0.0.0:5001")
    print("=" * 50)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
