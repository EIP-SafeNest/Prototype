#!/usr/bin/env python3
"""
Hand Detection System with Video Stream Analysis
- Captures video from IP camera
- Compresses and reduces framerate
- Analyzes frames with OpenAI Vision API
- Sends Twilio SMS notification when hand is detected
"""

import cv2
import base64
import time
import os
from openai import OpenAI
from twilio.rest import Client
import numpy as np
from datetime import datetime

# Configuration
# Pour IP camera: "http://IP:PORT/video"
# Pour webcam Mac: mettre None et utiliser CAMERA_INDEX
VIDEO_URL = None  # Mettre l'URL ou None pour webcam locale
CAMERA_INDEX = 1  # 0 = caméra par défaut, 1 = caméra avant/alternative
TARGET_FPS = 3  # 2-5 frames per second (using 3 as middle ground)
COMPRESSION_QUALITY = 30  # JPEG quality (lower = more compression)
RESIZE_SCALE = 0.5  # Reduce image size by 50%

# OpenAI Configuration
OPENAI_API_KEY = "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"

# Twilio Configuration (WhatsApp Sandbox)
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "67907e241d8371c65d25fd1d167b77be"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox WhatsApp
DESTINATION_WHATSAPP = "whatsapp:+33684466232"  # Your WhatsApp number

# Cooldown period to avoid spam (in seconds)
NOTIFICATION_COOLDOWN = 30


class HandDetectionSystem:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        self.last_notification_time = 0
        self.frame_interval = 1.0 / TARGET_FPS
        
    def compress_frame(self, frame):
        """Compress and resize the frame to reduce data size."""
        # Resize frame
        height, width = frame.shape[:2]
        new_width = int(width * RESIZE_SCALE)
        new_height = int(height * RESIZE_SCALE)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Compress to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, COMPRESSION_QUALITY]
        _, encoded = cv2.imencode('.jpg', resized, encode_params)
        
        return encoded
    
    def frame_to_base64(self, encoded_frame):
        """Convert encoded frame to base64 string."""
        return base64.b64encode(encoded_frame).decode('utf-8')
    
    def analyze_frame_for_hand(self, base64_image):
        """Use OpenAI Vision API to detect hands in the image."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image. Is there a human hand visible in this image? Respond with only 'YES' if you can see a hand (including partially visible hands, fingers, or palm), or 'NO' if there is no hand visible. Be strict and only say YES if you are confident a hand is present."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Use low detail for faster processing
                                }
                            }
                        ]
                    }
                ],
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
            
        except Exception as e:
            print(f"[ERROR] OpenAI API error: {e}")
            return False
    
    def send_notification(self):
        """Send SMS notification via Twilio."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_notification_time < NOTIFICATION_COOLDOWN:
            remaining = int(NOTIFICATION_COOLDOWN - (current_time - self.last_notification_time))
            print(f"[INFO] Notification cooldown active. {remaining}s remaining.")
            return False
        
        try:
            message = self.twilio_client.messages.create(
                body=f"🚨 ALERTE: Une main a été détectée dans le flux vidéo à {datetime.now().strftime('%H:%M:%S')}",
                from_=TWILIO_WHATSAPP_NUMBER,
                to=DESTINATION_WHATSAPP
            )
            
            self.last_notification_time = current_time
            print(f"[SUCCESS] SMS envoyé! SID: {message.sid}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Twilio error: {e}")
            return False
    
    def run(self):
        """Main loop to capture and analyze video stream."""
        print("=" * 60)
        print("Hand Detection System - Starting...")
        print(f"Video URL: {VIDEO_URL}")
        print(f"Target FPS: {TARGET_FPS}")
        print(f"Compression Quality: {COMPRESSION_QUALITY}%")
        print(f"Resize Scale: {RESIZE_SCALE * 100}%")
        print(f"Notification Cooldown: {NOTIFICATION_COOLDOWN}s")
        print("=" * 60)
        
        # Open video stream
        print("\n[INFO] Connecting to video stream...")
        
        if VIDEO_URL:
            cap = cv2.VideoCapture(VIDEO_URL)
            source_name = VIDEO_URL
        else:
            cap = cv2.VideoCapture(CAMERA_INDEX)
            source_name = f"Camera index {CAMERA_INDEX}"
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot connect to {source_name}!")
            print("[INFO] Trying other camera indices...")
            # Try different camera indices
            for idx in [0, 1, 2]:
                if idx != CAMERA_INDEX:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        print(f"[INFO] Found camera at index {idx}")
                        break
            
            if not cap.isOpened():
                print("[ERROR] No video source available. Exiting.")
                return
        
        print("[SUCCESS] Connected to video stream!")
        print("[INFO] Press Ctrl+C to stop.\n")
        
        last_analysis_time = 0
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("[WARNING] Failed to read frame. Reconnecting...")
                    time.sleep(1)
                    cap = cv2.VideoCapture(VIDEO_URL)
                    continue
                
                current_time = time.time()
                
                # Control framerate
                if current_time - last_analysis_time < self.frame_interval:
                    continue
                
                last_analysis_time = current_time
                frame_count += 1
                
                # Compress frame
                compressed_frame = self.compress_frame(frame)
                base64_image = self.frame_to_base64(compressed_frame)
                
                # Calculate compressed size
                compressed_size_kb = len(compressed_frame) / 1024
                
                print(f"[Frame {frame_count}] Analyzing... (Size: {compressed_size_kb:.1f} KB)")
                
                # Analyze for hand detection
                hand_detected = self.analyze_frame_for_hand(base64_image)
                
                if hand_detected:
                    detection_count += 1
                    print(f"[ALERT] 🖐️ MAIN DÉTECTÉE! (Detection #{detection_count})")
                    self.send_notification()
                else:
                    print(f"[Frame {frame_count}] No hand detected.")
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Stopping...")
        finally:
            cap.release()
            print(f"\n[SUMMARY]")
            print(f"  Total frames analyzed: {frame_count}")
            print(f"  Total hands detected: {detection_count}")
            print("[INFO] System stopped.")


def main():
    """Entry point."""
    system = HandDetectionSystem()
    system.run()


if __name__ == "__main__":
    main()
