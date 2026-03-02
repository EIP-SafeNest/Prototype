#!/usr/bin/env python3
"""
Test script to verify OpenAI and Twilio integration.
This script creates a test image with a hand drawing and sends it for analysis.
"""

import cv2
import base64
import numpy as np
from openai import OpenAI
from twilio.rest import Client
from datetime import datetime

# Configuration
OPENAI_API_KEY = "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"

# Twilio Configuration (WhatsApp Sandbox)
TWILIO_SID = "AC5008cc02f11dd209bbe632befa2d9808"
TWILIO_AUTH_TOKEN = "67907e241d8371c65d25fd1d167b77be"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio Sandbox WhatsApp
DESTINATION_WHATSAPP = "whatsapp:+33684466232"  # Your WhatsApp number


def test_openai_vision():
    """Test OpenAI Vision API with a simple image."""
    print("=" * 60)
    print("Testing OpenAI Vision API...")
    print("=" * 60)
    
    # Create a simple test image (blue background)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 50, 50)  # Dark blue background
    
    # Add text
    cv2.putText(img, "Test Image - No Hand", (100, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode to base64
    _, encoded = cv2.imencode('.jpg', img)
    base64_image = base64.b64encode(encoded).decode('utf-8')
    
    # Send to OpenAI
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image. Is there a human hand visible in this image? Respond with only 'YES' if you can see a hand, or 'NO' if there is no hand visible."
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
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"[SUCCESS] OpenAI Response: {answer}")
        print(f"[INFO] Expected: NO (test image has no hand)")
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return False


def test_twilio_whatsapp():
    """Test Twilio WhatsApp sending."""
    print("\n" + "=" * 60)
    print("Testing Twilio WhatsApp...")
    print("=" * 60)
    
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body=f"🔧 TEST: Système de détection de main - Test de connexion à {datetime.now().strftime('%H:%M:%S')}",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATION_WHATSAPP
        )
        
        print(f"[SUCCESS] SMS envoyé!")
        print(f"[INFO] Message SID: {message.sid}")
        print(f"[INFO] Status: {message.status}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Twilio error: {e}")
        return False


def main():
    print("\n🧪 Hand Detection System - Integration Test\n")
    
    # Test OpenAI
    openai_ok = test_openai_vision()
    
    # Test Twilio WhatsApp
    twilio_ok = test_twilio_whatsapp()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"OpenAI Vision API: {'✅ PASS' if openai_ok else '❌ FAIL'}")
    print(f"Twilio WhatsApp:   {'✅ PASS' if twilio_ok else '❌ FAIL'}")
    
    if openai_ok and twilio_ok:
        print("\n✅ All integrations working! System ready.")
    else:
        print("\n❌ Some integrations failed. Please check the errors above.")


if __name__ == "__main__":
    main()
