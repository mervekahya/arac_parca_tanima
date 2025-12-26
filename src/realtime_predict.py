import cv2
import tensorflow as tf
import numpy as np
import os
import time
from voice_utils import VoiceAssistant

MODEL_PATH = 'models/arac_parca_model.keras'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMG_SIZE = (224, 224)

# Global değişkenler (Thread erişimi için)
current_prediction = "Tanımsız"
current_confidence = 0.0
last_spoken_time = 0
voice_assistant = VoiceAssistant()

def load_inference_model():
    if not os.path.exists(MODEL_PATH):
        print("Model bulunamadı! Lütfen önce train.py ile modeli eğitin.")
        return None, None
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    if not os.path.exists(CLASS_NAMES_PATH):
        print("Sınıf etiketleri bulunamadı!")
        return None, None
        
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return model, class_names

def voice_callback(text):
    global current_prediction, current_confidence, last_spoken_time
    
    keywords = ["bu nedir", "bu ne", "nedir bu", "nadir bu", "parça ne"]
    
    # Kelime kontrolü
    if any(keyword in text for keyword in keywords):
        current_time = time.time()
        # En az 5 saniyede bir cevap versin (Flood koruması)
        if current_time - last_spoken_time > 5:
            if current_confidence > 50: # Sadece eminse konuşsun
                message = f"Bu bir {current_prediction}."
                voice_assistant.speak(message)
                last_spoken_time = current_time
            else:
                voice_assistant.speak("Tam olarak emin değilim, lütfen kamerayı biraz daha yaklaştırın.")
                last_spoken_time = current_time

def main():
    global current_prediction, current_confidence
    
    model, class_names = load_inference_model()
    if model is None:
        return

    # Kamerayı başlat (0 varsayılan kamera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Çıkış için 'q' tuşuna basın.")
    
    # Sesli asistanı başlat
    voice_assistant.start_listening_loop(voice_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı.")
            break

        # Görüntüyü işle
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, IMG_SIZE)
        img_array = tf.expand_dims(resized_frame, 0)
        
        # Tahmin (Daha hızlı olması için her frame yerine her 5 frame'de bir yapılabilir ama şimdilik her frame)
        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        predicted_label = class_names[predicted_index]
        
        # Global değişkenleri güncelle
        current_prediction = predicted_label
        current_confidence = confidence

        # Sonucu ekrana yazdır
        color = (0, 255, 0) if confidence > 70 else (0, 165, 255)
        text = f"{predicted_label} ({confidence:.1f}%)"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Sesli komut: 'Bu nedir?'", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Arac Parca Tanima', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    voice_assistant.stop_listening()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
