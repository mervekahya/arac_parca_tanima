import cv2
import sounddevice as sd
import numpy as np

def check_camera():
    print("--- KAMERA KONTROLU ---")
    
    for idx in [0, 1]:
        print(f"\nKamera Index {idx} deneniyor...")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"HATA: Kamera (Index {idx}) açılamadı.")
        else:
            print(f"Kamera (Index {idx}) bağlantısı başarılı. Okunuyor...")
            # Birkaç frame atla ki kamera ısınsın/açılsın
            for _ in range(5):
                cap.read()
                
            ret, frame = cap.read()
            if ret:
                print(f"BAŞARILI: Index {idx} üzerinden görüntü alındı.")
                print(f"Çözünürlük: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"HATA: Kamera (Index {idx}) açık ama görüntü alınamadı.")
            cap.release()
    
    return True # Genel test bitti

def check_microphone():
    print("\n--- MİKROFON KONTROLU ---")
    try:
        print("Mikrofonlar listeleniyor:")
        print(sd.query_devices())
        
        fs = 44100
        seconds = 1
        print(f"\nTest kaydı yapılıyor ({seconds} sn)...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        print("BAŞARILI: Ses kaydı tamamlandı (Hata fırlatılmadı).")
        if np.max(np.abs(myrecording)) < 0.001:
             print("UYARI: Kaydedilen ses tamamen sessiz görünüyor. Mikrofon izni olmayabilir veya ses kısık.")
        else:
             print("Ses verisi algılandı.")
        return True
    except Exception as e:
        print(f"HATA: Mikrofon hatası: {e}")
        return False

if __name__ == "__main__":
    print("Donanım kontrolü başlatılıyor...")
    cam_ok = check_camera()
    mic_ok = check_microphone()
    
    print("\n--- SONUÇ ---")
    if cam_ok and mic_ok:
        print("Her şey yolunda görünüyor!")
    else:
        print("Sorunlar tespit edildi. Lütfen yukarıdaki hataları kontrol edin.")
