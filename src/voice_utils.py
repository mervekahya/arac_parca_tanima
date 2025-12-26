import speech_recognition as sr
from gtts import gTTS
import os
import threading
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.last_speech_time = 0
        self.is_listening = False
        self.latest_command = None
        self.lock = threading.Lock()
        
    def speak(self, text, lang='tr'):
        """Metni sesli okur."""
        def _speak():
            try:
                print(f"Seslendiriliyor: {text}")
                tts = gTTS(text=text, lang=lang)
                
                # Geçici dosya oluştur
                with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                    temp_filename = fp.name + ".mp3" # Gerekirse
                    # Ancak delete=True olduğu için fp kapanınca silinir.
                    # Windows/Mac farkı olmaması için manuel isim verelim.
                    
                temp_filename = "temp_speech.mp3"
                tts.save(temp_filename)
                
                # Mac için afplay, Linux için aplay/mpg123, Windows için start
                if os.name == 'posix':
                    os.system(f"afplay {temp_filename}")
                else:
                    os.system(f"start {temp_filename}")
                    
                # Temizle (async olduğu için hemen silinirse çalmaz, afplay blokluyor mu? Evet bloklar.)
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as e:
                print(f"Seslendirme hatası: {e}")

        # Konuşmayı bloklamadan yap (Thread)
        threading.Thread(target=_speak).start()

    def listen_and_recognize(self):
        """Kısa bir süre dinler ve metne çevirir (Sounddevice kullanarak)."""
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording
        
        try:
            print("Dinleniyor... (3sn)")
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            
            # Geçici wav dosyasına kaydet
            temp_wav = "temp_rec.wav"
            wav.write(temp_wav, fs, myrecording)
            
            # SpeechRecognition ile tanı
            with sr.AudioFile(temp_wav) as source:
                audio_data = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio_data, language="tr-TR")
                    print(f"Algılanan Ses: {text}")
                    return text.lower()
                except sr.UnknownValueError:
                    return ""
                except sr.RequestError:
                    print("API Hatası")
                    return ""
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
        except Exception as e:
            print(f"Dinleme hatası: {e}")
            return ""

    def start_listening_loop(self, callback_command):
        """Arka planda sürekli dinler."""
        self.is_listening = True
        
        def _loop():
            while self.is_listening:
                text = self.listen_and_recognize()
                if text:
                    callback_command(text)
                time.sleep(0.5)
                
        threading.Thread(target=_loop, daemon=True).start()

    def stop_listening(self):
        self.is_listening = False

