import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import time

MODEL_PATH = 'models/arac_parca_model.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMG_SIZE = (224, 224)

def load_inference_model(device):
    if not os.path.exists(MODEL_PATH):
        print("Model bulunamadı! Lütfen önce train_torch.py ile modeli eğitin.")
        return None, None
    
    if not os.path.exists(CLASS_NAMES_PATH):
        print("Sınıf etiketleri bulunamadı!")
        return None, None
        
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    num_classes = len(class_names)
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None, None

    model.to(device)
    model.eval()
    
    return model, class_names

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Kullanılan cihaz: {device}")

    model, class_names = load_inference_model(device)
    if model is None:
        return

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Kamera (Index 1 - Dahili) açılamadı, Index 0 deneniyor...")
        cap = cv2.VideoCapture(0)
        
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # Transform (PyTorch için normalizasyon önemli)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Çıkış için 'q' tuşuna basın.")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı.")
            break

        # OpenCV BGR -> PIL RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Preprocess
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Tahmin
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_index = predicted.item()
        conf_score = confidence.item() * 100
        predicted_label = class_names[predicted_index]
        
        # Görselleştirme
        color = (0, 255, 0) if conf_score > 70 else (0, 165, 255)
        text = f"{predicted_label} ({conf_score:.1f}%)"
        
        # Arka plan kutusu
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 5), (10 + text_w, 35 + 5), (0,0,0), -1)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Arac Parca Tanima (PyTorch)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
