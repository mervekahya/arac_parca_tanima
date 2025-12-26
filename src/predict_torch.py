import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import os

MODEL_PATH = 'models/arac_parca_model.pth'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMG_SIZE = (224, 224)

def load_inference_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen önce eğitimi çalıştırın.")
    
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
        
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Model mimarisini tekrar oluştur
    num_classes = len(class_names)
    model = models.mobilenet_v2(pretrained=False) # Ağırlıkları biz yükleyeceğiz
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model, class_names

def predict_image(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, class_names = load_inference_model(device)
    
    # Görüntü işleme
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) # Batch boyutu ekle
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    conf_score = confidence.item() * 100
    
    print(f"Bu resim {conf_score:.2f}% ihtimalle '{predicted_class}' sınıfına ait.")
    return predicted_class, conf_score

import random
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Araç parçası tahminleme (PyTorch).')
    parser.add_argument('--image', type=str, help='Tahmin edilecek resmin yolu')
    
    args = parser.parse_args()
    
    if args.image:
        predict_image(args.image)
    else:
        # Rastgele bir resim seç
        print("Resim yolu belirtilmedi, veri setinden rastgele bir resim seçiliyor...")
        all_images = glob.glob("data/*/*.jpg")
        if not all_images:
             # data_clean yapısını da kontrol et
             all_images = glob.glob("data_clean/train/*/*.jpg")
             
        if all_images:
            random_image = random.choice(all_images)
            print(f"Seçilen resim: {random_image}")
            predict_image(random_image)
        else:
            print("Hata: Test edilecek resim bulunamadı!")
