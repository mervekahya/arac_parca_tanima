import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time

# Konfigürasyon
DATA_DIR = 'data/train'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'models/arac_parca_model.pth'

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Hata: '{DATA_DIR}' klasörü bulunamadı.")
        return

    # Veri ön işleme ve artırma
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Veri setlerini yükle
    print("Veri seti yükleniyor...")
    try:
        # Klasör yapısına göre otomatik etiketleme
        full_dataset = datasets.ImageFolder(DATA_DIR)
        
        # Train/Val split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Transformları ata (ImageFolder ile direkt split sonrası atamak zor olduğu için özel Dataset sarmalayıcı gerekebilir ama basitlik için manuel yapacağız veya subset kullanıp transform'u dataset classında halledeceğiz.
        # Basit yol: Dataset'i iki kere yükle.
        train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
        val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
        
        # İndeksleri tekrar ayır ki data leakage olmasın (seed sabit)
        indices = torch.randperm(len(full_dataset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        class_names = full_dataset.classes
        num_classes = len(class_names)
        print(f"Sınıflar: {class_names}")
        
    except Exception as e:
        print(f"Hata: {e}")
        return

    # Model oluştur (MobileNetV2)
    print("Model oluşturuluyor...")
    model = models.mobilenet_v2(pretrained=True)
    
    # Son katmanı değiştir
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Eğitim döngüsü
    print(f"Eğitim başlıyor ({EPOCHS} epoch)... CPU/GPU: {device}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Acc: {100 * correct / total:.2f}%")

    # Modeli kaydet
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model kaydedildi: {MODEL_SAVE_PATH}")
    
    with open('models/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

if __name__ == "__main__":
    train()
