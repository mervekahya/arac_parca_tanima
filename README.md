# Araç Parça Tanıma Sistemi

Bu proje, görüntü işleme ve derin öğrenme (Deep Learning) tekniklerini kullanarak araç parçalarını kameradan otomatik olarak tanıyan bir yapay zeka sistemidir.

Proje, kameradan aldığı görüntüyü analiz eder, parçayı tanır ve sonucu diğer uygulamaların (Web veya Masaüstü) kullanabilmesi için bir metin dosyasına (`current_search.txt`) yazar. Bu sayede, kameraya gösterilen parça otomatik olarak e-ticaret sitenizde veya uygulamanızda aratılabilir.

## Özellikler

*   **Gerçek Zamanlı Tanıma:** Kameradan canlı olarak parçaları tespit eder.
*   **Otomatik Tetikleme:** Parçayı %75 ve üzeri bir kesinlikle tanıdığında otomatik aksiyon alır.
*   **Entegrasyon Kolaylığı:** Tanınan parçanın ismini `current_search.txt` dosyasına yazar. Web arayüzü bu dosyayı okuyarak otomatik arama yapabilir.
*   **PyTorch Tabanlı:** Güçlü ve esnek model yapısı (MobileNetV2).

## Gereksinimler

Projenin çalışması için Python yüklü olmalıdır. Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Sistemi çalıştırmak için aşağıdaki adımları takip edin:

### 1. Modeli Eğitme (İlk Kurulum)
Veri setiniz `data/` klasöründe hazır olduktan sonra, yapay zekayı eğitmek için şu komutu çalıştırın. Bu işlem `models/` klasörüne eğitilmiş modeli kaydedecektir.

```bash
python src/train_torch.py
```

### 2. Canlı Tanıma Sistemini Başlatma
Eğitim tamamlandıktan sonra kamerayı açmak ve sistemi başlatmak için:

```bash
python src/realtime_predict_torch.py
```

Uygulama açıldığında:
*   Kameraya parçayı gösterin.
*   Sistem parçadan emin olduğunda ekranda **"Search Bar'a gönderiliyor..."** yazar.
*   Ana dizinde `current_search.txt` dosyasına parça ismi yazılır.

### 3. Entegrasyon Mantığı (Search Bar için)
Web siteniz veya uygulamanızda bir "file watcher" veya basit bir döngü ile `current_search.txt` dosyasını izleyebilirsiniz.

**Örnek Mantık:**
1.  Uygulama `current_search.txt` dosyasını her saniye kontrol eder.
2.  Dosya içeriği değiştiğinde (yeni bir parça ismi geldiğinde), bu ismi alır.
3.  Arama çubuğuna (Search Bar) yazdırır ve "Ara" butonuna basar.

## Klasör Yapısı

*   `data/`: Eğitim için kullanılacak parça resimleri (Her parça ayrı klasörde).
*   `models/`: Eğitilmiş model dosyası (`arac_parca_model.pth`) ve sınıf isimleri (`class_names.txt`).
*   `src/`: Kaynak kodlar.
    *   `train_torch.py`: Modeli eğiten kod.
    *   `realtime_predict_torch.py`: Kamera ile canlı tanıma yapan ve sonuçları dosyaya yazan kod.
*   `current_search.txt`: (Otomatik oluşur) Son tanınan ve aratılan parçanın ismini tutar.
