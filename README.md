# Araç Parça Tanıma Modeli

Bu proje, araç parçalarını fotoğraflardan veya kamera görüntüsünden tanıyarak, parçanın ne olduğunu (örn: Buji) ve hangi araç modeline ait olduğunu tespit eden bir yapay zeka modelini eğitmek ve çalıştırmak için oluşturulmuştur.

## Proje Amacı
- Araç parçalarının görüntülerini işlemek.
- Parçanın türünü ve ait olduğu araç modelini sınıflandırmak.
- Sonuçları (tahminleri) döndürmek.

## Kurulum
Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
```

## Veri Seti Yapısı
Modelin eğitilebilmesi için verilerin `data/` klasörü altında şu yapıda olması beklenmektedir (Örnek):

```
data/
  train/
    Toyota_Corolla_Buji/
      resim1.jpg
      resim2.jpg
    Ford_Focus_FrenBalatasi/
      ...
  val/
    ...
```

Veya etiketlerin bir CSV dosyasında tutulduğu bir yapı da kurgulanabilir.

## Kullanım
Eğitim işlemini başlatmak için:
```bash
python src/train.py
```

Tahmin yapmak için:
```bash
python src/predict_torch.py --image_path path/to/image.jpg
```

Kameradan canlı tespit yapmak için (Sesli Asistanlı):
```bash
python src/realtime_predict_torch.py
```
