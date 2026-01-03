from icrawler.builtin import BingImageCrawler
import os

def download_images():
    # Kategori ve Arama kelimeleri
    # Format: "Klasör_Adi": "Arama Kelimesi"
    searches = {
        # Motor ve Güç Aktarımı
        "Hava_Filtresi": "Car Air Filter",
        "Yag_Filtresi": "Car Oil Filter",
        "Yakit_Filtresi": "Car Fuel Filter",
        "Motor_Yagi": "Motor Oil Bottle",
        "Buji": "Spark Plug",
        "Atesleme_Bobini": "Ignition Coil",
        "Triger_Seti": "Timing Belt Kit",
        "Turbo": "Car Turbocharger",
        "Radyator": "Car Radiator",
        "Termostat": "Car Thermostat",
        
        # Şanzıman ve Aktarma
        "Baski_Balata_Seti": "Clutch Kit",
        "Volan": "Flywheel",
        "Sanziman_Yagi": "Transmission Fluid",
        "Aks_Kafasi": "CV Joint",
        
        # Fren Sistemi
        "Fren_Balatasi": "Brake Pads",
        "Fren_Diski": "Brake Disc",
        "El_Freni_Teli": "Handbrake Cable",
        "ABS_Sensoru": "ABS Sensor",
        
        # Süspansiyon ve Direksiyon
        "Amortisor": "Shock Absorber",
        "Helezon_Yayi": "Coil Spring",
        "Salincak": "Suspension Control Arm",
        "Rotil": "Ball Joint",
        "Z_Rot": "Sway Bar Link",
        "Direksiyon_Kutusu": "Steering Rack",
        
        # Elektrik ve Aydınlatma
        "Aku": "Car Battery",
        "Alternator": "Alternator",
        "Mars_Motoru": "Starter Motor",
        "Far_Lambasi": "Car Headlight",
        "Stop_Lambasi": "Car Taillight",
        "Ampul": "Car Light Bulb",
        "Park_Sensoru": "Parking Sensor",
        
        # Soğutma ve İklimlendirme
        "Klima_Kompresoru": "AC Compressor",
        "Kondanser": "AC Condenser",
        "Radyator_Hortumu": "Radiator Hose",
        "Polen_Filtresi": "Cabin Air Filter",
        
        # Yakıt Sistemi
        "Yakit_Pompasi": "Fuel Pump",
        "Enjektor": "Fuel Injector",
        
        # Egzoz ve Emisyon
        "Egzoz_Manifoldu": "Exhaust Manifold",
        "Katalitik_Konvertor": "Catalytic Converter",
        "DPF": "Diesel Particulate Filter",
        "Susturucu": "Exhaust Muffler",
        "O2_Sensoru": "Oxygen Sensor",
        
        # Karoser ve İç Mekan
        "Kapi_Kolu": "Car Door Handle",
        "Cam_Fitili": "Car Window Seal",
        "Koltuk_Rayi": "Car Seat Rail",
        "Emniyet_Kemeri": "Seat Belt",
        
        # Silecek ve Görüş
        "Silecek_Kolu": "Wiper Arm",
        "Cam_Suyu_Deposu": "Windshield Washer Reservoir",
        "Ayna_Cami": "Side Mirror Glass",
        
        # Jant ve Rulman
        "Jant": "Car Rim",
        "Bijon": "Lug Nut",
        "Porya_Rulmani": "Wheel Hub Bearing",
        
        # Diğerleri
        "Conta": "Car Gasket",
        "Antifriz": "Antifreeze",
        "Cam_Suyu": "Windshield Washer Fluid",
        "Krank_Sensoru": "Crankshaft Position Sensor",
        "Hiz_Sensoru": "Speed Sensor",
        "Yag_Basinc_Sensoru": "Oil Pressure Sensor"
    }

    base_dir = "data"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for folder_name, search_query in searches.items():
        print(f"--- İndiriliyor: {folder_name} ({search_query}) ---")
        save_dir = os.path.join(base_dir, folder_name)
        
        # Klasör yoksa oluştur
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Crawler'ı başlat
        # max_num: İndirilecek resim sayısı. Yüksek doğruluk için artırılabilir.
        google_crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': save_dir},
            log_level='ERROR'
        )
        
        google_crawler.crawl(keyword=search_query, max_num=50, file_idx_offset=0)

if __name__ == "__main__":
    download_images()
