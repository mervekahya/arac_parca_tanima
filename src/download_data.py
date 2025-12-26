from icrawler.builtin import BingImageCrawler
import os

def download_images():
    # Kategori ve Arama kelimeleri
    # Format: "Klasör_Adi": "Arama Kelimesi"
    searches = {
        "Toyota_Corolla_Buji": "Toyota Corolla spark plug",
        "Ford_Focus_Fren_Balatasi": "Ford Focus brake pads",
        "Honda_Civic_Far": "Honda Civic headlight",
        "Renault_Clio_Aku": "Renault Clio car battery",
        "Fiat_Egea_Lastik": "Fiat Egea tire"
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
            storage={'root_dir': save_dir}
        )
        
        google_crawler.crawl(keyword=search_query, max_num=60, file_idx_offset=0)

if __name__ == "__main__":
    download_images()
