import tensorflow as tf
import numpy as np
import argparse
import os

MODEL_PATH = 'models/arac_parca_model.keras'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMG_SIZE = (224, 224)

def load_inference_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}. Lütfen önce eğitimi çalıştırın.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
        
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
        
    return model, class_names

def predict_image(image_path):
    model, class_names = load_inference_model()
    
    img = tf.keras.utils.load_img(
        image_path, target_size=IMG_SIZE
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Batch boyutu ekle

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = 100 * np.max(predictions[0]) # Softmax yerine direkt probs da olabilir ama categorical_crossentropy kullandık, output zaten prob dağılımı olur genelde ama logits değilse.
    # Modelin son katmanı softmax aktivasyonuna sahip, direkt alabiliriz.
    
    # Not: Modelde son katman softmax olduğu için predictions[0] direkt olasılıkları verir.
    # tf.nn.softmax tekrar uygulamak gereksiz olabilir ama zararı olmaz (logit çıktı vermiyorsak).
    # Benim train.py kodumda `activation='softmax'` var, o yüzden çıktı zaten olasılık.
    confidence = 100 * np.max(predictions[0])

    print(f"Bu resim {confidence:.2f}% ihtimalle '{predicted_class}' sınıfına ait.")
    return predicted_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Araç parçası tahminleme.')
    parser.add_argument('--image', type=str, required=True, help='Tahmin edilecek resmin yolu')
    
    args = parser.parse_args()
    predict_image(args.image)
