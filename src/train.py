import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, layers
import os
import matplotlib.pyplot as plt

# Konfigürasyon
DATA_DIR = 'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 # Fine-tuning ile artacak
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'models/arac_parca_model.keras'

def build_model(num_classes):
    # Veri artırma katmanları (Augmentation)
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    inputs = Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x) # MobileNetV2 için normalizasyon (-1, 1 arasına getirir)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    
    # Base modelin parametrelerini dondur
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(1280, activation='relu')(x) # Ara katman
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    return model, base_model

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Hata: '{DATA_DIR}' klasörü bulunamadı.")
        return

    print("Veri seti yükleniyor...")
    try:
        train_ds = image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )

        val_ds = image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
    except ValueError as e:
        print(f"Hata: {e}")
        return

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Sınıflar: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Model oluşturuluyor...")
    model, base_model = build_model(num_classes)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("--- Aşama 1: Transfer Learning (Head Training) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stopping]
    )

    print("--- Aşama 2: Fine-Tuning ---")
    # Base modelin son 30 katmanını eğitime açalım
    base_model.trainable = True
    fine_tune_at = 120 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Learning rate'i düşürerek derle
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE/10),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    total_epochs = 20 # 10 + 10
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=history.epoch[-1],
        epochs=total_epochs,
        callbacks=[early_stopping]
    )

    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save(MODEL_SAVE_PATH)
    print(f"Model kaydedildi: {MODEL_SAVE_PATH}")
    
    with open('models/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

if __name__ == "__main__":
    train()
