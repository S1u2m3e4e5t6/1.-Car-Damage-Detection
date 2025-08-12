import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from utils import load_image, load_mask

# Paths
IMG_DIR = "images"
MASK_DIR = "masks"
IMG_SIZE = (128, 128)
EPOCHS = 20
BATCH_SIZE = 8
MODEL_PATH = "saved_model/model_weights.h5"

# Load dataset
print("[INFO] Loading dataset...")
images = []
masks = []
for filename in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, filename)
    mask_path = os.path.join(MASK_DIR, filename)
    if os.path.exists(mask_path):
        images.append(load_image(img_path, IMG_SIZE))
        masks.append(load_mask(mask_path, IMG_SIZE))

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# UNet model
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling2D((2,2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c7)

    model = models.Model(inputs, outputs)
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
