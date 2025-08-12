import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import load_image, damage_percentage

IMG_SIZE = (128, 128)
MODEL_PATH = "saved_model/model_weights.h5"
CLAIM_THRESHOLD = 30.0  # percentage

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
img_array = load_image(image_path, IMG_SIZE)
input_img = np.expand_dims(img_array, axis=0)

# Prediction
pred_mask = model.predict(input_img)[0]
pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

# Calculate damage %
damage_percent = damage_percentage(pred_mask_bin)
claim = "YES" if damage_percent >= CLAIM_THRESHOLD else "NO"

print(f"Damage: {damage_percent:.2f}%")
print(f"Insurance Claim Eligible: {claim}")

# Show result
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_array)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Damage Mask ({damage_percent:.2f}%)")
plt.imshow(pred_mask.squeeze(), cmap="Reds")
plt.axis("off")

plt.show()
