import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image(path, target_size):
    img = load_img(path, target_size=target_size)
    img_arr = img_to_array(img) / 255.0
    return img_arr

def load_mask(path, target_size):
    mask = load_img(path, target_size=target_size, color_mode="grayscale")
    mask_arr = img_to_array(mask) / 255.0
    mask_arr = (mask_arr > 0.5).astype(np.float32)  # binary mask
    return mask_arr

def damage_percentage(mask):
    return (np.sum(mask) / mask.size) * 100
