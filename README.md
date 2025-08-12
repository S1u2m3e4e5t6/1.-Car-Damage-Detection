# 🚗 Car Damage Prediction using Deep Learning

![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.9+-brightgreen.svg)
![Framework: TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)

This project uses a Convolutional Neural Network (CNN) to predict whether a car is damaged or not, based on an image. The model is trained on the Kaggle Car Damage Dataset and implemented using TensorFlow/Keras.

## ✨ Features

-   Classifies images into two categories: **Damaged** or **Not Damaged**.
-   Achieves **>97% accuracy** on the test dataset.
-   Simple and clear project structure.
-   Provides a script for both training the model from scratch and running predictions on new images.

## 🖼️ Example Predictions

Here are a few examples of how the model performs on test images.

| Input Image                               | Predicted Label | Confidence Score |
| ----------------------------------------- | --------------- | ---------------- |
| *(Image of a damaged car would go here)* | `Damaged`       | `98.2%`          |
| *(Image of an undamaged car would go here)*| `Not Damaged`   | `99.5%`          |
| *(Image of a damaged car would go here)* | `Damaged`       | `95.7%`          |

*(Aap yahan apne model dwara predict ki gayi actual images aur unke results daal sakte hain)*

## 🧠 Model Architecture

The project uses a standard CNN architecture for image classification.

1.  **Input Layer**: Accepts images of size `(224, 224, 3)`.
2.  **Convolutional Layers**: Multiple `Conv2D` layers with `ReLU` activation to extract features from the image.
3.  **Pooling Layers**: `MaxPooling2D` layers to reduce the spatial dimensions of the feature maps.
4.  **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
5.  **Dense Layers**: Fully connected layers for high-level reasoning.
6.  **Output Layer**: A `Sigmoid` activation function that outputs a probability score for the "damaged" class.

## 💾 Dataset

This model is trained on the **Vehicle Dataset - Automatic Vehicle Damage Detection** from Kaggle. It contains thousands of images of damaged and undamaged cars.

-   **Dataset Link:** [https://www.kaggle.com/datasets/hendrichscullen/vehicle-dataset-automatic-vehicle-damage-detection](https://www.kaggle.com/datasets/hendrichscullen/vehicle-dataset-automatic-vehicle-damage-detection)

## 🛠️ Tech Stack

-   Python 3.9+
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Matplotlib

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Make sure you have Python 3.9 or a newer version installed on your system.

### 2. Installation

```bash
# 1. Clone this repository
git clone [https://github.com/yourusername/car-damage-prediction.git](https://github.com/yourusername/car-damage-prediction.git)

# 2. Navigate to the project directory
cd car-damage-prediction

# 3. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 4. Install the required dependencies
pip install -r requirements.txt
```
### 3. Usage
python car_damage_predict.py --mode train

To predict a single image:

python car_damage_predict.py --mode predict --image_path "path/to/your/car_image.jpg"

   
###   🙏 Acknowledgements
A big thank you to Kaggle for providing the dataset.

This project was built using the powerful TensorFlow and Keras libraries.
