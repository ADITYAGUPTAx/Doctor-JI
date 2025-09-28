# ABSTRACT


Doctor Ji is an AI-powered mobile healthcare platform that addresses global healthcare challenges through advanced multimodal deep learning. The application integrates:

• **Skin Disease Detection**: ResNet-based CNN achieving ~85% accuracy on HAM10000 dataset for dermatoscopic image classification across 7 disease categories

• **Chest X-ray Analysis**: Novel FiLM-augmented DenseNet-121 framework for thoracic disease classification, incorporating patient metadata (age, gender, view position) to achieve Micro AUC of 0.882 across 14 disease categories on NIH ChestX-ray14 dataset

• **Comprehensive Healthcare Services**: Rule-based chatbot, ambulance booking, doctor appointments, emergency SOS, and medical record management

Built using Flutter for cross-platform compatibility and Firebase for scalable backend services, Doctor Ji demonstrates the clinical value of multimodal AI in democratizing healthcare access.

### 🎯 Project Impact
- **Clinical**: Improved diagnostic accuracy through multimodal AI
- **Technical**: Novel application of FiLM in medical imaging
- **Social**: Democratizing healthcare access through mobile AI
- **Academic**: Open-source research contribution to medical AI community


# 🚀 KEY FEATURES

## AI-Powered Diagnostics
- **Skin Disease Detection**: ResNet-50 based classification for 7 skin conditions (akiec, bcc, bkl, df, mel, nv, vasc)
- **Chest X-ray Analysis**: Multimodal FiLM-augmented DenseNet-121 for 14 thoracic diseases
- **Metadata Integration**: Patient context (age, gender, view position) incorporated via Feature-wise Linear Modulation

## Healthcare Services
- Rule-based health chatbot for symptom analysis
- Ambulance booking and emergency services
- Doctor appointment scheduling
- Medical record management
- Emergency SOS functionality

## Technical Excellence
- Cross-platform Flutter mobile application
- Firebase backend integration
- Transfer learning with multiple CNN architectures
- Grad-CAM visualizations for model interpretability
- Open-source research contributions

<img width="900" height="543" alt="Screenshot 2025-04-29 194914" src="https://github.com/user-attachments/assets/3c8cae7f-3674-4581-864d-b6a2f3613950" />

# Skin Disease Detection with ResNet-50 on HAM10000

This project trains a deep learning model using **ResNet-50** to classify skin diseases from the **HAM10000 dataset** (Human Against Machine with 10,000 training images). The dataset contains dermatoscopic images of skin lesions categorized into **7 classes**:

* **akiec**: Actinic keratoses and intraepithelial carcinoma
* **bcc**: Basal cell carcinoma
* **bkl**: Benign keratosis-like lesions
* **df**: Dermatofibroma
* **mel**: Melanoma
* **nv**: Melanocytic nevi
* **vasc**: Vascular lesions

---

## 🔹 Project Workflow

### 1. Dataset Preparation

* The HAM10000 dataset is downloaded using **KaggleHub**.
* Two ZIP archives (`HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`) are extracted.
* Metadata file (`HAM10000_metadata.csv`) is used to map `image_id` to diagnosis (`dx`) labels.
* Final dataframe contains image paths + encoded labels.

### 2. Data Preprocessing

* Images resized to **224×224×3** (ResNet-50 input size).
* Pixel values normalized using **ResNet-50 preprocess\_input** (scales to \[-1, 1]).
* Data augmentation with TensorFlow preprocessing layers:

  * Random horizontal flips
  * Random rotations (±10%)
  * Random zooms
  * Random contrast changes
* Train/validation/test split: **70% / 15% / 15%** with stratification.

### 3. Handling Class Imbalance

* The dataset is highly imbalanced (e.g., many `nv` samples, few `df`).
* **Class weights** are computed with `sklearn.utils.class_weight` to penalize underrepresented classes.

### 4. Model Architecture

* Base: **ResNet-50 pretrained on ImageNet**, without the fully connected (top) layer.
* Added layers:

  * Global Average Pooling
  * Dropout (0.3)
  * Dense softmax output (7 classes)
* Training strategy:

  1. **Stage 1**: Freeze most of ResNet-50, train only the top layers.
  2. **Stage 2 (Fine-tuning)**: Unfreeze the top 50 ResNet layers and train with a lower learning rate.

### 5. Training Setup

* **Loss**: Categorical Crossentropy
* **Optimizer**: Adam (`lr=1e-4` → reduced to `2e-5` for fine-tuning)
* **Metrics**: Accuracy
* **Callbacks**:

  * EarlyStopping (patience=6, monitor=val\_accuracy)
  * ReduceLROnPlateau (halve LR if val\_loss plateaus)
  * ModelCheckpoint (save best model by val\_accuracy)
  * CSVLogger (log training progress)

### 6. Evaluation

* Metrics: Accuracy, Confusion Matrix, Classification Report (precision, recall, F1-score).
* Expected performance:

  * Train Accuracy: \~85–90%
  * Validation Accuracy: \~78–82%
  * Test Accuracy: \~78–82%
* Accuracy is limited by class imbalance, so F1-score is the preferred metric.

---

## 🔹 Files & Artifacts

* `best_model.keras` → Saved trained ResNet-50 model.
* `label_map.json` → Mapping of class labels to indices.
* `history.csv` → Training history (loss/accuracy per epoch).

---

## 🔹 Inference Example

```python
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import json
from pathlib import Path

# Load model & label map
model = load_model("artifacts/best_model.keras")
with open("artifacts/label_map.json") as f:
    label_map = json.load(f)
labels = list(label_map.keys())

# Prediction helper
def predict_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224,224))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    return labels[idx], float(preds[idx])

print(predict_image("HAM10000_images_part_1/ISIC_0024306.jpg"))
```

Output:

```
('nv', 0.92)
```

---

## 🔹 Technical Highlights

* **ResNet-50 Backbone**: Deep residual connections help avoid vanishing gradients and enable training of very deep networks.
* **Transfer Learning**: Pretrained ImageNet weights accelerate convergence and improve generalization.
* **Class Imbalance Handling**: Using `class_weight` ensures minority classes (like `df`) get properly recognized.
* **Fine-tuning**: Gradually unfreezing deeper layers allows the model to learn skin-specific features while retaining generic image features.

# 🛠️ IMPLEMENTATION DETAILS

## Skin Disease Detection (Original)
- **Architecture**: ResNet-50 on HAM10000 dataset
- **Classes**: 7 dermatological conditions
- **Performance**: ~85% accuracy with class imbalance handling
- **Features**: Data augmentation, transfer learning, fine-tuning strategies

## Chest X-ray Classification (New Research Extension)
- **Primary Model**: DenseNet-121 + FiLM layers
- **Dataset**: NIH ChestX-ray14 (curated 15,000 images)
- **Multimodal Input**: Images + patient metadata (age, gender, view position)
- **Training Strategy**: Progressive unfreezing with metadata conditioning
- **Visualization**: Grad-CAM attention maps for clinical interpretability

  <img width="384" height="403" alt="original sample x-ray" src="https://github.com/user-attachments/assets/46e186ad-6f29-4bc6-995f-9390277258f1" />, <img width="428" height="432" alt="Densenet with film gradcam" src="https://github.com/user-attachments/assets/4f49c5dd-af36-4158-9aaa-01d182003e23" />



## Technology Stack
- **Frontend**: Flutter (Dart)
- **Backend**: Firebase
- **AI/ML**: Python, TensorFlow, PyTorch
- **Research**: Jupyter Notebooks
- **Deployment**: Cross-platform mobile (iOS/Android)


# CONCLUSION

Doctor Ji represents a transformative approach to healthcare delivery, leveraging AI to address critical challenges in accessibility, affordability, and reliability. By integrating a rulebased chatbot, a ResNet-based CNN for skin diagnostics, and multi-service features like ambulance booking, doctor appointments, and emergency SOS, the app empowers users across urban and rural settings. With 85% diagnostic accuracy, chatbot, and a user-friendly Flutterbased interface, Doctor Ji delivers a seamless and impactful healthcare experience. The case studies highlight its potential to save lives, manage chronic conditions, and bridge healthcare gaps. Despite challenges like mobile optimization and data privacy, the app’s future scope, including multilingual support, expanded diagnostics, and LLM integration, promises to amplify its global impact. Doctor Ji stands as a beacon of innovation in mHealth, paving the way for a more equitable and accessible healthcare ecosystem.
