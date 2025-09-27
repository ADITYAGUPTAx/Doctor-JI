# ABSTRACT

The global healthcare system grapples with challenges of accessibility, affordability, and diagnostic accuracy, particularly in underserved regions. Doctor Ji is an AI-powered mobile application designed to address these issues by integrating a rule-based chatbot, a convolutional neural network (CNN) for skin condition diagnostics, and multi-service features such as ambulance booking, doctor appointments, emergency SOS, and medical record management. Built using Flutter for crossplatform compatibility and Firebase for scalable backend services, the app leverages the HAM10000 dataset and ResNet18 architecture to achieve ∼85% diagnostic accuracy. This paper provides an in-depth exploration of the system’s architecture, methodology, implementation, performance evaluation, and realworld applications through case studies. It also explains foundational AI concepts like CNNs, ResNets, and activation functions to ensure accessibility for non-technical readers while maintaining a professional tone. The paper concludes with challenges, limitations, and future enhancements to expand Doctor Ji’s impact in democratizing healthcare.

 # KEY FEATURES

To address these gaps, Doctor Ji is developed as an all-inone mobile health (mHealth) application powered by artificial intelligence (AI). The app integrates:

• A rule-based chatbot for health guidance and symptom analysis.

• A convolutional neural network (CNN) for diagnosing skin conditions with ∼85% accuracy.

• Multi-service features, including ambulance booking,doctor appointment scheduling, emergency SOS, and medical record management.

Built using Flutter for seamless cross-platform performance and Firebase for robust backend services, Doctor Ji aims to provide accessible, affordable, and reliable healthcare solutions. This paper elaborates on the system’s architecture,methodology, implementation details, performance metrics, and practical applications through imaginary case studies. It also provides detailed explanations of AI concepts such as CNNs, ResNets, and activation functions to ensure clarity for diverse readers, while maintaining a professional and technical depth suitable for academic and industry audiences

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




# CONCLUSION

Doctor Ji represents a transformative approach to healthcare delivery, leveraging AI to address critical challenges in accessibility, affordability, and reliability. By integrating a rulebased chatbot, a ResNet-based CNN for skin diagnostics, and multi-service features like ambulance booking, doctor appointments, and emergency SOS, the app empowers users across urban and rural settings. With 85% diagnostic accuracy, chatbot, and a user-friendly Flutterbased interface, Doctor Ji delivers a seamless and impactful healthcare experience. The case studies highlight its potential to save lives, manage chronic conditions, and bridge healthcare gaps. Despite challenges like mobile optimization and data privacy, the app’s future scope, including multilingual support, expanded diagnostics, and LLM integration, promises to amplify its global impact. Doctor Ji stands as a beacon of innovation in mHealth, paving the way for a more equitable and accessible healthcare ecosystem.
