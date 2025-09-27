Skin Disease Detection with ResNet-50 on HAM10000
This project trains a deep learning model using ResNet-50 to classify skin diseases from the HAM10000 dataset (Human Against Machine with 10,000 training images). The dataset contains dermatoscopic images of skin lesions categorized into 7 classes:

akiec: Actinic keratoses and intraepithelial carcinoma
bcc: Basal cell carcinoma
bkl: Benign keratosis-like lesions
df: Dermatofibroma
mel: Melanoma
nv: Melanocytic nevi
vasc: Vascular lesions
🔹 Project Workflow
1. Dataset Preparation
The HAM10000 dataset is downloaded using KaggleHub.
Two ZIP archives (HAM10000_images_part_1.zip, HAM10000_images_part_2.zip) are extracted.
Metadata file (HAM10000_metadata.csv) is used to map image_id to diagnosis (dx) labels.
Final dataframe contains image paths + encoded labels.
2. Data Preprocessing
Images resized to 224×224×3 (ResNet-50 input size).

Pixel values normalized using ResNet-50 preprocess_input (scales to [-1, 1]).

Data augmentation with TensorFlow preprocessing layers:

Random horizontal flips
Random rotations (±10%)
Random zooms
Random contrast changes
Train/validation/test split: 70% / 15% / 15% with stratification.

3. Handling Class Imbalance
The dataset is highly imbalanced (e.g., many nv samples, few df).
Class weights are computed with sklearn.utils.class_weight to penalize underrepresented classes.
4. Model Architecture
Base: ResNet-50 pretrained on ImageNet, without the fully connected (top) layer.

Added layers:

Global Average Pooling
Dropout (0.3)
Dense softmax output (7 classes)
Training strategy:

Stage 1: Freeze most of ResNet-50, train only the top layers.
Stage 2 (Fine-tuning): Unfreeze the top 50 ResNet layers and train with a lower learning rate.
5. Training Setup
Loss: Categorical Crossentropy

Optimizer: Adam (lr=1e-4 → reduced to 2e-5 for fine-tuning)

Metrics: Accuracy

Callbacks:

EarlyStopping (patience=6, monitor=val_accuracy)
ReduceLROnPlateau (halve LR if val_loss plateaus)
ModelCheckpoint (save best model by val_accuracy)
CSVLogger (log training progress)
6. Evaluation
Metrics: Accuracy, Confusion Matrix, Classification Report (precision, recall, F1-score).

Expected performance:

Train Accuracy: ~85–90%
Validation Accuracy: ~78–82%
Test Accuracy: ~78–82%
Accuracy is limited by class imbalance, so F1-score is the preferred metric.

🔹 Files & Artifacts
best_model.keras → Saved trained ResNet-50 model.
label_map.json → Mapping of class labels to indices.
history.csv → Training history (loss/accuracy per epoch).
🔹 Inference Example
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
Output:

('nv', 0.92)
🔹 Technical Highlights
ResNet-50 Backbone: Deep residual connections help avoid vanishing gradients and enable training of very deep networks.
Transfer Learning: Pretrained ImageNet weights accelerate convergence and improve generalization.
Class Imbalance Handling: Using class_weight ensures minority classes (like df) get properly recognized.
Fine-tuning: Gradually unfreezing deeper layers allows the model to learn skin-specific features while retaining generic image features.
