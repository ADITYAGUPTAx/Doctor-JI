# NIH Chest X-ray Multi-Label Classification

**Extension of Doctor Ji Project**

---

## Project Motivation
Chest X-rays are a frontline imaging modality for diagnosing thoracic diseases (e.g., **Cardiomegaly**, **Effusion**, **Pneumothorax**), yet interpretation can be challenging—especially in resource-limited settings. This extension of the Doctor Ji platform aims to:
- **Assist clinicians** by offering an AI-driven second opinion on chest X-rays  
- **Support patients and families** lacking specialist access  
- **Reduce misdiagnosis** and improve healthcare equity through decision-support tools  

---

## Approach

### 1. Dataset  
- **NIH ChestX-ray14**: Subset of ~15 000 images  
- **Metadata**: Patient age, gender, view position  

### 2. Problem Formulation  
- **Multi-label classification**: Each image may exhibit multiple thoracic disease labels  
- **Labels**: 14 disease categories (e.g., atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema, emphysema, fibrosis, pleural thickening, hernia)  

### 3. Modeling Pipeline  
- **Image Input**: Preprocessed chest X-ray images (normalized, augmented)  
- **Tabular Input**: One-hot/ordinal encoding of metadata (age binning, gender, view)  
- **Backbone Architectures**:  
  - DenseNet-121 with FiLM conditioning  
  - Base DenseNet-121 (transfer-learned)  
  - ResNet-50 (fine-tuned last 50 layers)  
  - ConvNeXt-tiny  
  - ViT-Base-16  
- **Fusion Strategy**: FiLM layers inject metadata into convolutional feature maps  
- **Training Details**:  
  - Data augmentation (flips, rotations, contrast)  
  - Multi-hot label vectors  
  - Optimizer: Adam with learning-rate scheduling  
  - Early stopping on validation AUC  

### 4. Evaluation Metrics  
- **Primary**: Micro-average AUC, Macro-average AUC  
- **Secondary**: Precision, Recall, F1-score per label  
- **Model Selection**: Highest validation micro-AUC  

---

## Results Summary

| Model                     | Micro Test AUC | Macro Test AUC | Notes                                              |
|---------------------------|---------------:|---------------:|----------------------------------------------------|
| DenseNet-121 + FiLM       |          0.882 |          0.74  | FiLM layers for metadata conditioning              |
| DenseNet-121 (fine-tuned) |          0.877 |          0.72  | Last block unfrozen and fine-tuned                 |
| ResNet-50                 |          0.840 |          0.68  | Last 50 layers trained                             |
| ConvNeXt-tiny             |          0.800 |          0.59  | Promising; further tuning on larger subset planned |
| ViT-Base-16               |          0.740 |          0.65  | Transformer baseline                               |

---

## Next Steps
- **Full Fine-tuning** of DenseNet-121 backbone to further boost AUC  
- **Experiment** with additional architectures (EfficientNet, VGG16)  
- **Integrate Best Model** into the Doctor Ji mobile platform for field validation  
- **Expand Dataset** and metadata (clinical history, lab values) for richer multimodal learning  
