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
| ResNet-50                 |          0.842 |          0.68  | Last 50 layers trained                             |
| ConvNeXt-tiny             |          0.806 |          0.59  | Promising; further tuning on larger subset planned |
| ViT-Base-16               |          0.747 |          0.65  | Transformer baseline                               |

###  Per-class AUC scores  

| Disease             | ConvNeXt | ResNet-50 | DenseNet-121 | DenseNet+FiLM | ViT Base |
|---------------------|----------|-----------|--------------|---------------|----------|
| Atelectasis         | 0.714    | **0.791** | 0.711        | 0.722         | 0.716    |
| Cardiomegaly        | 0.749    | **0.824** | 0.792        | **0.824**     | 0.744    |
| Consolidation       | 0.683    | 0.683     | 0.706        | **0.731**     | 0.696    |
| Edema               | 0.781    | 0.826     | 0.819        | 0.844         | **0.864**|
| Effusion            | 0.797    | 0.798     | 0.798        | **0.829**     | 0.773    |
| Emphysema           | 0.722    | 0.819     | 0.818        | **0.824**     | 0.548    |
| Fibrosis            | 0.667    | **0.677** | 0.676        | 0.660         | 0.652    |
| Hernia              | 0.689    | 0.863     | 0.863        | **0.918**     | –        |
| Infiltration        | 0.638    | 0.681     | 0.654        | 0.682         | **0.696**|
| Mass                | 0.674    | 0.686     | 0.643        | **0.687**     | 0.594    |
| No Finding          | 0.674    | **0.713** | 0.693        | **0.713**     | 0.685    |
| Nodule              | 0.561    | **0.581** | **0.581**    | 0.568         | 0.470    |
| Pleural Thickening  | 0.611    | 0.654     | 0.654        | **0.687**     | 0.599    |
| Pneumonia           | 0.507    | 0.601     | 0.601        | 0.621         | **0.659**|
| Pneumothorax        | 0.775    | 0.796     | 0.795        | **0.808**     | 0.560    |

---

## Next Steps
- **Full Fine-tuning** of DenseNet-121 backbone to further boost AUC  
- **Experiment** with additional architectures (EfficientNet, VGG16)  
- **Integrate Best Model** into the Doctor Ji mobile platform for field validation  
- **Expand Dataset** and metadata (clinical history, lab values) for richer multimodal learning  
