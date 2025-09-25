# NIH Chest X-ray Multi-Label Classification
**Extension of Doctor Ji Project**

---

## Project Motivation
This project is an extension of my **Doctor Ji AI-based healthcare platform**, which focuses on assisting doctors and patients in diagnosing medical conditions.

Chest X-rays are commonly used to detect thoracic diseases such as **Cardiomegaly, Effusion, Pneumothorax**, and others. However, interpreting X-rays can sometimes be **confusing**, especially for doctors in rural or resource-limited areas.

This AI system aims to:  
- **Assist doctors** by providing a second opinion when reading chest X-rays.  
- **Support families and patients** who may not have access to specialists.  
- Serve as a **decision-support tool** to reduce misdiagnosis and improve healthcare accessibility.

---

## Approach
1. **Dataset:** NIH Chest X-ray dataset (~15,000 images for initial experiments), including metadata: Patient Age, Gender, and View Position.  
2. **Problem:** Multi-label classification — each X-ray may have **multiple thoracic disease labels**.  
3. **Modeling:**  
   - Transfer learning using pre-trained CNNs: **DenseNet121, ResNet50, EfficientNetB0, VGG16**.  
   - Multi-input pipeline:  
     - **Image input:** chest X-ray images  
     - **Tabular input:** patient metadata (age, gender, view)  
   - Multi-hot encoding for labels.  
   - Training pipeline includes **data augmentation, normalization, and batch processing**.  
4. **Evaluation:**  
   - Metrics: **AUC, Precision, Recall, F1-score** per disease  
   - Best model selected based on **validation AUC**.  
   - Comparison across different CNN backbones to find the most suitable architecture.

---

## Impact
- Provides **decision support for doctors**, especially in rural areas.  
- Helps **families without easy access to specialists**.  
- Forms the foundation for **future enhancements** with more data and better fine-tuning.

---


---

## Results Summary
| Model       |  Micro Test AUC | Macro Test AUC | Notes |
|------------|----------------|----------|-------|
| DenseNet121 | 0.89         | 0.74    | Base frozen → fine-tuned last block |
| ResNet50    | 0.84           | 0.68      | Last 50 layers trained |
| EfficientNetB0 | TBD        | TBD      | Planned |
| VGG16       | TBD           | TBD      | Planned |


---

## Next Steps
- Fine-tune DenseNet fully to maximize performance.  
- Train and compare other backbones (ResNet, EfficientNet, VGG16).  
- Incorporate the best-performing model into the **Doctor Ji platform** for real-world testing.

---

