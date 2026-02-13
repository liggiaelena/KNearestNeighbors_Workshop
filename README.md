# 🩺 Medical Image Classification: Skin Lesion Diagnosis using KNN & CNN Embeddings
**MedMNIST Team Project - Foundations of Machine Learning Frameworks**

## 📘 Project Summary
This project implements a **K-Nearest Neighbors (KNN)** algorithm to classify dermatoscopic images from the **DermMNIST (MedMNIST v3)** dataset. To overcome the limitations of raw pixel intensity in medical imaging, we utilized pre-trained **ResNet-18 (CNN)** to extract 512-dimensional feature embeddings. This allows the model to calculate diagnostic similarity based on high-level visual features rather than simple color values.

---

## 🚀 Key Technical Improvements
In this final iteration, the team improved model reliability and clinical utility through advanced validation techniques:

**Note on Evaluation Strategy :** While the initial plan focused on generating standard metrics (Confusion Matrix, Classification Report), I upgraded this phase to a Comparative Evaluation Framework. This involved moving from a single train/test split to 5-Fold Cross-Validation, allowing us to mathematically prove that our optimized $k=17$ parameter provides superior stability and clinical safety compared to the baseline.

1. **Implementation of 5-Fold Cross-Validation:** We moved beyond a simple train/test split by implementing 5-fold cross-validation. This ensures that the chosen parameters are stable across different subsets of medical data, reducing the risk of "lucky" results.
2. **Hyperparameter Optimization:** While initial tests suggested $k=19$, our rigorous cross-validation identified **$k=17$** as the optimal neighbor count for this specific feature space.
3. **Performance Boost:** - **Overall Test Accuracy:** Improved from 71.52% to **72.12%**.
   - **Macro F1-score:** Increased from 0.30 to **0.32**, confirming the model became more "comprehensive" and balanced across diverse lesion types.
4. **Clinical Error Reduction:** Specifically for **Melanoma (Class 4)**—the most critical malignant lesion—the optimized model reduced the diagnostic error rate by **1.79%**.

---

## 🏗️ Machine Learning Pipeline
Our project follows a modular **ML Pipeline Architecture**:
- **Data Acquisition:** Loading ResNet-18 feature embeddings (.npz files).
- **Preprocessing:** Standardizing features using `StandardScaler` to ensure distance calculation fairness.
- **Optimization:** Using 5-Fold CV to find the most robust *k* parameter for clinical stability.
- **Evaluation:** Generating Confusion Matrices and **Per-Class Error Rates** (Method 2) to identify diagnostic "blind spots."
- **Reflection:** Analyzing misclassification risks from a nursing and clinical perspective.

---

## 🩺 Reflection

- The model performs exceptionally well on common benign cases (Class 5) but struggles with rare diseases. 
- Although $k=17$ improved results, the error rate for malignant lesions remains a concern. In a clinical setting, a "False Negative" (missing a cancer case) can lead to delayed treatment and life-threatening consequences.
- **Conclusion:** KNN with CNN embeddings is a powerful screening aid, but for final clinical decisions, future work must include "Class Weighting" to increase sensitivity toward rare but fatal conditions.

---

## ✍️ Instructor Information
- **Instructor:** David Espinosa
- **Course:** Foundations of Machine Learning Frameworks
- **Date:** 13th February 2026


## 👥 Team Members & Contributions

| Name | Student ID | Contribution |
|------|-----------|-------------|
| Liggia Elena Taboada Cruz | 9085905 | Data Acquisition, Cleaning & Preprocessing, Feature Engineering |
| Emmanuel Ihejiamaizu | 9080005 | Model Training (KNN hyperparameter tuning, best k selection) |
| Chao-Chung Liu | 9067679 | 5-Fold Cross-Validation, k=17 Optimization, Comparative Error Analysis, and Clinical Reflection|

---

## 🛠️ Getting Started & Data Setup


Follow these steps to set up the project environment and prepare the dataset for analysis:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/liggiaelena/KNearestNeighbors_Workshop.git](https://github.com/liggiaelena/KNearestNeighbors_Workshop.git)
   cd KNearestNeighbors_Workshop

2. **Create the Environment:**
   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate on Windows:
   venv\Scripts\activate

   # Activate on macOS/Linux:
   source venv/bin/activate

3. **Install all required libraries from the requirements file:**
   ```bash
   pip install -r requirements.txt

4. **Prepare Dataset**
   ```bash
   # Run this script to generate the .npz data files
   python extract_embeddings_dermamnist.py
   ```