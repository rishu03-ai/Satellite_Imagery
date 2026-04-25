# 🌍 Landslide Detection using Transfer Learning on Satellite Imagery

---

## 📌 Overview
This project focuses on detecting landslides from satellite imagery using deep learning and transfer learning techniques. The system classifies images into **Landslide** and **Non-Landslide** categories and can be used in disaster monitoring and early warning systems.

---

## 🚀 Key Features
- Binary image classification (Landslide vs Non-Landslide)
- Transfer learning using pretrained CNN models
- Handles class imbalance using class weights
- Tracks accuracy, precision, recall, and F1-score
- Complete ML pipeline (EDA → Preprocessing → Training → Evaluation → Visualization)
- Model comparison across architectures

---

## 🧠 Models Used
- **VGG16** – Baseline model
- **MobileNetV2** – Best overall performance ✅
- **EfficientNetB0** – High recall (important for safety)

---

## 📊 Results Summary

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|--------|----------|--------|----------|
| VGG16           | 0.94   | 0.98     | 0.95   | 0.96     |
| MobileNetV2     | 0.95   | 0.98     | 0.94   | 0.96     |
| EfficientNetB0  | 0.79   | 0.80     | 0.95   | 0.87     |

🔍 **Insights:**
- MobileNetV2 provides the best balance of performance and efficiency
- EfficientNetB0 achieves highest recall (detects most landslides)
- VGG16 is stable but computationally heavier

---

## 📂 Project Structure
SATELLITE IMAGERY/
│
├── dataset/ # Raw dataset (landslide & non-landslide)
├── logs/ # TensorBoard logs
│
├── models/ # Trained models
│ ├── vgg16_best.keras
│ ├── mobilenet_best.keras
│ └── efficientnet_best.keras
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02Preprocessing.ipynb
│ ├── 03Model_Training.ipynb
│ ├── 04Evaluation.ipynb
│ └── 05Visualization.ipynb
│
├── results/ # Logs & evaluation outputs
│ ├── vgg16_log.csv
│ ├── mobilenet_log.csv
│ ├── efficientnet_log.csv
│ ├── VGG16_report.txt
│ ├── MobileNetV2_report.txt
│ ├── EfficientNetB0_report.txt
│ └── report.txt
│
└── README.md


---

## ⚙️ Workflow

1. **EDA (Exploratory Data Analysis)**
   - Class distribution analysis
   - Image inspection
   - Dataset understanding

2. **Preprocessing**
   - Image resizing (224x224)
   - Normalization using dataset statistics
   - Data augmentation
   - Train/Validation/Test split

3. **Model Training**
   - Transfer learning applied
   - Frozen base models
   - Custom classification layers
   - Binary classification using sigmoid

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - Classification reports

5. **Visualization**
   - Training vs validation curves
   - Model comparison graphs
   - Precision & recall analysis

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 🔍 Key Insights

- **Recall is critical** in landslide detection:
  - Missing a landslide (false negative) is dangerous
  - False positives are acceptable in safety systems

- **Best Model Choice:**
  - MobileNetV2 → Balanced + efficient
  - EfficientNet → High recall (safety-focused)

---


## 🧪 How to Run

```bash
git clone <your-repo-link>
cd your-project-folder
pip install -r requirements.txt