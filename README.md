
# 🦴 Knee Osteoarthritis Classification using Deep Learning Ensemble

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Accuracy-84.66%25-brightgreen)

---

## 📌 Overview

https://eleonore-energetic-conqueringly.ngrok-free.dev 

This project builds a **high-performance deep learning system** for detecting **Knee Osteoarthritis severity** from X-ray images.

It combines **multiple CNN architectures** into an **ensemble model** to improve accuracy, generalization, and robustness.

---

## 🎯 Objectives

* Classify Knee OA severity from X-ray images
* Reduce class complexity (5 → 3 classes)
* Improve performance using **ensemble learning**
* Achieve **high accuracy within limited training time (<1 hour)**

---

## 🧠 Model Architecture Overview

```
                 Input Image
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   DenseNet121   InceptionV3   MobileNet   EfficientNet
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
              Output Averaging
                      │
              Final Prediction
```

---

## 📊 Dataset Details

* Dataset: Knee Osteoarthritis X-ray images
* Directory structure:

```
train/
test/
val/
```

### Original Classes (5)

* Healthy
* Doubtful
* Minimal
* Moderate
* Severe

### Final Classes (3)

| Original Classes           | Final Class |
| -------------------------- | ----------- |
| Healthy, Doubtful, Minimal | Healthy     |
| Moderate                   | Moderate    |
| Severe                     | Severe      |

---

## ⚙️ Data Preprocessing Pipeline

### ✔ Image Processing

* Resize:

  * 224×224 → DenseNet, MobileNet, Inception
  * 300×300 → EfficientNet
* Normalization
* CLAHE (optional enhancement)

### ✔ Data Augmentation

* Rotation (20–30°)
* Horizontal Flip
* Zoom & Shift
* Brightness variation

### ✔ Dataset Balancing

* Reduced to **500 samples per class**
* Faster training + balanced learning

---

## 🧩 Individual Models

---

### 🔵 1. DenseNet121 (Best Individual Model)

**Architecture:**

* Pretrained on ImageNet
* Global pooling + Dense layers
* Dropout regularization

**Training Strategy:**

* Stage 1: Freeze base model
* Stage 2: Fine-tune last 50 layers

**Key Features:**

* Feature reuse via dense connections
* Strong generalization

✅ **Accuracy: 81.58%** 

---

### 🟠 2. EfficientNetB3

**Architecture:**

* Compound scaling (depth, width, resolution)
* Input: 300×300

**Techniques Used:**

* Mixed precision training
* AdamW optimizer
* Label smoothing

⚠️ **Accuracy: 43.24% (underperformed in this setup)** 

**Reason:**

* Difficulty handling class imbalance
* Overfitting despite augmentation

---

### 🟢 3. InceptionV3

**Architecture:**

* Multi-scale feature extraction
* Parallel convolution filters

**Usage:**

* Pretrained model
* Outputs 5 classes → mapped to 3

**Strength:**

* Captures fine-grained patterns

---

### 🟡 4. MobileNet

**Architecture:**

* Depthwise separable convolutions
* Lightweight and fast

**Usage:**

* Efficient feature extractor
* Included in ensemble

---

## 🔁 Class Conversion Logic

All 5-class models are converted to 3-class outputs:

```python
healthy = y[:,0] + y[:,1] + y[:,2]
moderate = y[:,3]
severe = y[:,4]
```

---

## 🤝 Ensemble Model

### 🔹 Strategy

* Combine predictions from all models
* Use **average pooling**

```python
output = Average()(model_outputs)
```

### 🔹 Why Ensemble?

* Reduces variance
* Combines strengths of models
* Improves stability

---

## 📈 Results

### 🔹 Before Fine-Tuning

* Accuracy: **77.54%**

### 🔹 After Fine-Tuning

* ✅ **Final Accuracy: 84.66%** 

### 🔹 Improvement

* 🚀 **+7.13% boost**

---

## 📊 Performance Breakdown

| Model          | Classes | Accuracy   |
| -------------- | ------- | ---------- |
| DenseNet121    | 3       | 81.58%     |
| EfficientNetB3 | 5       | 43.24%     |
| InceptionV3    | 5       | Ensemble   |
| MobileNet      | 5       | Ensemble   |
| **Ensemble**   | 3       | **84.66%** |

---

## 📉 Key Insights

* Severe class has **low recall**
* Dataset imbalance affects results
* Ensemble improves:

  * Accuracy
  * Robustness
* DenseNet is strongest standalone model

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🖼️ Prediction Demo

* Random test image selected
* Model predicts:

  * Class label
  * Confidence

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python
```

### 3️⃣ Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4️⃣ Set Dataset Paths

```python
train_path = 'your_path/train'
test_path = 'your_path/test'
valid_path = 'your_path/val'
```

### 5️⃣ Run Models

* Train DenseNet
* Train EfficientNet
* Load all models
* Run Ensemble

---

## 🛠️ Tech Stack

* TensorFlow / Keras
* OpenCV
* NumPy / Pandas
* Matplotlib / Seaborn
* Scikit-learn

---

## 🔮 Future Improvements

* Improve minority class performance
* Use:

  * Focal Loss
  * Class re-weighting
* Try advanced models:

  * Vision Transformers (ViT)
  * ConvNeXt
* Advanced ensemble:

  * Stacking / weighted averaging

---

## ⭐ Key Highlights

✔ Multi-model ensemble
✔ Real-world medical application
✔ Fast training (<1 hour on T4 GPU)
✔ Improved performance via model fusion

---

## 📜 License

For educational and research purposes only.

---

## 👤 Author

**Shriya Mohanty**

---



