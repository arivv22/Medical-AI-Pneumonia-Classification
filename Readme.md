# 🩺 Medical AI – Image Classification (Mini Project)

A mini project exploring **deep learning in medical imaging** using **transfer learning with ResNet18 (PyTorch)**.  
The goal is to build an AI model that can classify medical images (e.g., chest X-rays) into diagnostic categories such as **Normal** or **Pneumonia**.

---

## 📘 Project Overview
This project demonstrates a complete yet lightweight workflow for **medical image classification**.  
It was designed to be finished in under **4 hours**, easily understandable for learners, and suitable for academic or professional portfolios.

---

## 🚀 Features
- ✅ Transfer Learning with **ResNet18**
- ✅ Medical dataset (e.g., Chest X-ray Pneumonia)
- ✅ Evaluation: accuracy, F1-score, confusion matrix
- ✅ Visualization of predictions
- ✅ Reproducible in **Google Colab**

---

## 🩸 Dataset
Dataset used:
> **Chest X-Ray Images (Pneumonia)** – [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

```bash
Structure:
dataset/
│
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
│
└── test/
├── NORMAL/
└── PNEUMONIA/
```

---

## ⚙️ Installation
Run on **Google Colab** or locally.

```bash
# Clone repository
git clone https://github.com/<your-username>/medical-ai-image-classification.git
cd medical-ai-image-classification

# Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn
```

--- 
## 🧠 Model Overview

Base model: ResNet18 pretrained on ImageNet

Fine-tuned last layer for 2-class classification

Loss: CrossEntropyLoss

Optimizer: Adam

```bash
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (2 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
```

---

## 🧩 Training & Evaluation

Train for 5–10 epochs.

```bash
# Example training snippet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```
Example result (after few epochs):
Accuracy: 0.79
Precision: 0.86
Recall: 0.73
F1-score: 0.79

--- 

## 📊 Results & Visualization

Confusion Matrix


Sample Predictions


Model	Accuracy	Framework	Dataset
ResNet18 (Transfer Learning)	89%	PyTorch	Chest X-Ray Pneumonia

---

## 📁 File Structure

```bash
medical-ai-image-classification/
│
├── Medical_AI_Image_Classification.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── docs/
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## 📚 Learnings

- Transfer learning boosts medical imaging performance even with small datasets.

- Pretrained CNNs drastically reduce training time.

- Visualization helps interpret model reliability for healthcare AI.

---

## 🧑‍💻 Author    

M. Afdhal Arief Malik
Aspiring AI Researcher | Backend Developer 
