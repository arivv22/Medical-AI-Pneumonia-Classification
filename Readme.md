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
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace last FC layer
model.fc = nn.Linear(model.fc.in_features, 2)
```

---

## 🧩 Training & Evaluation

Train for 5–10 epochs.

```bash
# Example training snippet
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```
Example result (after few epochs):
Accuracy: 0.89
Precision: 0.88
Recall: 0.91
F1-score: 0.89

--- 

## 📊 Results & Visualization

Confusion Matrix


Sample Predictions


Model	Accuracy	Framework	Dataset
ResNet18 (Transfer Learning)	89%	PyTorch	Chest X-Ray Pneumonia

---

## 📁 File Structure

medical-ai-image-classification/
│
├── Medical_AI_Image_Classification.ipynb   # Main notebook
├── README.md
├── requirements.txt
└── docs/
    ├── confusion_matrix.png
    └── sample_predictions.png

---

## 📚 Learnings

- Transfer learning boosts medical imaging performance even with small datasets.

- Pretrained CNNs drastically reduce training time.

- Visualization helps interpret model reliability for healthcare AI.

---

## 🧑‍💻 Author    

M. Afdhal Arief Malik
Aspiring AI Researcher | Backend Developer 
