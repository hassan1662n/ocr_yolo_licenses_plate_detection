# 🚘 Automatic Number Plate Recognition (ANPR) System  
**YOLOv8 + EasyOCR | Video-based License Plate Detection & Recognition**

---

## 📌 Overview

This project is a **complete Automatic Number Plate Recognition (ANPR) system** built using **deep learning and computer vision**.  
It detects vehicle license plates from videos, recognizes the text on them, and outputs an annotated video with stable plate numbers.

The system is designed to work on **real-world traffic videos** and focuses on:
- High detection accuracy
- OCR stability across frames
- Practical video processing performance

---

## 🎥 Demo – Output Video (YOLO + OCR)


https://github.com/user-attachments/assets/49f63669-ec9f-4603-b1cd-d41a50656c5e


---

## ✨ Features

- ✅ Custom **YOLOv8 model fine-tuned** for license plate detection  
- 🎥 Video-based processing (MP4 input → annotated MP4 output)  
- 🔍 OCR using **EasyOCR**
- 🧠 Plate text **format correction & validation**
- 🔁 Temporal plate stabilization (reduces OCR flicker)
- 📊 Real-time processing progress logging
- 💾 Clean output video with bounding boxes and plate numbers

---

## 🧠 Model Training (Google Colab)

The license plate detection model was **fine-tuned in Google Colab** using **YOLOv8 (Ultralytics)** and a dataset from **Roboflow Universe**.

### 🔹 Dataset
- **Source:** Roboflow Universe  
- **Project:** License Plate Recognition  
- **Version:** 13  
- **Format:** YOLOv8  
- **Classes:** 1 (License Plate)  
- **Training Images:** ~98,000  
- **Validation Images:** ~2,000  

### 🔹 Training Configuration
- **Model:** YOLOv8n (Nano)
- **Image Size:** 640 × 640
- **Epochs:** 5
- **Batch Size:** 16
- **GPU:** Tesla T4 (CUDA)
- **Optimizer:** AdamW (auto)

### 🔹 Training Results
| Metric | Value |
|------|------|
| Precision | **98.5%** |
| Recall | **95.5%** |
| mAP@50 | **97.7%** |
| mAP@50–95 | **70.5%** |
| Inference Speed | ~1.7 ms / image (GPU) |

The best model weights were saved as:

## ⚙️ Google Colab Training Code

### 1️⃣ Install Dependencies
```python
!pip install roboflow
!pip install ultralytics

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # replace with your Roboflow API key
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(13)
dataset = version.download("yolov8")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Nano YOLOv8 pre-trained model

result = model.train(
    data = dataset.location + "/data.yaml",
    epochs = 5,
    imgsz = 640,
    batch = 16,
    workers = 2,
    device = 0
)

import shutil, os

os.makedirs("saved_models", exist_ok=True)

# Save best and last weights
shutil.copy("runs/detect/train/weights/best.pt", "saved_models/license_plate_best.pt")
shutil.copy("runs/detect/train/weights/last.pt", "saved_models/license_plate_last.pt")

print("Weights saved in saved_models/")


