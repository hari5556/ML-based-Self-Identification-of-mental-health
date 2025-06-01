# 🧠 ML-Based Self-Identification of Mental Health

This academic mini project aims to detect and classify a person’s emotional state using **voice** and **facial expressions**. By combining speech and image data, the system attempts to provide an early indication of mental health status using machine learning techniques.

---

## 📌 Project Overview

The system is divided into three core modules:

1. **Voice Emotion Recognition** – Classifies emotions based on audio signals
2. **Facial Emotion Recognition** – Identifies emotions from facial images
3. **Integrated Model** – Combines predictions from both modules for better accuracy

This project was developed as a team effort during our final year of B.Tech in Information Technology.

---

## 🧩 Modules

### 🎙️ Voice Emotion Recognition
- **Dataset:** RAVDESS (not included in repo due to size)
- **Features Used:** MFCC, Chroma, Mel Spectrogram
- **Model:** CNN-LSTM
- **Output:** Emotion label (e.g., happy, sad, angry)

### 😀 Facial Emotion Recognition
- **Dataset:** FER2013 (publicly available on Kaggle)
- **Model:** CNN
- **Output:** Facial emotion classification into predefined categories

### 🔁 Integrated Model
- Merges predictions from both modules
- Gives a more reliable assessment of emotional state

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Librosa (audio feature extraction)
- OpenCV (image processing)
- Scikit-learn
- Matplotlib (visualization)

---
