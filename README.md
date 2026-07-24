# 🧑 Face Recognition System using FaceNet & MTCNN

A Deep Learning based Face Recognition System built using **PyTorch**, **FaceNet (InceptionResnetV1)** and **MTCNN**.

This project detects faces from images, generates high-dimensional face embeddings, and identifies individuals using cosine similarity.

---

## 📌 Features

- Face Detection using MTCNN
- Face Embedding Generation using FaceNet
- Image-based Face Recognition
- Cosine Similarity Matching
- Custom Face Database
- Modular Project Structure
- Easy to Add New Identities

---

## 🛠️ Tech Stack

- Python
- PyTorch
- FaceNet (InceptionResnetV1)
- MTCNN
- OpenCV
- Pillow
- NumPy

---

## 📂 Project Structure

```
Face-Recognition-System/

│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── inference/
│   └── utils/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

Dataset

↓

Face Detection (MTCNN)

↓

Face Alignment

↓

Face Embedding (FaceNet)

↓

Embedding Database

↓

Cosine Similarity

↓

Identity Prediction

---

## 📊 Dataset

The project uses the **Labeled Faces in the Wild (LFW)** dataset.

Due to GitHub size limitations, the dataset is not included in this repository.

---

## 🚀 Installation

```bash
git clone https://github.com/AliBaig98/Face-Recognition-System.git

cd Face-Recognition-System

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## 📈 Results

- Detects faces accurately
- Generates FaceNet embeddings
- Matches identities using cosine similarity
- Supports custom face datasets

---

## 🔮 Future Improvements

- Real-Time Face Recognition
- Attendance Management System
- GUI Application
- Face Registration
- Face Tracking

---

## 👨‍💻 Author

**Ali Abdul Qadeer Baig**

LinkedIn:
(https://www.linkedin.com/in/ali-baig-089b082ba/)

GitHub:
https://github.com/AliBaig98
