# Face Recognition SDK using Deep Learning

A Face Recognition system built using Deep Learning with MTCNN and FaceNet.

## Features

- Face Detection using MTCNN
- Face Embedding using FaceNet (InceptionResnetV1)
- Face Matching using Cosine Similarity
- Face Database Generation
- Face Recognition Script
- Streamlit Web Application

## Tech Stack

- Python
- PyTorch
- FaceNet
- MTCNN
- OpenCV
- Streamlit

## Project Structure

```text
app.py
src/
scripts/
notebooks/
data/
```

## Installation

```bash
git clone <repo-url>

cd FaceRecognition

python -m venv .venv

pip install -r requirements.txt
```

## Run

Build database

```bash
python scripts/build_database.py
```

Recognize face

```bash
python scripts/recognize_face.py
```

Run Streamlit

```bash
streamlit run app.py
```

## Dataset

LFW (Labeled Faces in the Wild)

## Future Improvements

- Multi-face Recognition
- Face Registration
- Live Webcam Recognition
- FastAPI Deployment
