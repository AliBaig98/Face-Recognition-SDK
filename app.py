import pickle
from pathlib import Path

import streamlit as st

from src.data.face_detector import FaceDetector
from src.models.face_embedder import FaceEmbedder
from src.inference.face_matcher import FaceMatcher

# ------------------------
# Project Paths
# ------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_PATH = PROJECT_ROOT / "data" / "embeddings" / "face_database.pkl"

# ------------------------
# Load Models
# ------------------------

detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher()

# ------------------------
# Load Database
# ------------------------

with open(DATABASE_PATH, "rb") as f:
    database = pickle.load(f)

# ------------------------
# UI
# ------------------------

st.set_page_config(page_title="Face Recognition SDK", layout="centered")

st.title("🧑 Face Recognition SDK")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = PROJECT_ROOT / "temp.jpg"

    with open(image, "wb") as f:
        f.write(uploaded_file.read())

    st.image(str(image), caption="Uploaded Image", width=300)

    face = detector.detect_face(image)

    embedding = embedder.get_embedding(face)

    best_person = "Unknown"
    best_score = -1

    THRESHOLD = 0.70

    for person, db_embedding in database.items():

        score, _ = matcher.compare(embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_person = person

    if best_score < THRESHOLD:
        best_person = "Unknown"

    st.success(f"Prediction : {best_person}")

    st.info(f"Similarity : {best_score:.4f}")