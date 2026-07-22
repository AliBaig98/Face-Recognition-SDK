from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.face_detector import FaceDetector
from src.models.face_embedder import FaceEmbedder
from src.inference.face_matcher import FaceMatcher

# ----------------------------
# Load Models
# ----------------------------

detector = FaceDetector()
embedder = FaceEmbedder()
matcher = FaceMatcher()

# ----------------------------
# Load Database
# ----------------------------

database_path = PROJECT_ROOT / "data" / "embeddings" / "face_database.pkl"

with open(database_path, "rb") as f:
    database = pickle.load(f)

# ----------------------------
# Input Image
# ----------------------------

image_path = PROJECT_ROOT / "test_images" / "test.jpg"

face = detector.detect_face(image_path)
embedding = embedder.get_embedding(face)

# ----------------------------
# Compare
# ----------------------------

# ----------------------------
# Compare with Database
# ----------------------------

best_person = "Unknown"
best_score = -1

THRESHOLD = 0.70

for person, db_embedding in database.items():

    score, _ = matcher.compare(embedding, db_embedding)

    if score > best_score:
        best_score = score
        best_person = person

# Check confidence
if best_score < THRESHOLD:
    best_person = "Unknown"

# Display Result
print("-" * 40)
print("Prediction")
print("-" * 40)
print(f"Person     : {best_person}")
print(f"Similarity : {best_score:.4f}")
print("-" * 40)