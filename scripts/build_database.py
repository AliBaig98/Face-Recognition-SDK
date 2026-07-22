from pathlib import Path
import pickle
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.face_detector import FaceDetector
from src.models.face_embedder import FaceEmbedder

DATASET = PROJECT_ROOT / "data" / "processed" / "lfw_filtered"

detector = FaceDetector()
embedder = FaceEmbedder()

database = {}

print("Building Face Database...\n")

for person_dir in DATASET.iterdir():

    if not person_dir.is_dir():
        continue

    embeddings = []

    for image_path in person_dir.glob("*.jpg"):

        try:
            face = detector.detect_face(image_path)
            embedding = embedder.get_embedding(face)
            embeddings.append(embedding)

        except Exception:
            continue

    if len(embeddings) == 0:
        continue

    # Average embedding
    mean_embedding = torch.stack(embeddings).mean(dim=0)

    database[person_dir.name] = mean_embedding

    print(f"{person_dir.name} ✓")

# Save database
save_path = PROJECT_ROOT / "data" / "embeddings" / "face_database.pkl"

with open(save_path, "wb") as f:
    pickle.dump(database, f)

print("\nDatabase Saved Successfully!")
print(f"Total Persons : {len(database)}")