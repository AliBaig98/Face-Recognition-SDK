import pickle
from pathlib import Path

from src.models.face_embedder import FaceEmbedder
from src.inference.face_matcher import FaceMatcher


class FaceRecognizer:

    def __init__(self, database_path, threshold=0.70):

        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher(threshold)

        with open(database_path, "rb") as f:
            self.database = pickle.load(f)

    def recognize(self, face_tensor):

        embedding = self.embedder.get_embedding(face_tensor)

        best_person = "Unknown"
        best_score = -1

        for person, db_embedding in self.database.items():

            score, _ = self.matcher.compare(embedding, db_embedding)

            if score > best_score:
                best_score = score
                best_person = person

        if best_score < self.matcher.threshold:
            best_person = "Unknown"

        return best_person, best_score