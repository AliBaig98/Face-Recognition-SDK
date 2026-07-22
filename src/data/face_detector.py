print("face_detector.py is being imported")

from facenet_pytorch import MTCNN
from PIL import Image

print("Imports completed")

class FaceDetector:
    def __init__(self, image_size=224):
        print("FaceDetector initialized")

        self.detector = MTCNN(
            image_size=image_size,
            margin=20,
            keep_all=False,
        )

    def detect_face(self, image_path):
        image = Image.open(image_path).convert("RGB")

        face = self.detector(image)

        return face

print("Class created successfully")