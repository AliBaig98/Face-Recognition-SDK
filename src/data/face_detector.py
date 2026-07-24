from facenet_pytorch import MTCNN
from PIL import Image


class FaceDetector:
    """
    Face Detector using MTCNN
    Supports:
    1. Single face extraction (old project)
    2. Multi-face detection (real-time project)
    """

    def __init__(self, image_size=224):

        self.detector = MTCNN(
            image_size=image_size,
            margin=20,
            keep_all=True,      # Detect multiple faces
        )

    def detect_face(self, image_path):
        """
        Returns cropped face tensor.
        Used while building embeddings/database.
        """

        image = Image.open(image_path).convert("RGB")

        face = self.detector(image)

        if face is None:
            raise ValueError("No face detected.")

        # If multiple faces are returned,
        # use the first one.
        if len(face.shape) == 4:
            face = face[0]

        return face

    def detect_boxes(self, pil_image):
        """
        Returns bounding boxes.
        Used for real-time webcam detection.
        """

        boxes, probs = self.detector.detect(pil_image)

        return boxes, probs