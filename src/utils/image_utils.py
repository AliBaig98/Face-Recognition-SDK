import cv2
import torch


def preprocess_face(face):

    # Resize face to FaceNet input size
    face = cv2.resize(face, (160, 160))

    # Convert BGR to RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Convert pixel values to float
    face = face.astype("float32") / 255.0

    # Convert HWC to CHW
    face = torch.from_numpy(
        face
    ).permute(2, 0, 1)

    return face