from pathlib import Path

import cv2
from PIL import Image

from src.data.face_detector import FaceDetector
from src.inference.face_recognizer import FaceRecognizer
from src.utils.image_utils import preprocess_face


# ------------------------------------
# Paths
# ------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "face_database.pkl"
)


# ------------------------------------
# Initialize models
# ------------------------------------

print("Loading face detector...")
detector = FaceDetector()

print("Loading face recognition database...")
recognizer = FaceRecognizer(DATABASE_PATH)

print("Models loaded successfully.")


# ------------------------------------
# Open webcam
# ------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Convert OpenCV BGR frame to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb_frame)

    # Detect face bounding boxes
    boxes, probabilities = detector.detect_boxes(
        pil_image
    )

    if boxes is not None:

        for box, probability in zip(
            boxes,
            probabilities
        ):

            # Ignore weak face detections
            if probability is None or probability < 0.90:
                continue

            x1, y1, x2, y2 = map(
                int,
                box
            )

            # Keep coordinates inside the webcam frame
            height, width = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            # Draw the detection box immediately
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Crop the detected face
            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                continue

            try:

                # Convert the face to a FaceNet input tensor
                face_tensor = preprocess_face(
                    face_crop
                )

                # Recognize the person
                person, score = recognizer.recognize(
                    face_tensor
                )

                # Green = known
                # Red = unknown
                if person == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # Final name and score
                label = (
                    f"{person} | {score:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                print(
                    f"Prediction: "
                    f"{person}, "
                    f"Score: {score:.4f}"
                )

            except Exception as error:

                print(
                    "Recognition error:",
                    error
                )

                cv2.putText(
                    frame,
                    "Recognition Error",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

    else:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Show webcam
    cv2.imshow(
        "Real-Time Face Recognition",
        frame
    )

    # Q or ESC closes the application
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ------------------------------------
# Cleanup
# ------------------------------------

camera.release()

cv2.destroyAllWindows()