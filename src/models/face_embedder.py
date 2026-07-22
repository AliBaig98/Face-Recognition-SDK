import torch
from facenet_pytorch import InceptionResnetV1


class FaceEmbedder:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def get_embedding(self, face):
        face = face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(face)

        return embedding.squeeze().cpu()