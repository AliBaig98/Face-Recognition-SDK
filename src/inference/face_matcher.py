import torch.nn.functional as F


class FaceMatcher:
    def __init__(self, threshold=0.70):
        self.threshold = threshold

    def compare(self, emb1, emb2):
        similarity = F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        matched = similarity >= self.threshold

        return similarity, matched