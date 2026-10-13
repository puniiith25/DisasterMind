import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def get_device():

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


print(f"Loading embedding model: {MODEL_NAME}")

device = get_device()

print(f"Embedding device: {device}")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


def embed_text(text: str):

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()