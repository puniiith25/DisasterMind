import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()

print(f"Loading CLIP: {MODEL_NAME}")
print(f"CLIP device: {device}")


model = CLIPModel.from_pretrained(
    MODEL_NAME
).to(device)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.eval()


def embed_image(image_path: str):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        # Use the vision model directly
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output

        # Project vision features into CLIP embedding space
        features = model.visual_projection(
            pooled_output
        )

    # Normalize
    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    return features[0].cpu().tolist()