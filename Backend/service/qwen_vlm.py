import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)
print("Loading Qwen2.5-VL 3B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
)

model = model.to(device)

# Reduce visual token count for the 8 GB Mac
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=224 * 224,
    max_pixels=448 * 448,
)

print("Model loaded successfully.")


def answer_question(pre_image, post_image, question):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pre_image,
                },
                {
                    "type": "image",
                    "image": post_image,
                },
                {
                    "type": "text",
                    "text": (
                        "Compare the pre-disaster and post-disaster "
                        "images and answer the question briefly.\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[pre_image, post_image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )

    # Remove the input tokens from the output
    input_length = inputs["input_ids"].shape[1]

    generated_ids = generated_ids[:, input_length:]

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return output[0].strip()