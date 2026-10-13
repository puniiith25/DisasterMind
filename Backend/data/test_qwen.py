from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Processor loaded successfully.")