import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
)

print("Model:", MODEL_ID)
print("Device:", next(model.parameters()).device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("CUDA allocated, MiB:",
      round(torch.cuda.memory_allocated() / 1024**2, 1))
