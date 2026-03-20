from transformers import pipeline
import os

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B",
    token=os.environ.get("HF_TOKEN"),
    device_map="auto"
)

output = pipe("Hello, how are you?", max_new_tokens=50)
print(output)