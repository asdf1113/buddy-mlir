from transformers import AutoTokenizer
import os

model_name = "meta-llama/Llama-3.2-1B"
text = "hello"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=os.environ.get("HF_TOKEN"),
)

ids = tokenizer.encode(text, add_special_tokens=True)

print("input text:")
print(text)
print("token ids:")
print(ids)
print("length:")
print(len(ids))
print("eos_token_id:", tokenizer.eos_token_id)