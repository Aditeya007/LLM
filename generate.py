import torch
import tiktoken

from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

checkpoint = torch.load(
    "ckpt.pt",
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

enc = tiktoken.get_encoding("gpt2")

print(f"Using device: {device}")
print("Model loaded.")
print()

while True:
    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device,
            dtype=dtype,
            enabled=(device == "cuda")
        ):
            output = model.generate(
                idx,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50
            )

    generated_tokens = output[0].tolist()
    generated_text = enc.decode(generated_tokens)

    print(f"Model: {generated_text}")
    print()