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

def generate_with_repeat_block(idx, max_new_tokens, temperature, top_k, no_repeat_ngram_size):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if idx.size(1) >= no_repeat_ngram_size:
            seq = idx[0].tolist()
            banned = set()
            for i in range(len(seq) - no_repeat_ngram_size + 1):
                ngram = tuple(seq[i:i + no_repeat_ngram_size - 1])
                if ngram == tuple(seq[-(no_repeat_ngram_size - 1):]):
                    banned.add(seq[i + no_repeat_ngram_size - 1])
            for token_id in banned:
                logits[0, token_id] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

while True:
    prompt = input("You: ")

    if prompt.lower() in ["exit", "quit"]:
        break

    tokens = enc.encode_ordinary(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        output = generate_with_repeat_block(
            idx,
            max_new_tokens=100,
            temperature=0.9,
            top_k=50,
            no_repeat_ngram_size=3
        )

    generated_tokens = output[0].tolist()
    generated_text = enc.decode(generated_tokens)

    print(f"Model: {generated_text}")
    print()