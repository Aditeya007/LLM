import torch
import tiktoken

from model import GPT, GPTConfig


# ============================================================
# Setup
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

checkpoint = torch.load(
    "sft_ckpt.pt",
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

enc = tiktoken.get_encoding("gpt2")

EOT_TOKEN = 50256

print(f"Using device: {device}")
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
print()


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(
    idx,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    no_repeat_ngram_size=3
):

    for _ in range(max_new_tokens):

        # Keep only the model's context window
        idx_cond = (
            idx
            if idx.size(1) <= model.config.block_size
            else idx[:, -model.config.block_size:]
        )

        with torch.autocast(
            device_type=device,
            dtype=dtype,
            enabled=(device == "cuda")
        ):
            logits, _ = model(idx_cond)

        # Last-token logits
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Never generate EOT until the model actually decides to stop
        # (it is still allowed as the sampled stopping token)
        if top_k is not None:
            v, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            logits[logits < v[:, [-1]]] = float("-inf")

        # --------------------------------------------------------
        # No-repeat n-gram blocking
        # --------------------------------------------------------

        if idx.size(1) >= no_repeat_ngram_size:

            seq = idx[0].tolist()
            banned = set()

            prefix = tuple(
                seq[-(no_repeat_ngram_size - 1):]
            )

            for i in range(
                len(seq) - no_repeat_ngram_size + 1
            ):

                ngram_prefix = tuple(
                    seq[i:i + no_repeat_ngram_size - 1]
                )

                if ngram_prefix == prefix:
                    banned.add(
                        seq[i + no_repeat_ngram_size - 1]
                    )

            for token_id in banned:
                logits[0, token_id] = float("-inf")

        # --------------------------------------------------------
        # Sample
        # --------------------------------------------------------

        probs = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

        # --------------------------------------------------------
        # Stop at EOT
        # --------------------------------------------------------

        if idx_next.item() == EOT_TOKEN:
            break

    return idx


# ============================================================
# Chat loop
# ============================================================

while True:

    prompt = input("You: ").strip()

    if prompt.lower() in ["exit", "quit"]:
        break

    if not prompt:
        continue

    # IMPORTANT:
    # This matches the format used during SFT.
    formatted_prompt = f"You: {prompt}\nHimeko:"

    tokens = enc.encode_ordinary(formatted_prompt)

    idx = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device
    )

    output = generate(
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        no_repeat_ngram_size=3
    )

    # --------------------------------------------------------
    # Decode only the newly generated tokens
    # --------------------------------------------------------

    generated_tokens = output[0].tolist()

    new_tokens = generated_tokens[len(tokens):]

    # Remove EOT if present
    if EOT_TOKEN in new_tokens:
        new_tokens = new_tokens[
            :new_tokens.index(EOT_TOKEN)
        ]

    response = enc.decode(new_tokens).strip()

    print(f"Himeko: {response}")
    print()
