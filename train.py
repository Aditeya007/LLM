import os
import time
import math
import numpy as np
import torch

from model import GPT, GPTConfig

# ---- training config ----
DATA_DIR = "."                     
CKPT_PATH = "ckpt.pt"

micro_batch_size = 8               
gradient_accumulation_steps = 8    
block_size = 512                   

max_iters = 20000                  
eval_interval = 250                
eval_iters = 50                    
ckpt_interval = 500               

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 500
lr_decay_iters = max_iters
grad_clip = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
# ----------------------------


def get_batch(split):
    """Load a random chunk of block_size+1 tokens (input + the target it should predict)."""
    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(DATA_DIR, filename), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (micro_batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it):
    """Linear warmup, then cosine decay down to min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Using device: {device}, dtype: {dtype}")

    config = GPTConfig()
    model = GPT(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    start_iter = 0
    if os.path.exists(CKPT_PATH):
        print(f"Resuming from {CKPT_PATH}...")
        checkpoint = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter"] + 1
        print(f"Resumed at iteration {start_iter}")

    t0 = time.time()

    for it in range(start_iter, max_iters):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(gradient_accumulation_steps):
            x, y = get_batch("train")
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / gradient_accumulation_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            dt = time.time() - t0
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {dt:.1f}s elapsed")

        if it % ckpt_interval == 0 and it > 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it,
                "config": config,
            }, CKPT_PATH)
            print(f"Saved checkpoint at iter {it}")

    # final save
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": max_iters - 1,
        "config": config,
    }, CKPT_PATH)
    print("Training complete. Final checkpoint saved.")


if __name__ == "__main__":
    main()