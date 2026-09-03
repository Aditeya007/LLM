import os
import time
import numpy as np
import torch

from model import GPT, GPTConfig


# ============================================================
# CONFIG
# ============================================================

BASE_CKPT = "ckpt_base.pt"
SFT_CKPT = "sft_ckpt.pt"

TRAIN_INPUTS = "sft_train_inputs.npy"
TRAIN_LABELS = "sft_train_labels.npy"

VAL_INPUTS = "sft_val_inputs.npy"
VAL_LABELS = "sft_val_labels.npy"

BLOCK_SIZE = 256

micro_batch_size = 4
gradient_accumulation_steps = 4

# IMPORTANT:
# Much smaller than your pretraining LR.
learning_rate = 1e-5

weight_decay = 0.01
grad_clip = 1.0

max_iters = 1000

eval_interval = 100
eval_iters = 10

ckpt_interval = 100


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = (
    torch.bfloat16
    if device == "cuda"
    else torch.float32
)


# ============================================================
# DATA
# ============================================================

train_inputs = np.load(TRAIN_INPUTS)
train_labels = np.load(TRAIN_LABELS)

val_inputs = np.load(VAL_INPUTS)
val_labels = np.load(VAL_LABELS)

print(f"Train examples: {len(train_inputs)}")
print(f"Val examples:   {len(val_inputs)}")


def get_batch(split):

    if split == "train":
        inputs = train_inputs
        labels = train_labels
    else:
        inputs = val_inputs
        labels = val_labels

    ix = np.random.randint(
        0,
        len(inputs),
        size=micro_batch_size
    )

    x = torch.from_numpy(
        inputs[ix]
    ).long()

    y = torch.from_numpy(
        labels[ix]
    ).long()

    if device == "cuda":
        x = x.pin_memory().to(
            device,
            non_blocking=True
        )

        y = y.pin_memory().to(
            device,
            non_blocking=True
        )

    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss(model):

    model.eval()

    out = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = get_batch(split)

            with torch.autocast(
                device_type=device,
                dtype=dtype,
                enabled=(device == "cuda")
            ):

                _, loss = model(x, y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Using device: {device}, "
        f"dtype: {dtype}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    config = GPTConfig()

    # IMPORTANT:
    # The model itself remains a 512-token model.
    # We're simply feeding it 256-token SFT sequences.
    #
    # Do NOT change config.block_size here because the
    # checkpoint contains a 512-position embedding.
    # --------------------------------------------------------

    model = GPT(config).to(device)

    # --------------------------------------------------------
    # Load pretrained 40k checkpoint
    # --------------------------------------------------------

    print()
    print(f"Loading base model from {BASE_CKPT}...")

    checkpoint = torch.load(
        BASE_CKPT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print("Base model loaded.")
    print(
        f"Base checkpoint iteration: "
        f"{checkpoint.get('iter', 'unknown')}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Starting SFT...")
    print()

    t0 = time.time()

    for it in range(max_iters):

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0

        for micro_step in range(
            gradient_accumulation_steps
        ):

            x, y = get_batch("train")

            with torch.autocast(
                device_type=device,
                dtype=dtype,
                enabled=(device == "cuda")
            ):

                _, loss = model(x, y)

                loss = (
                    loss /
                    gradient_accumulation_steps
                )

            loss.backward()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if it % 10 == 0:

            elapsed = time.time() - t0

            print(
                f"iter {it}/{max_iters} | "
                f"loss "
                f"{total_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if it % eval_interval == 0:

            losses = estimate_loss(model)

            elapsed = time.time() - t0

            print(
                f"eval {it}: "
                f"train loss "
                f"{losses['train']:.4f}, "
                f"val loss "
                f"{losses['val']:.4f}, "
                f"{elapsed:.1f}s elapsed"
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            it % ckpt_interval == 0
            and it > 0
        ):

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": it,
                    "config": config,
                },
                SFT_CKPT
            )

            print(
                f"Saved SFT checkpoint "
                f"at iter {it}"
            )

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": max_iters - 1,
            "config": config,
        },
        SFT_CKPT
    )

    print()
    print("SFT complete.")
    print(
        f"Final checkpoint saved to "
        f"{SFT_CKPT}"
    )


if __name__ == "__main__":
    main()