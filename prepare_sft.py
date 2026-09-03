import json
import random
import numpy as np
import tiktoken

INPUT_FILE = "himeko_data.json"

TRAIN_INPUTS = "sft_train_inputs.npy"
TRAIN_LABELS = "sft_train_labels.npy"

VAL_INPUTS = "sft_val_inputs.npy"
VAL_LABELS = "sft_val_labels.npy"

BLOCK_SIZE = 256
VAL_FRACTION = 0.1

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


def encode_example(ex):
    prompt = f"You: {ex['user']}\nHimeko: "

    prompt_ids = enc.encode_ordinary(prompt)
    response_ids = enc.encode_ordinary(ex["himeko"]) + [EOT]

    # Complete sequence
    tokens = prompt_ids + response_ids

    # We need at least 2 tokens because the model predicts
    # the next token.
    if len(tokens) < 2:
        raise ValueError("Example is too short.")

    # x = current tokens
    # y = next tokens
    x = tokens[:-1]
    y = tokens[1:]

    # Original token positions corresponding to y.
    # y[0] corresponds to tokens[1], etc.
    #
    # We only calculate loss on Himeko's response.
    labels = []

    for original_position, target in enumerate(y, start=1):
        if original_position < len(prompt_ids):
            labels.append(-100)
        else:
            labels.append(target)

    # Truncate to BLOCK_SIZE
    x = x[:BLOCK_SIZE]
    y = y[:BLOCK_SIZE]
    labels = labels[:BLOCK_SIZE]

    # Pad
    pad_len = BLOCK_SIZE - len(x)

    x += [EOT] * pad_len
    y += [EOT] * pad_len
    labels += [-100] * pad_len

    return x, labels


def build_split(data):
    inputs = []
    labels = []

    for ex in data:
        x, y = encode_example(ex)
        inputs.append(x)
        labels.append(y)

    return (
        np.array(inputs, dtype=np.int64),
        np.array(labels, dtype=np.int64)
    )


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total examples: {len(data)}")

    # Check lengths before splitting
    lengths = []

    for ex in data:
        prompt_ids = enc.encode_ordinary(
            f"You: {ex['user']}\nHimeko: "
        )

        response_ids = enc.encode_ordinary(
            ex["himeko"]
        ) + [EOT]

        lengths.append(len(prompt_ids) + len(response_ids))

    print(f"Max tokens: {max(lengths)}")
    print(f"Average tokens: {sum(lengths) / len(lengths):.1f}")
    print(
        f"Examples over {BLOCK_SIZE}: "
        f"{sum(x > BLOCK_SIZE for x in lengths)}"
    )

    # Deterministic shuffle
    random.seed(42)
    random.shuffle(data)

    split = int(len(data) * (1 - VAL_FRACTION))

    train_data = data[:split]
    val_data = data[split:]

    train_inputs, train_labels = build_split(train_data)
    val_inputs, val_labels = build_split(val_data)

    np.save(TRAIN_INPUTS, train_inputs)
    np.save(TRAIN_LABELS, train_labels)

    np.save(VAL_INPUTS, val_inputs)
    np.save(VAL_LABELS, val_labels)

    print()
    print(f"Train examples: {len(train_data)}")
    print(f"Val examples:   {len(val_data)}")

    print()
    print("Saved:")
    print(f"  {TRAIN_INPUTS}")
    print(f"  {TRAIN_LABELS}")
    print(f"  {VAL_INPUTS}")
    print(f"  {VAL_LABELS}")


if __name__ == "__main__":
    main()