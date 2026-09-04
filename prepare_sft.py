import json
import random
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

BLOCK_SIZE = 256

with open("himeko_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.9)
train_data = data[:split]
val_data = data[split:]

def encode_example(ex):
    prompt_ids = enc.encode_ordinary(f"You: {ex['user']}\nHimeko: ")
    response_ids = enc.encode_ordinary(ex["himeko"]) + [eot]

    max_response_len = BLOCK_SIZE + 1 - len(prompt_ids)

    if max_response_len <= 0:
        prompt_ids = prompt_ids[:BLOCK_SIZE]
        response_ids = [eot]
    else:
        response_ids = response_ids[:max_response_len]

    full_ids = prompt_ids + response_ids

    is_response = [False] * len(prompt_ids) + [True] * len(response_ids)

    input_ids = full_ids[:-1]
    raw_targets = full_ids[1:]
    target_is_response = is_response[1:]

    labels = [
        raw_targets[i] if target_is_response[i] else -100
        for i in range(len(raw_targets))
    ]

    pad_len = BLOCK_SIZE - len(input_ids)

    input_ids = input_ids + [eot] * pad_len
    labels = labels + [-100] * pad_len

    return input_ids, labels

def build_split(split_data):
    inputs = []
    targets = []

    for ex in split_data:
        input_ids, labels = encode_example(ex)
        inputs.append(input_ids)
        targets.append(labels)

    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

train_inputs, train_labels = build_split(train_data)
val_inputs, val_labels = build_split(val_data)

np.save("sft_train_inputs.npy", train_inputs)
np.save("sft_train_labels.npy", train_labels)
np.save("sft_val_inputs.npy", val_inputs)
np.save("sft_val_labels.npy", val_labels)

print(f"train examples: {len(train_data)}")
print(f"val examples: {len(val_data)}")
print(f"train shape: {train_inputs.shape}")
print(f"val shape: {val_inputs.shape}")
