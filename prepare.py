import os
import numpy as np
import tiktoken

INPUT_FILE = "data.txt"
TRAIN_OUT = "train.bin"
VAL_OUT = "val.bin"
VAL_FRACTION = 0.1
CHUNK_CHARS = 20_000_000

def main():
    enc = tiktoken.get_encoding("gpt2")
    all_ids = []
    processed_chars = 0
    chunk_num = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_CHARS)
            if not chunk:
                break
            ids = enc.encode_ordinary(chunk)
            all_ids.append(np.array(ids, dtype=np.uint16))
            processed_chars += len(chunk)
            chunk_num += 1
            print(f"chunk {chunk_num} | {processed_chars:,} chars processed")

    ids = np.concatenate(all_ids)
    print(f"Total tokens: {len(ids):,}")

    split_point = int(len(ids) * (1 - VAL_FRACTION))
    train_ids = ids[:split_point]
    val_ids = ids[split_point:]

    train_ids.tofile(TRAIN_OUT)
    val_ids.tofile(VAL_OUT)

    print(f"train.bin: {len(train_ids):,} tokens ({os.path.getsize(TRAIN_OUT) / 1e6:.1f} MB)")
    print(f"val.bin:   {len(val_ids):,} tokens ({os.path.getsize(VAL_OUT) / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()