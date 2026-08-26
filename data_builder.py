"""
Streams a HuggingFace text dataset and writes out ~TARGET_GB of raw text
to a single .txt file. Doesn't download the full dataset — stops as soon
as it hits the target size.

pip install datasets
"""

from datasets import load_dataset

# ---- config ----
DATASET_NAME = "HuggingFaceFW/fineweb-edu"   # swap for "Skylion007/openwebtext" if you want raw web text instead
DATASET_CONFIG = "sample-10BT"                # fineweb-edu ships pre-made smaller samples, this one's plenty big enough
TARGET_GB = 4
OUTPUT_FILE = "data.txt"
# ----------------

TARGET_BYTES = TARGET_GB * 1024 ** 3

def main():
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

    bytes_written = 0
    n_examples = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in ds:
            text = example["text"]
            f.write(text)
            f.write("\n\n")  # separator between documents

            bytes_written += len(text.encode("utf-8")) + 2
            n_examples += 1

            if n_examples % 2000 == 0:
                gb_so_far = bytes_written / (1024 ** 3)
                print(f"{n_examples} docs, {gb_so_far:.2f} GB written")

            if bytes_written >= TARGET_BYTES:
                break

    final_gb = bytes_written / (1024 ** 3)
    print(f"\nDone. Wrote {final_gb:.2f} GB across {n_examples} docs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()