import os
from datasets import load_dataset

OUTPUT_FILE = "data.txt"
TARGET_GB = 10
TARGET_BYTES = TARGET_GB * 1024 ** 3

def daily_dialog_text(ex):
    return " ".join(ex["dialog"])

def dialogsum_text(ex):
    return ex["dialogue"]

def fineweb_text(ex):
    return ex["text"]

SOURCES = [
    {"name": "li2017dailydialog/daily_dialog", "config": None, "extract": daily_dialog_text},
    {"name": "knkarthick/dialogsum", "config": None, "extract": dialogsum_text},
    {"name": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "extract": fineweb_text},
]

def run():
    bytes_written = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for source in SOURCES:
            if bytes_written >= TARGET_BYTES:
                break

            print(f"Starting source: {source['name']}")

            try:
                if source["config"]:
                    ds = load_dataset(source["name"], source["config"], split="train", streaming=True)
                else:
                    ds = load_dataset(source["name"], split="train", streaming=True, trust_remote_code=True)
            except Exception as e:
                print(f"Skipping {source['name']}, failed to load: {e}")
                continue

            source_bytes = 0
            n = 0

            for example in ds:
                try:
                    text = source["extract"](example)
                except Exception:
                    continue

                if not text:
                    continue

                out.write(text)
                out.write("\n\n")

                added = len(text.encode("utf-8")) + 2
                bytes_written += added
                source_bytes += added
                n += 1

                if n % 5000 == 0:
                    print(f"{source['name']} | {n} examples | {source_bytes/1e9:.3f} GB this source | {bytes_written/1e9:.3f} GB total")

                if bytes_written >= TARGET_BYTES:
                    break

            print(f"Finished {source['name']}: {source_bytes/1e9:.3f} GB, {n} examples")

    print(f"Done. Total written: {bytes_written/1e9:.3f} GB")

if __name__ == "__main__":
    run()