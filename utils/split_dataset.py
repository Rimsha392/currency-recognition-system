"""
utils/split_dataset.py

Purpose
-------
Takes the merged, denomination-only dataset (dataset/merged/<denom>/) and
produces a train/validation/test split that train.py will consume directly.

Key design decisions
---------------------
1. Stratification without scikit-learn. "Stratified" just means each class
   is split independently, in the same proportion, rather than shuffling
   the whole dataset together (which could accidentally leave a class
   under-represented in one split by chance). Because our data is already
   organized into one folder per class, we get stratification for free by
   splitting each class folder's file list separately -- no need to pull in
   scikit-learn's train_test_split as an extra dependency just for this.

2. Reproducibility via a seeded random.Random(42) instance, not the global
   `random` module. Using a dedicated Random object (rather than
   random.seed(42) + random.shuffle) avoids any risk of global random state
   leaking in from other code that might run before this script, and makes
   the reproducibility guarantee local and self-contained.

3. Copy, not move. Exactly like merge_dataset.py, the merged dataset is
   left untouched. dataset/split/ is a derived, disposable artifact --
   if we ever want to change the split ratio or reshuffle, we just delete
   and regenerate dataset/split/ without needing to re-run the merge step.

4. dataset/split/ is deleted and rebuilt from scratch on every run. This
   is intentional: if we didn't do this, re-running the script after
   changing the split ratio would leave stale files from the old split
   mixed in with new ones (e.g. an image that was in "test" before could
   silently remain there even after a rerun assigns it to "train"),
   silently corrupting the split without any error being raised.

Run with (from the project root):
    python utils/split_dataset.py

Output:
    dataset/split/train/<denom>/
    dataset/split/val/<denom>/
    dataset/split/test/<denom>/
    split_report.txt
"""

import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SOURCE_DIR = Path("dataset/merged")
OUTPUT_DIR = Path("dataset/split")
REPORT_PATH = Path("split_report.txt")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 70/15/15 stratified split, as specified.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # kept explicit (rather than derived as 1 - train - val)
                    # for readability, even though it's redundant with the
                    # other two -- makes the ratios easy to scan and tweak.

RANDOM_STATE = 42

# Sanity-check target from the last known merge_dataset.py run. The script
# doesn't hardcode-trust this number -- it recomputes the actual merged
# total from disk and treats THAT as ground truth. This constant is only
# used to flag a warning if the dataset has changed size since the last
# time you looked (e.g. you added or removed images and forgot).
EXPECTED_TOTAL_IMAGES = 3611


# ----------------------------------------------------------------------
# Step 1: Reset the output directory
# ----------------------------------------------------------------------
def reset_output_dir(output_dir: Path) -> None:
    """Delete any previous dataset/split/ and recreate it fresh.

    See module docstring point 4 for why a clean rebuild matters here --
    stale files from a previous split ratio must never survive a rerun.
    """
    if output_dir.exists():
        print(f"Removing existing '{output_dir}' before rebuilding...")
        shutil.rmtree(output_dir)

    for split_name in ("train", "val", "test"):
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Step 2: Discover class folders in the merged dataset
# ----------------------------------------------------------------------
def get_class_dirs(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        print(f"ERROR: Merged dataset directory '{source_dir}' does not exist. "
              f"Run utils/merge_dataset.py first.")
        sys.exit(1)

    class_dirs = sorted(
        [p for p in source_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )

    if not class_dirs:
        print(f"ERROR: No class subfolders found inside '{source_dir}'.")
        sys.exit(1)

    return class_dirs


# ----------------------------------------------------------------------
# Step 3: Split one class's files into train/val/test
# ----------------------------------------------------------------------
def split_class_files(files: list[Path], rng: random.Random) -> dict:
    """Shuffle a class's file list deterministically, then slice it into
    train/val/test according to the configured ratios.

    Using round() for train/val counts and assigning the remainder to test
    guarantees every file is assigned exactly once and the three splits
    always sum back to the original count -- no files lost or duplicated
    to floating-point rounding.
    """
    shuffled = files.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = round(n_total * TRAIN_RATIO)
    n_val = round(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val  # remainder absorbs rounding error

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


# ----------------------------------------------------------------------
# Step 4: Process the whole dataset
# ----------------------------------------------------------------------
def process_dataset(class_dirs: list[Path]) -> dict:
    """Split and copy every class, printing progress as we go.

    Returns per-class, per-split counts for the report.
    """
    # One seeded Random instance shared across all classes. Because class
    # folders are always processed in the same sorted order, this keeps
    # the entire run fully reproducible end to end.
    rng = random.Random(RANDOM_STATE)

    # counts[class_name][split_name] = image count
    counts = defaultdict(lambda: defaultdict(int))
    total_source_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_files = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        total_source_images += len(image_files)

        print(f"Splitting class '{class_name}' ({len(image_files)} images)...")

        splits = split_class_files(image_files, rng)

        for split_name, split_files in splits.items():
            dest_dir = OUTPUT_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in split_files:
                shutil.copy2(src_path, dest_dir / src_path.name)

            counts[class_name][split_name] = len(split_files)

            print(f"  -> {split_name}: {len(split_files)}")

    return {
        "counts": {k: dict(v) for k, v in counts.items()},
        "total_source_images": total_source_images,
    }


# ----------------------------------------------------------------------
# Step 5: Report
# ----------------------------------------------------------------------
def build_report(stats: dict) -> str:
    out = []
    sep = "-" * 70

    out.append(sep)
    out.append("DATASET SPLIT REPORT")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Source: {SOURCE_DIR.resolve()}")
    out.append(f"Output: {OUTPUT_DIR.resolve()}")
    out.append(f"Split ratio (train/val/test): "
               f"{TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    out.append(f"Random state: {RANDOM_STATE}")
    out.append(sep)

    out.append("\nPER-CLASS SPLIT COUNTS")
    out.append(sep)
    out.append(f"{'Class':<12}{'Train':>10}{'Val':>10}{'Test':>10}{'Total':>10}")
    out.append(sep)

    split_totals = defaultdict(int)
    grand_total = 0

    for class_name, split_counts in stats["counts"].items():
        train_n = split_counts.get("train", 0)
        val_n = split_counts.get("val", 0)
        test_n = split_counts.get("test", 0)
        class_total = train_n + val_n + test_n

        split_totals["train"] += train_n
        split_totals["val"] += val_n
        split_totals["test"] += test_n
        grand_total += class_total

        out.append(f"{class_name:<12}{train_n:>10}{val_n:>10}{test_n:>10}{class_total:>10}")

    out.append(sep)
    out.append(f"{'TOTAL':<12}{split_totals['train']:>10}{split_totals['val']:>10}"
               f"{split_totals['test']:>10}{grand_total:>10}")
    out.append(sep)

    out.append("\nSPLIT SIZE SUMMARY")
    out.append(sep)
    out.append(f"Train images: {split_totals['train']}")
    out.append(f"Val images:   {split_totals['val']}")
    out.append(f"Test images:  {split_totals['test']}")
    out.append(f"Grand total:  {grand_total}")
    out.append(sep)

    out.append("\nVERIFICATION")
    out.append(sep)
    source_total = stats["total_source_images"]
    out.append(f"Total images found in dataset/merged/: {source_total}")
    out.append(f"Total images copied across train+val+test: {grand_total}")

    if grand_total == source_total:
        out.append("PASS: split total matches merged dataset total exactly. "
                   "No files lost or duplicated.")
    else:
        out.append(f"FAIL: mismatch of {abs(grand_total - source_total)} images! "
                   f"Investigate before proceeding to training.")

    if source_total != EXPECTED_TOTAL_IMAGES:
        out.append(f"\nNOTE: expected {EXPECTED_TOTAL_IMAGES} images based on the last "
                   f"merge_dataset.py run, but found {source_total} in dataset/merged/. "
                   f"This isn't necessarily an error -- it just means the dataset has "
                   f"changed size since that run. Update EXPECTED_TOTAL_IMAGES if this "
                   f"is intentional.")
    else:
        out.append(f"\nExpected total ({EXPECTED_TOTAL_IMAGES}) matches dataset/merged/ "
                   f"exactly.")

    out.append(sep)

    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Starting dataset split...\n")

    reset_output_dir(OUTPUT_DIR)
    class_dirs = get_class_dirs(SOURCE_DIR)
    stats = process_dataset(class_dirs)

    report_text = build_report(stats)
    print("\n" + report_text)

    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()