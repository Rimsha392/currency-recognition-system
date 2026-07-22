"""
utils/merge_dataset.py
 
Purpose
-------
inspect_dataset.py confirmed the raw data is clean enough to move forward.
Per the inspection recommendation, we now merge each denomination's
'<denom>_front' and '<denom>_back' folders into a single '<denom>' class,
because our actual task is denomination recognition, not side recognition.
 
Critically, this script COPIES files -- it never touches or moves the
original dataset/ folders. That's a deliberate safety decision: if a future
version of this project ever wants front/back classification (e.g. guiding
a visually impaired user to flip a note), the raw, unmerged data is still
sitting there untouched. Merged data is a derived artifact, not a
replacement for the source of truth.
 
Run with (from the project root):
    python utils/merge_dataset.py
 
Output:
    dataset/merged/10/, dataset/merged/20/, ... dataset/merged/5000/
    merge_report.txt
"""
 
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
 
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Paths are relative to wherever the script is run from (the project root),
# using pathlib so this works identically on Windows/macOS/Linux.
DATASET_DIR = Path("dataset/data-rescaled")
MERGED_DIR = Path("dataset/merged")
REPORT_PATH = Path("merge_report.txt")
 
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
# Suffixes that identify which side of the note a folder represents, and
# the filename prefix each maps to. Using an explicit mapping (rather than
# just stripping "_front"/"_back") means the prefix logic is obvious and
# easy to extend if a third category ever shows up.
SIDE_PREFIXES = {
    "front": "front_",
    "back": "back_",
}
 
 
# ----------------------------------------------------------------------
# Step 1: Discover source class folders and figure out denomination + side
# ----------------------------------------------------------------------
def parse_class_folder(folder_name: str) -> tuple[str, str] | None:
    """Given a folder name like '100_front', return ('100', 'front').
 
    Returns None if the folder name doesn't match the expected
    '<denom>_front' / '<denom>_back' pattern -- we don't want to silently
    merge folders we don't recognize.
    """
    for side in SIDE_PREFIXES:
        suffix = f"_{side}"
        if folder_name.endswith(suffix):
            denom = folder_name[: -len(suffix)]
            return denom, side
    return None
 
 
def get_source_class_dirs(dataset_dir: Path) -> list[Path]:
    """Return every class subfolder in dataset/, excluding 'merged' itself
    (so re-running this script doesn't try to merge its own output).
    """
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)
 
    return sorted([
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name != "merged"
    ])
 
 
# ----------------------------------------------------------------------
# Step 2: Merge
# ----------------------------------------------------------------------
def merge_dataset(source_dirs: list[Path]) -> dict:
    """Copy every valid image from each front/back folder into its merged
    denomination folder, prefixing filenames to avoid collisions.
 
    Returns a stats dict used to build the report.
    """
    # merged_counts: denom -> total copied images
    merged_counts = defaultdict(int)
    # breakdown: denom -> {'front': n, 'back': n}
    breakdown = defaultdict(lambda: defaultdict(int))
    # per-folder skipped (unrecognized name) and per-file skipped (unreadable/bad extension)
    skipped_folders = []
    skipped_files = []
 
    for class_dir in source_dirs:
        parsed = parse_class_folder(class_dir.name)
        if parsed is None:
            skipped_folders.append(class_dir.name)
            print(f"WARNING: '{class_dir.name}' doesn't match '<denom>_front/back' "
                  f"pattern -- skipping.")
            continue
 
        denom, side = parsed
        prefix = SIDE_PREFIXES[side]
 
        dest_dir = MERGED_DIR / denom
        dest_dir.mkdir(parents=True, exist_ok=True)
 
        image_files = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
 
        print(f"Merging '{class_dir.name}' -> 'merged/{denom}/' "
              f"({len(image_files)} files, prefix='{prefix}')")
 
        for src_path in image_files:
            new_name = f"{prefix}{src_path.name}"
            dest_path = dest_dir / new_name
 
            try:
                # copy2 preserves metadata (timestamps) -- pure copy, source untouched.
                shutil.copy2(src_path, dest_path)
                merged_counts[denom] += 1
                breakdown[denom][side] += 1
            except Exception as e:
                skipped_files.append((str(src_path), str(e)))
                print(f"  WARNING: failed to copy '{src_path.name}': {e}")
 
    return {
        "merged_counts": dict(merged_counts),
        "breakdown": {k: dict(v) for k, v in breakdown.items()},
        "skipped_folders": skipped_folders,
        "skipped_files": skipped_files,
    }
 
 
# ----------------------------------------------------------------------
# Step 3: Report
# ----------------------------------------------------------------------
def build_report(stats: dict) -> str:
    out = []
    sep = "-" * 70
 
    out.append(sep)
    out.append("DATASET MERGE REPORT")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(f"Source: {DATASET_DIR.resolve()}")
    out.append(f"Destination: {MERGED_DIR.resolve()}")
    out.append(sep)
 
    out.append("\nMERGED CLASS COUNTS")
    out.append(sep)
    out.append(f"{'Class':<15}{'Front':>8}{'Back':>8}{'Total':>8}")
    out.append(sep)
 
    total_images = 0
    for denom in sorted(stats["merged_counts"], key=lambda d: int(d) if d.isdigit() else d):
        front_n = stats["breakdown"].get(denom, {}).get("front", 0)
        back_n = stats["breakdown"].get(denom, {}).get("back", 0)
        total_n = stats["merged_counts"][denom]
        total_images += total_n
        out.append(f"{denom:<15}{front_n:>8}{back_n:>8}{total_n:>8}")
 
    out.append(sep)
    out.append(f"Total merged classes: {len(stats['merged_counts'])}")
    out.append(f"Total merged images: {total_images}")
    out.append(sep)
 
    out.append("\nSKIPPED SOURCE FOLDERS (name didn't match '<denom>_front/back')")
    out.append(sep)
    if stats["skipped_folders"]:
        for name in stats["skipped_folders"]:
            out.append(f"  {name}")
    else:
        out.append("None.")
    out.append(sep)
 
    out.append("\nSKIPPED FILES (copy failed)")
    out.append(sep)
    if stats["skipped_files"]:
        for path, error in stats["skipped_files"]:
            out.append(f"  {path}  -- {error}")
    else:
        out.append("None.")
    out.append(sep)
 
    out.append("\nNOTE: Original dataset/<denom>_front and <denom>_back folders are")
    out.append("untouched -- all files above were COPIED, not moved. The merged")
    out.append("folders under dataset/merged/ are what train.py will consume next.")
    out.append(sep)
 
    return "\n".join(out)
 
 
# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Starting dataset merge...\n")
 
    source_dirs = get_source_class_dirs(DATASET_DIR)
    stats = merge_dataset(source_dirs)
    report_text = build_report(stats)
 
    print("\n" + report_text)
 
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to: {REPORT_PATH.resolve()}")
 
 
if __name__ == "__main__":
    main()
 