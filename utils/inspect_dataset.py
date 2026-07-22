"""
inspect_dataset.py

Purpose
-------
Before we write a single line of training code, we need to know EXACTLY what
we're working with. This script performs a full audit of the raw dataset
directory and produces:

    1. Per-class image counts
    2. Corruption detection (unreadable / truncated files)
    3. Image dimension + color mode statistics (helps us decide input size,
       whether we need channel conversion, etc.)
    4. Exact-duplicate detection via file hashing (duplicates across folders
       or within a folder can leak between train/val/test splits later and
       silently inflate validation accuracy -- we want to catch that NOW,
       before splitting)
    5. Class balance statistics (max/min/mean) so we know if some
       denominations are under-represented
    6. A data-driven recommendation on whether to keep "front"/"back" as
       separate classes or merge them per denomination

Run with:
    python inspect_dataset.py

Output:
    - A summary printed to the console
    - A full report saved to dataset_report.txt
"""

import hashlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Using pathlib.Path throughout instead of hardcoded string paths keeps this
# script portable across Windows / macOS / Linux without any changes.
DATASET_DIR = Path("dataset/data-rescaled")
REPORT_PATH = Path("dataset_report.txt")

# Only these extensions are treated as candidate images. Anything else
# (e.g. .txt, .DS_Store, .ini) inside a class folder is silently ignored,
# not counted as corrupted -- it simply isn't image data.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Images with either dimension below this are flagged as "unusually small".
# 100px was chosen because MobileNetV2 will resize everything to 224x224
# anyway -- an image far smaller than that was almost certainly upscaled
# from a low-quality source and will contribute more noise than signal.
MIN_DIMENSION_THRESHOLD = 100

# Minimum images we'd want in a class before we trust it to train reliably,
# even with augmentation. This is a heuristic, not a hard rule -- we surface
# it in the report so the user can make the final call.
MIN_RELIABLE_CLASS_SIZE = 80


# ----------------------------------------------------------------------
# Step 1: Locate class folders
# ----------------------------------------------------------------------
def get_class_dirs(dataset_dir: Path) -> list[Path]:
    """Return every immediate subdirectory of dataset_dir, sorted by name.

    Each subdirectory is treated as one class (e.g. '100_front', '500_back').
    We don't recurse deeper -- if the dataset ever gains nested subfolders,
    that's a structural issue worth surfacing, not silently flattening.
    """
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist.")
        sys.exit(1)

    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

    if not class_dirs:
        print(f"ERROR: No class subfolders found inside '{dataset_dir}'.")
        sys.exit(1)

    return class_dirs


# ----------------------------------------------------------------------
# Step 2: Inspect a single image (validity, dimensions, mode, hash)
# ----------------------------------------------------------------------
def inspect_image(path: Path) -> dict:
    """Open an image file and extract diagnostic info.

    We do two separate passes with Pillow:
      - `verify()` checks the file isn't truncated/corrupted, but it closes
        the file handle afterwards and doesn't reliably return usable image
        objects -- so we can't pull width/height/mode from it directly.
      - A second `Image.open()` gets us the actual dimensions and mode
        once we know the file is valid.

    This two-step approach is the standard, reliable way to catch corrupted
    images with Pillow without false positives.
    """
    result = {
        "path": path,
        "valid": False,
        "width": None,
        "height": None,
        "mode": None,
        "md5": None,
        "error": None,
    }

    # --- Pass 1: corruption check ---
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        result["error"] = str(e)
        return result  # corrupted -- no point trying to read further

    # --- Pass 2: read actual metadata (only if pass 1 succeeded) ---
    try:
        with Image.open(path) as img:
            result["width"], result["height"] = img.size
            result["mode"] = img.mode
        result["valid"] = True
    except Exception as e:
        result["error"] = str(e)
        return result

    # --- MD5 hash of raw file bytes, for exact-duplicate detection ---
    # This catches identical files (e.g. the same photo copied into two
    # folders by mistake). It will NOT catch near-duplicates (same note,
    # slightly different crop/lighting) -- that requires perceptual hashing
    # (e.g. the `imagehash` library), which we're deliberately leaving out
    # to keep dependencies minimal for the MVP. Worth revisiting later.
    try:
        with open(path, "rb") as f:
            result["md5"] = hashlib.md5(f.read()).hexdigest()
    except Exception:
        pass  # non-fatal -- duplicate detection just skips this file

    return result


# ----------------------------------------------------------------------
# Step 3: Scan the whole dataset
# ----------------------------------------------------------------------
def scan_dataset(class_dirs: list[Path]) -> dict:
    """Walk every class folder and inspect every image inside it.

    Returns a dict keyed by class name, each value holding the list of
    per-image inspection results for that class.
    """
    dataset_info = {}

    for class_dir in class_dirs:
        image_paths = [
            p for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        print(f"Scanning '{class_dir.name}'... ({len(image_paths)} candidate files)")

        results = [inspect_image(p) for p in image_paths]
        dataset_info[class_dir.name] = results

    return dataset_info


# ----------------------------------------------------------------------
# Step 4: Aggregate statistics
# ----------------------------------------------------------------------
def compute_statistics(dataset_info: dict) -> dict:
    """Compute all the summary numbers the report needs."""

    class_counts = {}          # class_name -> valid image count
    corrupted_images = []      # list of (class_name, path, error)
    dimension_counter = defaultdict(int)   # (w, h) -> count
    mode_counter = defaultdict(int)        # color mode -> count
    small_images = []          # list of (class_name, path, w, h)
    hash_to_paths = defaultdict(list)      # md5 -> list of paths

    total_valid = 0
    total_corrupted = 0

    for class_name, results in dataset_info.items():
        valid_count = 0
        for r in results:
            if not r["valid"]:
                corrupted_images.append((class_name, r["path"], r["error"]))
                total_corrupted += 1
                continue

            valid_count += 1
            total_valid += 1

            dimension_counter[(r["width"], r["height"])] += 1
            mode_counter[r["mode"]] += 1

            if r["width"] < MIN_DIMENSION_THRESHOLD or r["height"] < MIN_DIMENSION_THRESHOLD:
                small_images.append((class_name, r["path"], r["width"], r["height"]))

            if r["md5"] is not None:
                hash_to_paths[r["md5"]].append(r["path"])

        class_counts[class_name] = valid_count

    # Duplicate groups: only keep hashes that appear more than once
    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    return {
        "class_counts": class_counts,
        "corrupted_images": corrupted_images,
        "total_valid": total_valid,
        "total_corrupted": total_corrupted,
        "dimension_counter": dimension_counter,
        "mode_counter": mode_counter,
        "small_images": small_images,
        "duplicate_groups": duplicate_groups,
    }


# ----------------------------------------------------------------------
# Step 5: Balance analysis
# ----------------------------------------------------------------------
def analyze_balance(class_counts: dict) -> dict:
    """Compute how balanced or imbalanced the dataset is across classes."""
    counts = list(class_counts.values())

    if not counts:
        return {}

    largest = max(counts)
    smallest = min(counts)
    mean = statistics.mean(counts)
    stdev = statistics.stdev(counts) if len(counts) > 1 else 0.0

    # Ratio of largest to smallest class -- a common, intuitive way to
    # describe imbalance. A ratio near 1.0 is well-balanced; ratios above
    # ~3-4x typically warrant class weighting or targeted data collection.
    imbalance_ratio = (largest / smallest) if smallest > 0 else float("inf")

    return {
        "largest": largest,
        "smallest": smallest,
        "mean": mean,
        "stdev": stdev,
        "imbalance_ratio": imbalance_ratio,
    }


# ----------------------------------------------------------------------
# Step 6: Front/back merge recommendation
# ----------------------------------------------------------------------
def generate_recommendation(class_counts: dict) -> str:
    """Decide, using the actual counts, whether to recommend merging
    '<denom>_front' / '<denom>_back' folders into a single '<denom>' class.

    ML reasoning behind this decision:

    1. Task definition matters most. Our end goal is DENOMINATION
       recognition (is this a Rs.500 note?), not SIDE recognition (is this
       the front or the back?). If we keep front/back as separate classes,
       we are training the model to solve a harder, different problem than
       the one we actually care about -- and at inference time, a user's
       photo of the back of a note would need to be correctly mapped back
       to its denomination anyway (e.g. by merging '500_front' and
       '500_back' predictions in post-processing). It's simpler and more
       correct to let the model learn "denomination, regardless of side"
       directly.

    2. Sample efficiency. With a small dataset, splitting each denomination
       into two classes halves the number of examples the model sees per
       *effective* label, right when we need more data per class, not less.
       Merging front and back doubles the training signal per denomination
       without collecting a single new image.

    3. Visual consistency within a merged class. Front and back designs of
       a given Pakistani note differ in imagery (portrait vs. monument) but
       share denomination-specific cues that matter for classification --
       size, color palette, printed number, security thread pattern, border
       design. A CNN with transfer-learned features (edges, colors,
       textures) is well suited to learning "this color/texture/number
       combination reliably means Rs.500" even when the specific artwork on
       each side differs. This is precisely the kind of intra-class visual
       variation that data augmentation and transfer learning are designed
       to handle.

    4. When you WOULD keep them separate: if the eventual product feature
       explicitly needs to tell front from back (e.g. an app that guides a
       visually impaired user to flip the note), then front/back becomes a
       genuinely separate, useful label -- but that's a different problem
       than the one scoped for this MVP.

    We still base the final recommendation on the actual numbers: if
    merging would create severe imbalance (e.g. one denomination's front+back
    combined is still tiny compared to others), that gets flagged too.
    """
    # Group counts by denomination, ignoring the front/back suffix
    denom_groups = defaultdict(int)
    for class_name, count in class_counts.items():
        denom = class_name.replace("_front", "").replace("_back", "")
        denom_groups[denom] += count

    lines = []
    lines.append("Per-side counts vs. merged (denomination-only) counts:")
    lines.append(f"  {'Class':<15}{'Images':>10}")
    for class_name, count in sorted(class_counts.items()):
        lines.append(f"  {class_name:<15}{count:>10}")

    lines.append("")
    lines.append(f"  {'Denomination (merged)':<24}{'Images':>10}")
    for denom, count in sorted(denom_groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        flag = "  <-- below reliable minimum" if count < MIN_RELIABLE_CLASS_SIZE else ""
        lines.append(f"  {denom:<24}{count:>10}{flag}")

    lines.append("")
    lines.append("RECOMMENDATION: Merge '<denom>_front' and '<denom>_back' into a single")
    lines.append("'<denom>' class for training (e.g. '500_front' + '500_back' -> '500').")
    lines.append("")
    lines.append("Reasoning:")
    lines.append("  1. The task is denomination recognition, not side recognition --")
    lines.append("     merging aligns the label space with the actual product goal.")
    lines.append("  2. It doubles effective samples per class with zero new data collection,")
    lines.append("     which matters given how small each per-side folder is.")
    lines.append("  3. Denomination-specific visual cues (color, size, printed number,")
    lines.append("     security features) are shared across front/back and are exactly")
    lines.append("     what transfer-learned CNN features are good at picking up on,")
    lines.append("     even amid the artwork differences between sides.")
    lines.append("  4. Only keep front/back separate if a future feature explicitly")
    lines.append("     requires knowing which side is shown -- out of scope for this MVP.")

    under_threshold = [d for d, c in denom_groups.items() if c < MIN_RELIABLE_CLASS_SIZE]
    if under_threshold:
        lines.append("")
        lines.append(f"  CAUTION: even after merging, these denominations remain under the")
        lines.append(f"  {MIN_RELIABLE_CLASS_SIZE}-image reliability heuristic: {', '.join(sorted(under_threshold))}.")
        lines.append("  Consider dropping them from V1 scope or sourcing more images before training.")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Step 7: Report building (console + file), shared formatting
# ----------------------------------------------------------------------
def build_report(dataset_info: dict, stats: dict, balance: dict) -> str:
    out = []
    sep = "-" * 70

    out.append(sep)
    out.append("DATASET INSPECTION REPORT")
    out.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.append(sep)

    # --- Section 1: Class counts table ---
    out.append("\nCLASS IMAGE COUNTS")
    out.append(sep)
    out.append(f"{'Class':<20}{'Images':>10}")
    out.append(sep)
    for class_name, count in sorted(stats["class_counts"].items()):
        out.append(f"{class_name:<20}{count:>10}")
    out.append(sep)
    out.append(f"Total classes: {len(stats['class_counts'])}")
    out.append(f"Total valid images: {stats['total_valid']}")
    out.append(sep)

    # --- Section 2: Corrupted images ---
    out.append("\nCORRUPTED / UNREADABLE IMAGES")
    out.append(sep)
    if stats["corrupted_images"]:
        for class_name, path, error in stats["corrupted_images"]:
            out.append(f"[{class_name}] {path}  -- {error}")
    else:
        out.append("None found.")
    out.append(sep)
    out.append(f"Corrupted images: {stats['total_corrupted']}")
    out.append(f"Valid images: {stats['total_valid']}")
    out.append(sep)

    # --- Section 3: Balance statistics ---
    out.append("\nCLASS BALANCE STATISTICS")
    out.append(sep)
    if balance:
        out.append(f"Largest class size:  {balance['largest']}")
        out.append(f"Smallest class size: {balance['smallest']}")
        out.append(f"Mean images/class:   {balance['mean']:.1f}")
        out.append(f"Std deviation:       {balance['stdev']:.1f}")
        out.append(f"Imbalance ratio (largest/smallest): {balance['imbalance_ratio']:.2f}x")
        if balance["imbalance_ratio"] > 3:
            out.append("-> Ratio > 3x: dataset is meaningfully imbalanced. Consider class")
            out.append("   weighting during training, or augmenting minority classes more.")
        else:
            out.append("-> Ratio is within a reasonable range; standard training should be fine.")
    out.append(sep)

    # --- Section 4: Image dimensions & color modes ---
    out.append("\nIMAGE DIMENSIONS")
    out.append(sep)
    dim_items = sorted(stats["dimension_counter"].items(), key=lambda x: -x[1])
    out.append(f"Unique resolutions found: {len(dim_items)}")
    out.append("Top 10 most common resolutions (width x height : count):")
    for (w, h), count in dim_items[:10]:
        out.append(f"  {w}x{h} : {count}")
    out.append(sep)

    out.append("\nCOLOR MODES")
    out.append(sep)
    for mode, count in sorted(stats["mode_counter"].items(), key=lambda x: -x[1]):
        out.append(f"  {mode}: {count}")
    if len(stats["mode_counter"]) > 1:
        out.append("-> Multiple color modes present. All images will need explicit conversion")
        out.append("   to RGB during preprocessing so the model receives consistent channel counts.")
    out.append(sep)

    # --- Section 5: Unusually small images ---
    out.append(f"\nUNUSUALLY SMALL IMAGES (< {MIN_DIMENSION_THRESHOLD}px in either dimension)")
    out.append(sep)
    if stats["small_images"]:
        for class_name, path, w, h in stats["small_images"]:
            out.append(f"[{class_name}] {path}  ({w}x{h})")
    else:
        out.append("None found.")
    out.append(sep)
    out.append(f"Count: {len(stats['small_images'])}")
    out.append(sep)

    # --- Section 6: Duplicates ---
    out.append("\nEXACT DUPLICATE IMAGES (identical file content, by MD5 hash)")
    out.append(sep)
    if stats["duplicate_groups"]:
        for h, paths in stats["duplicate_groups"].items():
            out.append(f"Hash {h}:")
            for p in paths:
                out.append(f"  {p}")
    else:
        out.append("None found.")
    out.append(sep)
    out.append(f"Duplicate groups: {len(stats['duplicate_groups'])}")
    total_dupe_files = sum(len(v) - 1 for v in stats["duplicate_groups"].values())
    out.append(f"Redundant files (extras beyond first copy): {total_dupe_files}")
    out.append(sep)

    # --- Section 7: Recommendation ---
    out.append("\nFRONT/BACK MERGE RECOMMENDATION")
    out.append(sep)
    out.append(generate_recommendation(stats["class_counts"]))
    out.append(sep)

    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("Starting dataset inspection...\n")

    class_dirs = get_class_dirs(DATASET_DIR)
    dataset_info = scan_dataset(class_dirs)
    stats = compute_statistics(dataset_info)
    balance = analyze_balance(stats["class_counts"])

    report_text = build_report(dataset_info, stats, balance)

    # Print to console
    print("\n" + report_text)

    # Save to file
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nFull report saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
