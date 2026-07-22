"""
predict.py

Purpose
-------
Command-line inference script for the Pakistani Currency Recognition System.
Given a path to an image, loads the trained model and class mapping, runs a
single prediction, and prints the result -- predicted denomination,
confidence, and the top-3 candidates.

This script is intentionally decoupled from Streamlit (see app.py for the
web interface). Keeping inference logic here, independent of any UI
framework, means it can be reused by the web app, tested from the command
line, or eventually wrapped in an API -- without dragging Streamlit in as
a dependency for something that has nothing to do with rendering a UI.

Usage:
    python predict.py path/to/image.jpg

Design decisions worth understanding
-------------------------------------
1. Nothing about the class labels is hardcoded. models/class_indices.json
   (written by train.py) is the single source of truth for how model
   output indices map to denomination names. If train.py is ever re-run
   on a different class list -- more denominations, a different split --
   this script adapts automatically without any code changes here.

2. JSON always stores keys as strings, even if they started as ints. The
   "index_to_class" mapping in class_indices.json therefore comes back as
   {"0": "10", "1": "100", ...}. We convert the model's predicted index to
   a string before doing the lookup -- forgetting this is a common, silent
   bug in inference scripts.

3. Preprocessing here must exactly mirror what train.py used: RGB
   conversion, 224x224 resize, and MobileNetV2's preprocess_input (which
   scales pixel values to the [-1, 1] range the backbone was trained on).
   Any mismatch between training-time and inference-time preprocessing is
   one of the most common sources of "the model works in training but
   seems dumb in production" bugs.

4. Errors are handled at the boundaries (missing file, unreadable image,
   missing model/mapping files) -- but as RAISED EXCEPTIONS, not as
   sys.exit() calls inside the library functions themselves. Only main()
   (the CLI entry point) decides to print a message and exit the process.
   This matters because app.py (the Streamlit UI) imports and reuses these
   same functions -- if load_trained_model() called sys.exit() on a bad
   path, one unlucky user hitting an error would silently kill the entire
   Streamlit server for every other user connected to it. Library code
   should raise; only a true CLI entry point should ever call sys.exit().

5. preprocess_image() accepts EITHER a filesystem path (str/Path, for CLI
   usage) OR a file-like object (e.g. Streamlit's UploadedFile, which
   behaves like a BytesIO stream). Pillow's Image.open() already supports
   both transparently, so this one function serves both app.py and the
   CLI without any duplicated preprocessing logic -- exactly the kind of
   reuse this module is designed to provide.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = Path("models/currency_model.keras")
CLASS_INDICES_PATH = Path("models/class_indices.json")

IMG_SIZE = (224, 224)   # must match the input size used in train.py
TOP_K = 3               # number of top predictions to display


# ----------------------------------------------------------------------
# Step 1: Load the trained model
# ----------------------------------------------------------------------
def load_trained_model(model_path: Path):
    """Load the saved Keras model from disk.

    Raises FileNotFoundError / RuntimeError instead of exiting the process
    directly -- see design decision 4 in the module docstring. Callers
    (main() below, or app.py) decide what to do with the failure: the CLI
    exits with a message; the Streamlit app shows an st.error() and keeps
    serving other users.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run train.py first to produce this file."
        )

    try:
        model = load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}': {e}") from e

    return model


# ----------------------------------------------------------------------
# Step 2 & 3: Load and reverse the class mapping
# ----------------------------------------------------------------------
def load_index_to_class(class_indices_path: Path) -> dict:
    """Load class_indices.json and return the index -> class-name mapping.

    train.py already saves both directions ("class_to_index" and
    "index_to_class") to avoid re-deriving this here, but we defensively
    rebuild "index_to_class" from "class_to_index" if the expected key is
    somehow missing -- e.g. from an older or hand-edited version of the
    file -- rather than crashing on a KeyError.
    """
    if not class_indices_path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found at '{class_indices_path}'. "
            f"Run train.py first to produce this file."
        )

    try:
        payload = json.loads(class_indices_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse '{class_indices_path}': {e}") from e

    if "index_to_class" in payload:
        index_to_class = payload["index_to_class"]
    elif "class_to_index" in payload:
        # Rebuild it: {"10": 0, "100": 1, ...} -> {"0": "10", "1": "100", ...}
        index_to_class = {str(v): k for k, v in payload["class_to_index"].items()}
    else:
        raise ValueError(
            f"'{class_indices_path}' does not contain a recognizable class mapping."
        )

    return index_to_class


# ----------------------------------------------------------------------
# Step 4-5: CLI argument + path validation
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the denomination of a Pakistani currency note from an image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to classify.",
    )
    return parser.parse_args()


def validate_image_path(image_path: Path) -> None:
    """Confirm the given path exists and points to a file (not a directory)
    before we ever attempt to open it with Pillow.

    This function is CLI-only (app.py never has a filesystem path to
    validate -- it receives an in-memory uploaded file instead), so it's
    the one function in this module that's fine to leave path-specific.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

    if not image_path.is_file():
        raise ValueError(f"'{image_path}' is not a file.")


# ----------------------------------------------------------------------
# Step 6-9: Load and preprocess the image
# ----------------------------------------------------------------------
def preprocess_image(image_source: Union[str, Path, "object"]) -> np.ndarray:
    """Open the image, convert to RGB, resize, and apply MobileNetV2's
    preprocessing -- producing a batch of size 1 ready for model.predict().

    image_source can be a filesystem path (str/Path, used by the CLI) OR
    a file-like object such as Streamlit's UploadedFile (used by app.py).
    Pillow's Image.open() accepts both transparently, which is what lets
    app.py reuse this exact function with zero duplicated preprocessing
    code -- see design decision 5 in the module docstring.

    Converting to RGB explicitly matters because uploaded images can be
    grayscale, RGBA (with an alpha channel), or palette-based -- feeding
    anything other than 3-channel RGB into a model expecting (224, 224, 3)
    would raise a shape error or silently corrupt the color channels.

    Raises ValueError if the source isn't a readable image -- the caller
    (CLI main(), or app.py) decides how to report that to the user.
    """
    try:
        with Image.open(image_source) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE)
            image_array = np.array(img, dtype=np.float32)
    except UnidentifiedImageError as e:
        raise ValueError(f"'{image_source}' does not appear to be a valid image file.") from e
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}") from e

    # Add a batch dimension: model.predict() expects shape (batch, H, W, C),
    # not a single (H, W, C) image.
    image_array = np.expand_dims(image_array, axis=0)

    # Scale pixel values exactly as train.py did -- MobileNetV2's
    # preprocess_input maps [0, 255] pixels to [-1, 1], matching what the
    # ImageNet-pretrained backbone expects as input.
    image_array = preprocess_input(image_array)

    return image_array


# ----------------------------------------------------------------------
# Step 10-12: Predict and report results
# ----------------------------------------------------------------------
def predict_denomination(model, image_array: np.ndarray, index_to_class: dict) -> dict:
    """Run inference and return the full set of results needed for
    reporting: the top prediction plus the top-K ranked list.
    """
    # predictions shape: (1, num_classes) -- one row since we passed a
    # single-image batch.
    predictions = model.predict(image_array, verbose=0)
    probabilities = predictions[0]

    # np.argsort sorts ascending by default; [::-1] reverses to descending
    # so index 0 is the most confident class.
    ranked_indices = np.argsort(probabilities)[::-1]

    top_k_indices = ranked_indices[:TOP_K]
    top_k_results = [
        {
            "label": index_to_class[str(idx)],
            "confidence": float(probabilities[idx]),
        }
        for idx in top_k_indices
    ]

    return {
        "predicted_label": top_k_results[0]["label"],
        "predicted_confidence": top_k_results[0]["confidence"],
        "top_k": top_k_results,
    }


def print_results(result: dict) -> None:
    sep = "-" * 50
    print(sep)
    print("PREDICTION RESULT")
    print(sep)
    print(f"Predicted denomination: Rs. {result['predicted_label']}")
    print(f"Confidence: {result['predicted_confidence'] * 100:.2f}%")
    print(sep)

    print(f"Top {len(result['top_k'])} predictions:")
    for rank, entry in enumerate(result["top_k"], start=1):
        print(f"  {rank}. Rs. {entry['label']:<8} - {entry['confidence'] * 100:.2f}%")
    print(sep)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    """CLI entry point. This is the ONLY place in the module that decides
    to print an error and exit the process -- every function it calls
    raises exceptions instead, so the same functions remain safe to import
    and reuse inside a long-running process like app.py (see design
    decision 4 in the module docstring).
    """
    args = parse_args()
    image_path = Path(args.image_path)

    try:
        validate_image_path(image_path)
        model = load_trained_model(MODEL_PATH)
        index_to_class = load_index_to_class(CLASS_INDICES_PATH)
        image_array = preprocess_image(image_path)
        result = predict_denomination(model, image_array, index_to_class)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print_results(result)


if __name__ == "__main__":
    main()