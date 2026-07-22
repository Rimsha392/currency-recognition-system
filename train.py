"""
models/train.py

Purpose
-------
Trains the currency denomination classifier using MobileNetV2 transfer
learning on the already-inspected, merged, and split dataset.

This script assumes dataset/split/{train,val,test}/<denom>/ already exists
(produced by utils/merge_dataset.py + utils/split_dataset.py). It does not
touch raw data or re-run any of the earlier pipeline steps.

Design decisions worth understanding
-------------------------------------
1. Backbone frozen, head-only training. MobileNetV2's convolutional base
   is loaded with ImageNet weights and frozen (trainable=False). We only
   train a small classification head on top. With ~3,600 images spread
   across 7 classes, fine-tuning the full backbone from the start would
   risk overfitting and would also be slower to iterate on. Unfreezing
   some backbone layers for fine-tuning is a natural V2 improvement, not
   done here -- flagged in a comment near the model-building step.

2. Augmentation on train only. Validation and test data must reflect the
   real distribution the model will see at inference time. Augmenting
   them would make val/test metrics unrealistically optimistic (or
   pessimistic) and unable to reliably signal overfitting -- augmentation
   is exclusively a training-time regularization tool.

3. No horizontal flip. The requirements mention horizontal *shift*
   (translation), not horizontal *flip* (mirroring), and that's
   intentional: a real photo of a currency note is never mirrored. If we
   flipped images during training, the model would learn to accept
   mirrored text/numbers as valid, which doesn't match real-world input
   and could hurt accuracy rather than help it.

4. class_indices.json is read directly from the training generator rather
   than assumed. Keras's flow_from_directory sorts class folder names
   *alphabetically as strings*, not numerically -- so the actual class
   order ends up like ['10', '100', '1000', '20', '5000', '500', '50'],
   NOT the numeric order ['10','20','50','100','500','1000','5000']. If
   predict.py assumed numeric order, predictions would be silently
   mislabeled. Saving the generator's real mapping to disk is what makes
   inference correct later.

Run with (from the project root):
    python models/train.py

Output:
    models/currency_model.keras       (best model checkpoint)
    models/class_indices.json         (label <-> index mapping)
    models/training_history.json      (per-epoch metrics)
    models/training_curves.png        (accuracy/loss plots)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write plots to file without needing a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SPLIT_DIR = Path("dataset/split")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "currency_model.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"
HISTORY_PATH = MODELS_DIR / "training_history.json"
CURVES_PATH = MODELS_DIR / "training_curves.png"

IMG_SIZE = (224, 224)          # MobileNetV2's native input resolution
BATCH_SIZE = 32
MAX_EPOCHS = 30                # EarlyStopping will typically stop well before this
RANDOM_SEED = 42               # kept consistent with the split step for full reproducibility


# ----------------------------------------------------------------------
# Step 0: Reproducibility + environment info
# ----------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Seed TensorFlow/NumPy so weight init and shuffling are reproducible.

    Note: full bit-for-bit reproducibility across different machines/GPUs
    isn't guaranteed by TF (some GPU ops are non-deterministic by default),
    but this removes the biggest sources of run-to-run variance.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


def print_environment_info() -> None:
    print("-" * 70)
    print("ENVIRONMENT")
    print("-" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")
    if not gpus:
        print("No GPU detected -- training will run on CPU (slower, but the small "
              "dataset and frozen backbone keep this manageable).")
    print("-" * 70)


# ----------------------------------------------------------------------
# Step 1: Verify the split dataset exists
# ----------------------------------------------------------------------
def verify_dataset_dirs() -> None:
    for path in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not path.exists():
            print(f"ERROR: '{path}' does not exist. Run utils/split_dataset.py first.")
            sys.exit(1)


# ----------------------------------------------------------------------
# Step 2: Build data generators
# ----------------------------------------------------------------------
def build_generators():
    """Create train/val/test generators.

    Train uses real-time augmentation; val/test use only the MobileNetV2
    preprocessing function (which scales pixel values to the [-1, 1] range
    the pretrained backbone expects) so their metrics reflect genuine,
    unmodified images.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,          # notes photographed at a slight angle
        width_shift_range=0.15,     # horizontal shift -- NOT a flip, see module docstring
        height_shift_range=0.15,
        zoom_range=0.15,            # varying camera distance
        brightness_range=[0.8, 1.2],  # varying lighting conditions
        fill_mode="nearest",        # how to fill pixels revealed by shifts/rotation
    )

    # Validation and test data: no augmentation, same preprocessing only.
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,   # keep order stable -- makes val metrics/debugging consistent
    )

    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ----------------------------------------------------------------------
# Step 3: Print dataset statistics
# ----------------------------------------------------------------------
def print_dataset_statistics(train_gen, val_gen, test_gen) -> None:
    """Print per-class image counts for each split, using the generators'
    own bookkeeping (.classes + .class_indices) as ground truth -- this
    guarantees the numbers printed match exactly what the model will train
    and evaluate on.
    """
    sep = "-" * 70
    print(f"\n{sep}")
    print("DATASET STATISTICS")
    print(sep)

    index_to_class = {v: k for k, v in train_gen.class_indices.items()}
    num_classes = len(index_to_class)

    print(f"{'Class':<12}{'Train':>10}{'Val':>10}{'Test':>10}")
    print(sep)

    for idx in sorted(index_to_class):
        class_name = index_to_class[idx]
        train_count = int(np.sum(train_gen.classes == idx))
        val_count = int(np.sum(val_gen.classes == idx))
        test_count = int(np.sum(test_gen.classes == idx))
        print(f"{class_name:<12}{train_count:>10}{val_count:>10}{test_count:>10}")

    print(sep)
    print(f"Total classes: {num_classes}")
    print(f"Total train images: {train_gen.samples}")
    print(f"Total val images:   {val_gen.samples}")
    print(f"Total test images:  {test_gen.samples}")
    print(sep)


# ----------------------------------------------------------------------
# Step 4: Build the model
# ----------------------------------------------------------------------
def build_model(num_classes: int) -> Model:
    """MobileNetV2 backbone (frozen) + a small custom classification head.

    Head design: GlobalAveragePooling2D collapses the backbone's spatial
    feature maps into a single feature vector per image (far fewer
    parameters than Flatten(), which matters a lot with a small dataset).
    A single Dense(128) + Dropout(0.3) layer gives the head just enough
    capacity to learn currency-specific decision boundaries without being
    large enough to overfit on ~2,500 training images.
    """
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # Freeze the backbone: its ImageNet-learned features (edges, colors,
    # textures) are reused as-is. Only the head below gets trained.
    # Future improvement (V2): unfreeze the last N backbone layers and
    # fine-tune with a very low learning rate once the head has converged.
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ----------------------------------------------------------------------
# Step 5: Callbacks
# ----------------------------------------------------------------------
def build_callbacks():
    """
    - EarlyStopping: stops training once val_loss stops improving, and
      restores the best-performing weights seen during training (not
      necessarily the weights from the final epoch).
    - ModelCheckpoint: independently saves the single best model to disk
      by val_accuracy, so we have a persisted artifact even if training
      is interrupted.
    - ReduceLROnPlateau: shrinks the learning rate when val_loss plateaus,
      letting the model take smaller, more precise steps as it converges
      instead of overshooting a good minimum.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, checkpoint, reduce_lr]


# ----------------------------------------------------------------------
# Step 6: Save class indices (label <-> index mapping)
# ----------------------------------------------------------------------
def save_class_indices(train_gen) -> None:
    """Persist the exact class-name -> index mapping Keras assigned.

    See module docstring point 4: this order is alphabetical-as-string,
    not numeric, so predict.py MUST load this file rather than assume any
    particular order.
    """
    class_indices = train_gen.class_indices  # e.g. {'10': 0, '100': 1, ...}
    index_to_class = {v: k for k, v in class_indices.items()}

    payload = {
        "class_to_index": class_indices,
        "index_to_class": index_to_class,
    }

    # Note for predict.py: JSON object keys are always strings. That means
    # "index_to_class" will be loaded back as {"0": "10", "1": "100", ...}
    # even though these started as Python ints -- predict.py must convert
    # the model's predicted index to a string before looking it up, e.g.
    # index_to_class[str(predicted_index)].
    CLASS_INDICES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved class indices to: {CLASS_INDICES_PATH.resolve()}")


# ----------------------------------------------------------------------
# Step 7: Save training history
# ----------------------------------------------------------------------
def save_history(history) -> None:
    """Save per-epoch metrics as JSON for later inspection/plotting without
    needing to retrain.
    """
    # Cast numpy float32 values to plain Python floats -- json can't
    # serialize numpy scalar types directly.
    history_dict = {
        metric: [float(v) for v in values]
        for metric, values in history.history.items()
    }
    HISTORY_PATH.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    print(f"Saved training history to: {HISTORY_PATH.resolve()}")


# ----------------------------------------------------------------------
# Step 8: Plot accuracy/loss curves
# ----------------------------------------------------------------------
def plot_training_curves(history) -> None:
    h = history.history
    epochs_range = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs_range, h["accuracy"], label="Train Accuracy")
    axes[0].plot(epochs_range, h["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, h["loss"], label="Train Loss")
    axes[1].plot(epochs_range, h["val_loss"], label="Val Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVES_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved training curves to: {CURVES_PATH.resolve()}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    set_seeds(RANDOM_SEED)
    print_environment_info()
    verify_dataset_dirs()

    train_gen, val_gen, test_gen = build_generators()
    print_dataset_statistics(train_gen, val_gen, test_gen)

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes)

    print("\n" + "-" * 70)
    print("MODEL SUMMARY")
    print("-" * 70)
    model.summary()

    callbacks = build_callbacks()

    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

    # Persist everything the rest of the project (predict.py, app.py,
    # README screenshots) will need.
    save_class_indices(train_gen)
    save_history(history)
    plot_training_curves(history)

    # Quick sanity check on the held-out test set. This is not a full
    # evaluation report (confusion matrix, per-class precision/recall) --
    # that belongs in a separate evaluation step once we're ready to look
    # at model quality in depth. Here it's just a fast, honest check that
    # the saved model generalizes reasonably before moving on.
    print("\n" + "-" * 70)
    print("TEST SET SANITY CHECK")
    print("-" * 70)
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("-" * 70)

    print(f"\nBest model saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()