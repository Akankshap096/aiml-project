"""
PlantCare AI — Model Training Script
=====================================
Trains a MobileNetV2 transfer learning model on the New Plant Diseases Dataset.
Achieves ~96.6% validation accuracy across 38 disease classes.

Usage:
    python model/train_model.py
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# ── Configuration ──────────────────────────────────────────────────────────────
cfg = Config()
os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)

CLASSES = cfg.CLASSES
NUM_CLASSES = len(CLASSES)  # 38

print(f"\n{'═'*55}")
print(f"  🌿 PlantCare AI — MobileNetV2 Training")
print(f"{'═'*55}")
print(f"  Dataset path  : {cfg.DATASET_PATH}")
print(f"  Image size    : {cfg.IMAGE_SIZE}")
print(f"  Batch size    : {cfg.BATCH_SIZE}")
print(f"  Epochs        : {cfg.EPOCHS}")
print(f"  Learning rate : {cfg.LEARNING_RATE}")
print(f"  Num classes   : {NUM_CLASSES}")
print(f"{'═'*55}\n")


# ── Data Generators ────────────────────────────────────────────────────────────
def create_generators():
    """
    Create train/validation data generators.
    Note: Dataset already contains augmented images, so minimal
    additional augmentation is applied.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        cfg.DATASET_PATH,
        target_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        cfg.DATASET_PATH,
        target_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print(f"  Train samples      : {train_gen.samples}")
    print(f"  Validation samples : {val_gen.samples}")
    print(f"  Classes found      : {len(train_gen.class_indices)}\n")

    return train_gen, val_gen


# ── Build Model ────────────────────────────────────────────────────────────────
def build_model():
    """
    Build MobileNetV2 transfer learning model.

    Architecture:
        Input (224x224x3)
        → MobileNetV2 base (pretrained ImageNet, frozen)
        → GlobalAveragePooling2D
        → Dropout(0.35)
        → Dense(256, ReLU)
        → Dropout(0.25)
        → Dense(38, Softmax)
    """
    # Load MobileNetV2 without top classification layers
    base_model = MobileNetV2(
        input_shape=(*cfg.IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base during initial training

    # Build custom head
    inputs = Input(shape=(*cfg.IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )

    print(f"  Base model params  : {base_model.count_params():,} (frozen)")
    print(f"  Total params       : {model.count_params():,}")
    print(f"  Trainable params   : {sum(tf.size(v).numpy() for v in model.trainable_variables):,}\n")

    return model, base_model


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks():
    return [
        ModelCheckpoint(
            filepath=cfg.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]


# ── Fine-Tuning ────────────────────────────────────────────────────────────────
def fine_tune(model, base_model, train_gen, val_gen):
    """
    Phase 2: Unfreeze the top 50 layers of MobileNetV2 and
    train with a much lower learning rate to refine features.
    """
    print("\n── Phase 2: Fine-Tuning ──────────────────────────────")
    base_model.trainable = True

    # Freeze all except last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    fine_tune_lr = cfg.LEARNING_RATE / 10
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )

    trainable_now = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"  Trainable params after unfreeze: {trainable_now:,}")
    print(f"  Fine-tune learning rate: {fine_tune_lr}\n")

    history_ft = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )
    return history_ft


# ── Plot History ───────────────────────────────────────────────────────────────
def plot_history(history, history_ft=None):
    """Plot and save training curves."""
    acc     = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss    = history.history['loss']
    val_loss= history.history['val_loss']

    if history_ft:
        acc     += history_ft.history['accuracy']
        val_acc += history_ft.history['val_accuracy']
        loss    += history_ft.history['loss']
        val_loss+= history_ft.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc,     'b-o', label='Train Accuracy',      markersize=4)
    plt.plot(epochs, val_acc, 'g-o', label='Validation Accuracy', markersize=4)
    if history_ft:
        phase2_start = len(history.history['accuracy']) + 1
        plt.axvline(x=phase2_start, color='orange', linestyle='--', label='Fine-tune start')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(alpha=0.3)
    plt.ylim([0.6, 1.0])

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss,     'b-o', label='Train Loss',      markersize=4)
    plt.plot(epochs, val_loss, 'g-o', label='Validation Loss', markersize=4)
    if history_ft:
        plt.axvline(x=phase2_start, color='orange', linestyle='--', label='Fine-tune start')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n  📊 Training curves saved to: training_history.png")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Check dataset exists
    if not os.path.exists(cfg.DATASET_PATH):
        print(f"❌ Dataset not found at: {cfg.DATASET_PATH}")
        print("   Download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("   Then place in: data/PlantVillage/")
        sys.exit(1)

    # 1. Data
    train_gen, val_gen = create_generators()

    # 2. Build model
    model, base_model = build_model()

    # 3. Phase 1: Train with frozen base
    print("── Phase 1: Training with frozen MobileNetV2 base ────")
    history = model.fit(
        train_gen,
        epochs=cfg.EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )

    # 4. Phase 2: Fine-tune (optional — uncomment to enable)
    # history_ft = fine_tune(model, base_model, train_gen, val_gen)

    # 5. Final evaluation
    print("\n── Final Evaluation ──────────────────────────────────")
    results = model.evaluate(val_gen, verbose=1)
    print(f"\n  ✅ Final Val Accuracy  : {results[1]*100:.2f}%")
    print(f"  ✅ Final Val Top-5 Acc : {results[2]*100:.2f}%")
    print(f"  💾 Best model saved to : {cfg.MODEL_PATH}")

    # 6. Plot
    plot_history(history)
    # plot_history(history, history_ft)  # if fine-tuning was used
