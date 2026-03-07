"""
PlantCare AI — Data Preprocessing Utilities
=============================================
Handles image preprocessing for both training pipelines
and single-image inference.
"""

import os
import sys
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class DataPreprocessor:
    def __init__(self):
        self.config = Config()

    def create_data_generators(self):
        """
        Create ImageDataGenerators for training and validation.
        Note: The New Plant Diseases Dataset is already augmented,
        so only minimal extra augmentation is applied here.

        Returns:
            tuple: (train_generator, validation_generator)
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

        train_generator = train_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = val_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=self.config.IMAGE_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        return train_generator, validation_generator

    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for model inference.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.config.IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def preprocess_pil_image(self, pil_image):
        """
        Preprocess a PIL Image object for model inference.

        Args:
            pil_image (PIL.Image): RGB PIL Image.

        Returns:
            np.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
        """
        img = pil_image.convert('RGB').resize(self.config.IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
