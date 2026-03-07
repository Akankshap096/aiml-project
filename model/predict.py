"""
PlantCare AI — Prediction / Inference Module
=============================================
Loads the trained MobileNetV2 model and runs inference
on leaf images to return top-3 disease predictions.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model.preprocess import DataPreprocessor


class PlantDiseasePredictor:
    def __init__(self):
        self.config = Config()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained Keras model from disk."""
        try:
            self.model = tf.keras.models.load_model(self.config.MODEL_PATH)
            print(f"✅ Model loaded: {self.config.MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Model not found at '{self.config.MODEL_PATH}'")
            print(f"   Train the model first: python model/train_model.py")
            self.model = None

    def predict(self, image_path):
        """
        Predict plant disease from an image file.

        Args:
            image_path (str): Path to the leaf image.

        Returns:
            list[dict] | None: Top-3 predictions with plant, disease,
                               confidence, severity, spread, treatment.
                               Returns None if model is not loaded.
        """
        if self.model is None:
            return None

        img = self.preprocessor.preprocess_single_image(image_path)
        preds = self.model.predict(img, verbose=0)[0]
        top3_idx = np.argsort(preds)[-3:][::-1]

        results = []
        for idx in top3_idx:
            class_name = self.config.CLASSES[idx]
            parts = class_name.split('___')
            plant   = parts[0].replace('_', ' ').replace(',', '').strip()
            disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
            confidence = float(round(preds[idx] * 100, 2))

            results.append({
                'plant':      plant,
                'disease':    disease,
                'confidence': confidence,
                'class_raw':  class_name,
                'is_healthy': 'healthy' in class_name.lower(),
            })

        return results

    def predict_top1(self, image_path):
        """Return only the single best prediction."""
        results = self.predict(image_path)
        return results[0] if results else None
