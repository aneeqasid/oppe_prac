import joblib
import pandas as pd
import numpy as np
import sys
import os

def load_model(model_path):
    """Load model from path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def predict_single(model, features):
    """Predict single sample"""
    return model.predict([features])[0]

if __name__ == "__main__":
    # Load model (default to v0+v1 model)
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model_v01.pkl"
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Test samples from all 3 species
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [6.7, 3.0, 5.2, 2.3],  # Virginica
    ]
    
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    
    print("\n=== Predictions ===")
    for i, sample in enumerate(test_samples):
        pred = predict_single(model, sample)
        print(f"Sample {i+1}: {sample} → Predicted: {species_map[pred]}")
    
    print("\n✅ Inference completed successfully!")
