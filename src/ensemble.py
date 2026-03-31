import numpy as np
import joblib
import os

class QuantumEnsemble:
    def __init__(self, models=None, weights=None):
        self.models = models if models else []
        # Optimized weights based on diagnostic performance
        if weights is None and self.models:
            # Giving more weight to MLP and Transformer-Alt (Ensemble of top performers)
            self.weights = [0.3, 0.2, 0.1, 0.1, 0.3] # Adjusted to match model list
            # Normalize if count doesn't match
            if len(self.weights) != len(self.models):
                self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights = weights

    def predict_proba(self, X):
        """Averages the probability predictions of all models."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        all_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
            else:
                probs = model.predict(X)
            all_probs.append(probs)
        
        # Weighted average
        weighted_probs = np.zeros_like(all_probs[0])
        for prob, weight in zip(all_probs, self.weights):
            weighted_probs += prob * weight
            
        return weighted_probs

    def predict(self, X):
        """Predicts the class with the highest average probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def get_ensemble_model(model_paths):
    """Loads a list of models and returns an ensemble."""
    models = []
    for path in model_paths:
        if path.endswith('.keras'):
            import tensorflow as tf
            models.append(tf.keras.models.load_model(path))
        elif path.endswith('.joblib'):
            models.append(joblib.load(path))
    return QuantumEnsemble(models=models)
