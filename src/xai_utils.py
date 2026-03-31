import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def explain_with_shap(model, X_train, X_sample, feature_names):
    """
    Explains predictions using SHAP GradientExplainer (faster for DL models).
    """
    # Use a subset of training data as background
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample)
    
    return explainer, shap_values

def get_integrated_gradients(model, baseline, target, m_steps=50):
    """
    Computes Integrated Gradients for a target sample.
    """
    # 1. Interpolate inputs between baseline and target
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    interpolated_inputs = baseline + alphas[:, tf.newaxis] * (target - baseline)
    
    # 2. Get gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)
    
    grads = tape.gradient(predictions, interpolated_inputs)
    
    # 3. Approximate the integral using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated_grads = (target - baseline) * tf.reduce_mean(avg_grads, axis=0)
    
    return integrated_grads.numpy()

def plot_feature_importance(importances, feature_names, title="Feature Importance"):
    """Plots a horizontal bar chart of feature importances."""
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"results/{title.lower().replace(' ', '_')}.png")
    plt.close()
