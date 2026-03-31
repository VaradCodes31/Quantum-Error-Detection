import joblib
import pandas as pd
import numpy as np
import os
from src.preprocessing import load_data, preprocess_data
from src.visualization import plot_multiclass_roc, plot_multiclass_pr, plot_tsne_clusters

def run_analytics():
    print("🛡️ Starting Advanced Analytics Generation...")
    
    # 1. Load Data
    path = "data/quantum_multiclass_dataset.csv"
    if not os.path.exists(path):
        print(f"❌ Error: Dataset not found at {path}")
        return

    df = load_data(path)
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    # 2. Load Model
    model_path = 'results/best_ensemble_model.joblib'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
        
    print(f"✅ Loading model: {model_path}")
    ensemble = joblib.load(model_path)
    
    # 3. Generate Probabilities for ROC/PR
    print("📊 Calculating prediction probabilities...")
    y_probs = ensemble.predict_proba(X_test)
    classes = encoder.classes_
    
    # 4. Generate Plots
    print("🎨 Generating ROC Curve...")
    plot_multiclass_roc(y_test, y_probs, classes)
    
    print("🎨 Generating Precision-Recall Curve...")
    plot_multiclass_pr(y_test, y_probs, classes)
    
    print("🎨 Generating t-SNE Cluster Plot...")
    # Use the scaled test data for t-SNE
    plot_tsne_clusters(X_test, y_test, classes, n_samples=2000)
    
    print("\n✨ ANALYTICS COMPLETE! All plots saved in 'results/'")

if __name__ == "__main__":
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    run_analytics()
