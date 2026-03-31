import numpy as np
import pandas as pd
import joblib
import os
import time
from src.preprocessing import load_data, preprocess_data
from src.classical_models import train_optimized_ensemble

def final_fix_train():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("🛡️ Starting Final Optimized Training...")
    path = "data/quantum_multiclass_dataset.csv"
    df = load_data(path)
    
    # Use all data! Scikit-learn is fast.
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    joblib.dump(feature_names, 'results/feature_names.pkl')
    
    # Train the new ensemble
    train_optimized_ensemble(X_train, X_test, y_train, y_test)
    
    print("\n✅ TRAINING COMPLETE! All models saved as .joblib in 'results/'")
    print("Now update your dashboard to load these models.")

if __name__ == "__main__":
    final_fix_train()
