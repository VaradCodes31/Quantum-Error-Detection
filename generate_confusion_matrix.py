import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.preprocessing import load_data, preprocess_data

def generate_confusion_matrix():
    print("📊 Generating Confusion Matrix...")
    
    # 1. Load Data
    path = "data/quantum_multiclass_dataset.csv"
    if not os.path.exists(path):
        print(f"❌ Error: Dataset not found at {path}")
        return

    df = load_data(path)
    
    # 2. Preprocess (Replicating exact training split)
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    # 3. Load Model
    model_path = 'results/best_ensemble_model.joblib'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Run final_train.py first.")
        return
        
    ensemble = joblib.load(model_path)
    
    # 4. Generate Predictions
    print("  > Predicting on Test Set...")
    y_pred = ensemble.predict(X_test)
    
    # 5. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalized
    
    # 6. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Quantum Error Classification: Confusion Matrix (%)', fontsize=14, pad=20)
    plt.ylabel('Actual Error Type', fontsize=12)
    plt.xlabel('Predicted Error Type', fontsize=12)
    plt.tight_layout()
    
    # Save result
    save_path = 'results/confusion_matrix.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ Confusion Matrix saved to {save_path}")
    
    # 7. Print Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

if __name__ == "__main__":
    generate_confusion_matrix()
