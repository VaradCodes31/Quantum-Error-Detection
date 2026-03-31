import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

def run_diagnostics():
    print("🔍 Running Ensemble Diagnostics...")
    
    # Load data
    df = pd.read_csv('data/quantum_multiclass_dataset.csv')
    from src.preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    print(f"Label Mapping: {dict(zip(range(len(encoder.classes_)), encoder.classes_))}")
    
    model_files = [
        ('MLP', 'results/best_mlp_model.joblib'),
        ('ResNet-Alt', 'results/best_resnet_alt_model.joblib'),
        ('Attention-Alt', 'results/best_attention_alt_model.joblib'),
        ('CNN-Alt', 'results/best_cnn_alt_model.joblib'),
        ('Transformer-Alt', 'results/best_transformer_alt_model.joblib')
    ]
    
    for name, path in model_files:
        if os.path.exists(path):
            print(f"\n--- {name} Performance ---")
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            print(classification_report(y_test, y_pred, target_names=encoder.classes_))
        else:
            print(f"Skipping {name} (not found)")

if __name__ == "__main__":
    run_diagnostics()
