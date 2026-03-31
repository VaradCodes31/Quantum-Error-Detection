import pandas as pd
import joblib
import numpy as np
import os
from src.preprocessing import feature_engineering, preprocess_data

def audit_ensemble():
    print("🕵️ Ensemble Audit Starting...")
    
    # 1. Load Data
    df = pd.read_csv('data/quantum_multiclass_dataset.csv')
    X_train_scaled, X_test_scaled, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    label_map = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    print(f"Index Mapping: {label_map}")

    model_info = [
        ('MLP', 'results/best_mlp_model.joblib'),
        ('ResNet-Alt', 'results/best_resnet_alt_model.joblib'),
        ('Attention-Alt', 'results/best_attention_alt_model.joblib'),
        ('Transformer-Alt', 'results/best_transformer_alt_model.joblib'),
        ('CNN-Alt', 'results/best_cnn_alt_model.joblib')
    ]

    for name, path in model_info:
        if not os.path.exists(path):
            print(f"❌ {name}: Missing artifact")
            continue
            
        model = joblib.load(path)
        
        # Test on 5 samples of each class
        print(f"\n--- Model Audit: {name} ---")
        for idx in range(len(encoder.classes_)):
            class_name = label_map[idx]
            # Find first occurrence of this class in y_test
            pos = np.where(y_test == idx)[0][0]
            
            x_sample = X_test_scaled[pos].reshape(1, -1)
            pred_idx = model.predict(x_sample)[0]
            pred_label = encoder.inverse_transform([pred_idx])[0]
            
            status = "✅" if pred_idx == idx else "❌"
            print(f"Actual: {class_name:12} | Predicted: {pred_label:12} {status}")

if __name__ == "__main__":
    audit_ensemble()
