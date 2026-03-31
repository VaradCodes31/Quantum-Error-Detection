import pandas as pd
import joblib
import numpy as np
import os
from src.preprocessing import preprocess_data

def audit_unified():
    print("🕵️ Unified Ensemble Audit Starting...")
    
    df = pd.read_csv('data/quantum_multiclass_dataset.csv')
    X_train_scaled, X_test_scaled, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    label_map = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    
    ensemble_path = 'results/best_ensemble_model.joblib'
    if not os.path.exists(ensemble_path):
        print("❌ Ensemble artifact missing")
        return
        
    ensemble = joblib.load(ensemble_path)
    
    print(f"\n--- Unified Ensemble Audit ---")
    correct = 0
    total = 0
    for idx in range(len(encoder.classes_)):
        class_name = label_map[idx]
        pos_indices = np.where(y_test == idx)[0][:5] # Test 5 samples each
        
        for pos in pos_indices:
            x_sample = X_test_scaled[pos].reshape(1, -1)
            pred_idx = ensemble.predict(x_sample)[0]
            pred_label = encoder.inverse_transform([pred_idx])[0]
            
            status = "✅" if pred_idx == idx else "❌"
            print(f"Actual: {class_name:12} | Predicted: {pred_label:12} {status}")
            if pred_idx == idx: correct += 1
            total += 1
    
    print(f"\nAudit Accuracy over 20 samples: {correct/total:.2%}")

if __name__ == "__main__":
    audit_unified()
