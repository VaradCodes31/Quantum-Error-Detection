import pandas as pd
import joblib
import os
import numpy as np
from src.preprocessing import feature_engineering

def check_collapse():
    print("🔍 Checking for Class Collapse...")
    df = pd.read_csv('data/quantum_multiclass_dataset.csv')
    print(f"Dataset Shape: {df.shape}")
    print("\nTraining Label Distribution:")
    print(df['error_type'].value_counts())
    
    # Load model (MLP is often the most sensitive)
    mlp_path = 'results/best_mlp_model.joblib'
    if os.path.exists(mlp_path):
        model = joblib.load(mlp_path)
        encoder = joblib.load('results/encoder.pkl')
        scaler = joblib.load('results/scaler.pkl')
        
        # Test with varied inputs
        test_inputs = [
            {'prepared_state':0, 'measured_0':900, 'measured_1':124, 'error_rate':0.1, 'noise_strength':0.01, 'gate_count':1, 'circuit_depth':3}, # Likely Readout
            {'prepared_state':0, 'measured_0':500, 'measured_1':524, 'error_rate':0.5, 'noise_strength':0.1, 'gate_count':20, 'circuit_depth':21}, # Likely Bit Flip
            {'prepared_state':1, 'measured_0':100, 'measured_1':924, 'error_rate':0.1, 'noise_strength':0.05, 'gate_count':5, 'circuit_depth':6}, # Likely Phase Flip
        ]
        
        df_test = pd.DataFrame(test_inputs)
        df_feat = feature_engineering(df_test)
        
        # We need the feature names used during training
        # I'll just reload from the diagnostics script I used before or similar
        from src.preprocessing import preprocess_data
        _, _, _, _, _, feature_names = preprocess_data(df)
        
        X = df_feat[feature_names]
        X_scaled = scaler.transform(X)
        
        probs = model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)
        labels = encoder.inverse_transform(preds)
        
        print("\nTest Predictions:")
        for i, label in enumerate(labels):
            print(f"Input {i}: {label} ({probs[i][preds[i]]:.2%})")

if __name__ == "__main__":
    check_collapse()
