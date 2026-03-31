import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully: {df.shape}")
    return df

def inspect_dataset(df):
    print("\nDataset Info:")
    print(df.info())
    print("\nClass distribution:")
    print(df['error_type'].value_counts())

def feature_engineering(df):
    """Adds advanced features for quantum error detection."""
    df = df.copy()
    
    # 1. Measurement probabilities
    total_shots = df['measured_0'] + df['measured_1'] + 1e-6
    df['prob_0'] = df['measured_0'] / total_shots
    df['prob_1'] = df['measured_1'] / total_shots
    
    # 2. Gate & Depth density
    df['gate_density'] = df['gate_count'] / (df['circuit_depth'] + 1e-6)
    
    # 3. Noise scaling
    df['log_noise'] = np.log1p(df['noise_strength'])
    df['noise_depth_product'] = df['noise_strength'] * df['circuit_depth']
    
    # 6. Interaction Terms
    df['gate_noise_interaction'] = df['gate_count'] * df['noise_strength']
    df['depth_gate_ratio'] = df['circuit_depth'] / (df['gate_count'] + 1e-6)
    df['noise_per_depth'] = df['noise_strength'] / (df['circuit_depth'] + 1e-6)
    
    # 7. Non-linear terms
    df['gate_count_sq'] = df['gate_count'] ** 2
    df['circuit_depth_sq'] = df['circuit_depth'] ** 2
    
    # 8. Combined metrics
    df['quantum_complexity'] = df['gate_count'] * df['circuit_depth'] * df['noise_strength']
    
    # 9. Physics Expectations (Crucial for class separation)
    # Theoretical probability of at least one gate error: 1 - (1-p)^n
    df['gate_error_prob'] = 1 - (1 - df['noise_strength'])**df['gate_count']
    df['readout_error_prob'] = df['noise_strength']
    
    # Likelihood Ratio: Gate vs Readout
    df['error_source_ratio'] = df['gate_error_prob'] / (df['readout_error_prob'] + 1e-6)
    
    # 10. Theoretical Likelihoods (One per class)
    # These match the physics of the generator
    s = df['noise_strength']
    n = df['gate_count']
    
    # Prob of flip for Readout: just s
    df['lik_readout'] = s
    # Prob of flip for Bit Flip (on 1 qubit): 0.5 * (1 - (1-2s)^n)
    df['lik_bit_flip'] = 0.5 * (1 - (1 - 2*s)**n)
    # Prob of flip for Phase Flip: (Z gate + X prep)
    df['lik_phase_flip'] = 0.5 * (1 - (1 - 2*s)**n) * 0.5 # Phase flips only affect X-basis prep
    # Depolarizing: (3/4) * (1 - (1 - 4s/3)^n)
    df['lik_depolarizing'] = 0.75 * (1 - (1 - 4*s/3)**n)
    
    # Delta from observed
    df['delta_readout'] = np.abs(df['error_rate'] - df['lik_readout'])
    df['delta_bit_flip'] = np.abs(df['error_rate'] - df['lik_bit_flip'])
    
    return df

def preprocess_data(df, use_robust=True):
    # Apply feature engineering
    df_feat = feature_engineering(df)
    
    # Drop target and original measurement counts
    X = df_feat.drop(columns=['error_type', 'measured_0', 'measured_1'])
    y = df_feat['error_type']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save artifacts
    os.makedirs('results', exist_ok=True)
    joblib.dump(scaler, 'results/scaler.pkl')
    joblib.dump(encoder, 'results/encoder.pkl')
    
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, encoder, feature_names