import numpy as np
import pandas as pd
import joblib
import os
import time

# FORCE CPU ONLY to avoid hangs on some Mac environments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Disable Metal/MPS plugin if it exists, as it can cause freezes
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

from src.preprocessing import load_data, preprocess_data
from src.deep_learning import train_deep_learning_model, train_mlp_model
from src.attention_model import train_attention_model
from src.cnn_model import train_cnn_model
from src.transformer_model import train_transformer_model

def check_data(X, name):
    print(f"  Validating {name}: shape={X.shape}, NaNs={np.isnan(X).any()}, Infs={np.isinf(X).any()}")

def balanced_train():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("🚀 Starting Balanced Training (Diagnostic Version)...")
    path = "data/quantum_multiclass_dataset.csv"
    df = load_data(path)
    
    # Use 2000 samples
    print(f"Subsampling 2000 samples...")
    df_small = df.sample(n=min(len(df), 2000), random_state=42)
    
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df_small)
    joblib.dump(feature_names, 'results/feature_names.pkl')

    check_data(X_train, "X_train")
    check_data(X_test, "X_test")

    # Use verbose=1 so user can see progress
    v = 1
    fast_params = {'epochs': 3, 'batch_size': 128}

    print("\n1. Training MLP...")
    start = time.time()
    train_mlp_model(X_train, X_test, y_train, y_test, params={**fast_params, 'hidden_units': [64, 32]}, verbose=v)
    print(f"MLP Done in {time.time()-start:.2f}s")

    print("\n2. Training ResNet...")
    start = time.time()
    train_deep_learning_model(X_train, X_test, y_train, y_test, params={**fast_params, 'dense_units': 64}, verbose=v)
    print(f"ResNet Done in {time.time()-start:.2f}s")

    print("\n3. Training Attention...")
    start = time.time()
    train_attention_model(X_train, X_test, y_train, y_test, params={**fast_params, 'dense_units': 32}, verbose=v)
    print(f"Attention Done in {time.time()-start:.2f}s")

    print("\n4. Training CNN...")
    start = time.time()
    train_cnn_model(X_train, X_test, y_train, y_test, params={**fast_params, 'filters': [16, 32]}, verbose=v)
    print(f"CNN Done in {time.time()-start:.2f}s")

    print("\n5. Training Transformer...")
    start = time.time()
    train_transformer_model(X_train, X_test, y_train, y_test, params={**fast_params, 'embed_dim': 8}, verbose=v)
    print(f"Transformer Done in {time.time()-start:.2f}s")

    print("\n✅ Balanced training complete! Models are now trained.")

if __name__ == "__main__":
    balanced_train()
