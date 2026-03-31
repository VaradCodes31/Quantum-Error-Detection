import numpy as np
import pandas as pd
import joblib
import os
import time

# FORCE CPU ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from src.preprocessing import load_data, preprocess_data

def minimal_train():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("🚀 Starting STABILIZED Training (Float32 + No Progress Bar)...")
    path = "data/quantum_multiclass_dataset.csv"
    
    print("Step 1: Loading data...")
    df = load_data(path)
    df_small = df.sample(n=min(len(df), 500), random_state=42)
    
    print("Step 2: Preprocessing...")
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df_small)
    
    # CONVERT TO FLOAT32 EXPLICITLY
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    num_classes = len(encoder.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes).astype('float32')
    
    print(f"Step 3: Training models (SILENT MODE)...")
    models = ['mlp', 'resnet', 'attention', 'cnn', 'transformer']
    
    for name in models:
        print(f"  > Training {name}...", end=" ", flush=True)
        start = time.time()
        
        inputs = tf.keras.Input(shape=(X_train.shape[1],))
        if name == 'cnn':
            x = tf.keras.layers.Reshape((X_train.shape[1], 1))(inputs)
            x = tf.keras.layers.Conv1D(16, 3, activation='relu')(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif name == 'transformer':
            x = tf.keras.layers.Dense(16)(inputs)
            x = tf.keras.layers.Reshape((1, 16))(x)
            attn = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=16)(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            x = tf.keras.layers.Dense(32, activation='relu')(inputs)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # USE VERBOSE=0 (Completely Silent)
        model.fit(X_train, y_train_cat, epochs=5, batch_size=64, verbose=0)
        model.save(f'results/best_{name}_model.keras')
        print(f"Done ({time.time()-start:.2f}s)")

    joblib.dump(feature_names, 'results/feature_names.pkl')
    print("\n✅ TRAINING SUCCESSFUL! ALL MODELS SAVED.")
    print("You can now run 'streamlit run app.py'.")

if __name__ == "__main__":
    minimal_train()
