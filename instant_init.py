import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Conv1D, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from src.preprocessing import load_data, preprocess_data
import os

def create_and_save_placeholder_models():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("Loading metadata...")
    df = load_data("data/quantum_multiclass_dataset.csv")
    # Just need the structure/shapes
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df.head(100))
    
    input_dim = X_train.shape[1]
    num_classes = len(encoder.classes_)

    # 1. MLP
    print("Generating MLP...")
    inputs = Input(shape=(input_dim,))
    outputs = Dense(num_classes, activation='softmax')(inputs)
    Model(inputs, outputs).save('results/best_mlp_model.keras')

    # 2. ResNet (Simplified)
    print("Generating ResNet...")
    inputs = Input(shape=(input_dim,))
    outputs = Dense(num_classes, activation='softmax')(inputs)
    Model(inputs, outputs).save('results/best_resnet_model.keras')

    # 3. Attention (Simplified)
    print("Generating Attention...")
    inputs = Input(shape=(input_dim,))
    outputs = Dense(num_classes, activation='softmax')(inputs)
    Model(inputs, outputs).save('results/best_attention_model.keras')

    # 4. CNN
    print("Generating CNN...")
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(16, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    Model(inputs, outputs).save('results/best_cnn_model.keras')

    # 5. Transformer
    print("Generating Transformer...")
    inputs = Input(shape=(input_dim,))
    x = Dense(16)(inputs)
    x = Reshape((1, 16))(x)
    attn = MultiHeadAttention(num_heads=1, key_dim=16)(x, x)
    x = LayerNormalization()(x + attn)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    Model(inputs, outputs).save('results/best_transformer_model.keras')

    joblib.dump(feature_names, 'results/feature_names.pkl')
    print("\n✅ All 5 model placeholders created in 'results/' folder!")
    print("You can now run 'streamlit run app.py' and it will work immediately.")

if __name__ == "__main__":
    create_and_save_placeholder_models()
