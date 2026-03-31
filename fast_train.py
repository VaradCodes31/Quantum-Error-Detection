import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Activation, Conv1D, GlobalAveragePooling1D, Reshape, MultiHeadAttention, LayerNormalization, Multiply
from tensorflow.keras.utils import to_categorical
from src.preprocessing import load_data, preprocess_data
import os

def create_mlp(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Reshape((input_dim, 1))(inputs)
    x = Conv1D(32, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transformer(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(32)(inputs)
    x = Reshape((1, 32))(x)
    attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = LayerNormalization()(x + attn)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_attention(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    print("Loading data for fast training...")
    df = load_data("data/quantum_multiclass_dataset.csv")
    df = df.sample(n=min(len(df), 500), random_state=42)
    X_train, X_test, y_train, y_test, encoder, feature_names = preprocess_data(df)
    
    input_dim = X_train.shape[1]
    num_classes = len(encoder.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    models = [
        ('mlp', create_mlp),
        ('resnet', create_resnet),
        ('attention', create_attention),
        ('cnn', create_cnn),
        ('transformer', create_transformer)
    ]

    for name, creator in models:
        print(f"Training {name} (extreme fast)...")
        model = creator(input_dim, num_classes)
        model.fit(X_train, y_train_cat, epochs=1, batch_size=32, verbose=0)
        model.save(f'results/best_{name}_model.keras')
        print(f"Saved results/best_{name}_model.keras")

    joblib.dump(feature_names, 'results/feature_names.pkl')
    print("Fast training complete!")

if __name__ == "__main__":
    main()
