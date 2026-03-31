from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Multiply, BatchNormalization, Dropout, LayerNormalization, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

def gated_residual_network(x, units, dropout_rate=0.1):
    shortcut = x
    
    # Linear layer
    a = Dense(units, activation='relu')(x)
    a = Dense(units)(a)
    a = Dropout(dropout_rate)(a)
    
    # Gating mechanism
    g = Dense(units, activation='sigmoid')(x)
    x = Multiply()([a, g])
    
    # Residual connection
    if shortcut.shape[-1] == units:
        x = Add()([x, shortcut])
    
    x = LayerNormalization()(x)
    return x


def train_attention_model(X_train, X_test, y_train, y_test, params=None, verbose=0):
    """
    Trains a Gated Attention model with configurable parameters.
    """
    default_params = {
        'dense_units': 128,
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 64
    }
    if params:
        default_params.update(params)
    params = default_params

    print(f"\nTraining Gated Attention Neural Network with params: {params}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    inputs = Input(shape=(input_dim,))

    # Feature Attention (Gating)
    feature_gate = Dense(input_dim, activation='sigmoid')(inputs)
    x = Multiply()([inputs, feature_gate])
    
    # Dense projection
    x = Dense(params['dense_units'])(x)
    x = LayerNormalization()(x)
    
    # Gated Residual Blocks
    x = gated_residual_network(x, params['dense_units'], dropout_rate=params['dropout_rate'])
    x = gated_residual_network(x, params['dense_units'] // 2, dropout_rate=params['dropout_rate'])
    
    x = Dense(params['dense_units'] // 4, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint('results/best_attention_model.keras', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=callbacks,
        verbose=verbose
    )

    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Gated Attention Accuracy: {acc:.4f}")

    return acc, model