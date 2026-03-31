from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np

def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed forward
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(embed_dim)(ff_output)
    ff_output = Dropout(rate)(ff_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ff_output)

def train_transformer_model(X_train, X_test, y_train, y_test, params=None, verbose=0):
    """
    Trains a Transformer Encoder model.
    """
    default_params = {
        'embed_dim': 16,
        'num_heads': 2,
        'ff_dim': 32,
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 128
    }
    if params:
        default_params.update(params)
    params = default_params

    print(f"\nTraining Transformer Encoder Model with params: {params}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    inputs = Input(shape=(input_dim,))
    
    # Initial projection to embed_dim
    x = Dense(params['embed_dim'])(inputs)
    
    # Reshape for Attention
    x = Reshape((1, params['embed_dim']))(x)
    
    # Transformer Blocks
    x = transformer_block(x, params['embed_dim'], params['num_heads'], params['ff_dim'], params['dropout_rate'])
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(params['dropout_rate'])(x)
    x = Dense(32, activation='relu')(x)
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
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
        ModelCheckpoint('results/best_transformer_model.keras', save_best_only=True)
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
    print(f"Transformer DL Accuracy: {acc:.4f}")

    return acc, model, history
