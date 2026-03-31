from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv1D, GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np

def train_cnn_model(X_train, X_test, y_train, y_test, params=None, verbose=0):
    """
    Trains a 1D-CNN model for feature pattern extraction.
    """
    default_params = {
        'filters': [32, 64],
        'kernel_size': 3,
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 128
    }
    if params:
        default_params.update(params)
    params = default_params

    print(f"\nTraining 1D-CNN Model with params: {params}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    inputs = Input(shape=(input_dim,))
    
    # Reshape for Conv1D: (batch, steps, features)
    x = Reshape((input_dim, 1))(inputs)
    
    for f in params['filters']:
        x = Conv1D(f, params['kernel_size'], padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout_rate'])(x)
        
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
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
        ModelCheckpoint('results/best_cnn_model.keras', save_best_only=True)
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
    print(f"CNN DL Accuracy: {acc:.4f}")

    return acc, model, history
