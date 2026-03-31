from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np


def res_block(x, units, dropout_rate=0.2):
    shortcut = x
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(units)(x)
    x = BatchNormalization()(x)

    # If shortcut shape matches, add it
    if shortcut.shape[-1] == units:
        x = Add()([x, shortcut])

    x = Activation('relu')(x)
    return x


def train_deep_learning_model(X_train, X_test, y_train, y_test, params=None, verbose=0):
    """
    Trains a ResNet-style Deep Learning model with configurable parameters.
    """
    default_params = {
        'dense_units': 256,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 150,
        'batch_size': 64
    }
    if params:
        default_params.update(params)
    params = default_params

    print(f"\nTraining ResNet-style Deep Learning Model with params: {params}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    inputs = Input(shape=(input_dim,))

    # Initial projection
    x = Dense(params['dense_units'])(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual Blocks
    x = res_block(x, params['dense_units'], dropout_rate=params['dropout_rate'])
    x = res_block(x, params['dense_units'], dropout_rate=params['dropout_rate'])
    
    # Bottleneck
    bottleneck_units = params['dense_units'] // 2
    x = Dense(bottleneck_units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = res_block(x, bottleneck_units, dropout_rate=params['dropout_rate']/2)
    
    x = Dense(bottleneck_units // 2, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint('results/best_resnet_model.keras', save_best_only=True)
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
    print(f"ResNet DL Accuracy: {acc:.4f}")

    return acc, model, history


def train_mlp_model(X_train, X_test, y_train, y_test, params=None, verbose=0):
    """
    Trains a simple Multi-Layer Perceptron (MLP) model.
    """
    default_params = {
        'hidden_units': [256, 128],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 20,
        'batch_size': 128
    }
    if params:
        default_params.update(params)
    params = default_params

    print(f"\nTraining MLP Deep Learning Model with params: {params}")

    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    inputs = Input(shape=(input_dim,))
    x = inputs
    
    for units in params['hidden_units']:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(params['dropout_rate'])(x)
        
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
        ModelCheckpoint('results/best_mlp_model.keras', save_best_only=True)
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
    print(f"MLP DL Accuracy: {acc:.4f}")

    return acc, model, history