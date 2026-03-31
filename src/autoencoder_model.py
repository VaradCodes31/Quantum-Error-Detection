from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam


def train_autoencoder_model(X_train, X_test, y_train, y_test):

    print("\nTraining Autoencoder + Classifier")

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    num_classes = len(set(y_train_enc))

    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)

    input_dim = X_train.shape[1]

    # ---------- Encoder ----------
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    latent = Dense(8, activation='relu')(encoded)

    # ---------- Decoder ----------
    decoded = Dense(16, activation='relu')(latent)
    decoded = Dense(32, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    # Autoencoder model
    autoencoder = Model(input_layer, output_layer)

    autoencoder.compile(
        optimizer=Adam(),
        loss='mse'
    )

    # Train autoencoder
    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # ---------- Encoder Model ----------
    encoder = Model(input_layer, latent)

    # Transform data
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # ---------- Classifier ----------
    classifier = Model(
        inputs=encoder.input,
        outputs=Dense(num_classes, activation='softmax')(latent)
    )

    classifier.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train classifier
    classifier.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    loss, acc = classifier.evaluate(X_test, y_test_cat, verbose=0)

    print(f"Autoencoder Model Accuracy: {acc}")

    return acc, classifier