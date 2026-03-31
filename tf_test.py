import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import time

print("TF Version:", tf.__version__)
print("Testing simple matrix multiplication...")
start = time.time()
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])
c = tf.matmul(a, b)
print("Result shape:", c.shape)
print(f"Time taken: {time.time() - start:.4f} seconds")

print("\nTesting simple model fit (100 samples)...")
import numpy as np
X = np.random.random((100, 10))
y = np.random.randint(0, 2, 100)
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy')
start = time.time()
model.fit(X, y, epochs=1, verbose=0)
print(f"Model fit time: {time.time() - start:.4f} seconds")
print("✅ Test complete!")
