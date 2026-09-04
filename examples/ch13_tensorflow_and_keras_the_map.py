# Chapter 13 — 13.5 TensorFlow and Keras: The Map
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4
    ),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC()],
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "best.keras", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=2
        ),
    ],
)
predictions = model.predict(X_test)
