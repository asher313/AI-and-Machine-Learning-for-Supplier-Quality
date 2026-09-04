# Chapter 13 — 13.5 TensorFlow and Keras: The Map
inputs = keras.Input(shape=(20,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
