# Chapter 13 — 13.5 TensorFlow and Keras: The Map
optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
metric = keras.metrics.BinaryAccuracy()


@tf.function                    # compile the graph, faster
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables)
    )
    metric.update_state(y, logits)
    return loss


for epoch in range(epochs):
    for x, y in train_dataset:
        loss = train_step(x, y)
    print(f"epoch {epoch} acc {metric.result():.4f}")
    metric.reset_state()
