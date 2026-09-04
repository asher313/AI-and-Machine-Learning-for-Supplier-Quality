# Chapter 13 — 13.5 TensorFlow and Keras: The Map
class SupplierRiskModel(keras.Model):
    def __init__(self, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim)
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(hidden_dim)
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(1)

    def call(self, x, training=False):
        x = self.bn1(self.dense1(x), training=training)
        x = self.dropout(tf.nn.relu(x), training=training)
        x = self.bn2(self.dense2(x), training=training)
        x = self.dropout(tf.nn.relu(x), training=training)
        return self.classifier(x)
