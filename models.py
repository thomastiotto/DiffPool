import tensorflow as tf
from layers_nocheb import *
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU


class GCNMaxPool(tf.keras.Model):

    def __init__(self, num_classes):
        super(GCNMaxPool, self).__init__()

        self.conv_1 = GCN(features=4, dropout=0.5)
        self.pool = SimplePool(mode="max")
        self.classifier = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1((filtre, X))
        x = self.pool((x, node_indicator))
        x = self.classifier(x[1])

        return x


class GCNMeanPool(tf.keras.Model):

    def __init__(self, num_classes):
        super(GCNMeanPool, self).__init__()

        self.conv_1 = GCN(features=4, dropout=0.5)
        self.pool = SimplePool(mode="mean")
        self.classifier = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1((filtre, X))
        x = self.pool((x, node_indicator))
        x = self.classifier(x[1])

        return x


class GCNDiffPool(tf.keras.Model):

    def __init__(self, num_classes, batch_size):
        super(GCNDiffPool, self).__init__()

        self.conv_1 = GCN(features=4, dropout=0.5)
        self.pool = DiffPool(max_clusters=2)
        self.classifier = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1((filtre, X))
        x = self.pool(x, node_indicator)
        x = self.classifier(x[1])

        return x