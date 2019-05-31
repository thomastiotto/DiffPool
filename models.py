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

        x = self.conv_1(inputs)
        x = self.pool(x)
        x = Dense(512, activation='relu')(x[1])
        x = self.classifier(x)

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

        x = self.conv_1(inputs)
        x = self.pool(x)
        x = Dense(512, activation='relu')(x[1])
        x = self.classifier(x)

        return x


class GCNDiffPool(tf.keras.Model):

    def __init__(self, num_classes, avg_num_nodes):
        import math

        super(GCNDiffPool, self).__init__()

        self.conv_1 = GCN(features=math.ceil(avg_num_nodes/10), dropout=0.5)
        self.pool = DiffPool(max_clusters=1)
        self.classifier = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1(inputs)
        x = self.pool(x)
        x = Dense(512, activation='relu')(x[1])
        x = self.classifier(x)

        return x