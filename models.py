import tensorflow as tf
from layers_nocheb import *
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU

class GCNMaxPool(tf.keras.Model):

    def __init__(self, num_classes, batch_size):
        super(GCNMaxPool, self).__init__()

        self.conv_1 = GCN(features=32, dropout=0.5)
        self.pool = SimplePool(mode="max", batch_size=batch_size)
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1((filtre, X))
        x = self.pool((x, node_indicator))
        x = self.classifier(x[1])

        return x