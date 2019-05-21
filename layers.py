import numpy as np
import tensorflow as tf
from tensorflow import keras


class GCN(keras.layers.Layer):

    def __init__(self, features, cheb=False, **kwargs):
        self.F_prime = features
        self.cheb = cheb

        self.w = []

        super(GCN, self).__init__(**kwargs)

    def build(self, input_shape):
        # input is a tuple of tensors (A, X)
        self.w.append(self.add_weight(name="W_0",
                                 shape=(input_shape[1][2], self.F_prime),
                                 initializer=tf.initializers.GlorotUniform(),
                                 trainable=True)
                            )
        if self.cheb:
            self.w.append(self.add_weight(name="W_1",
                                          shape=(input_shape[1][2], self.F_prime),
                                          initializer=tf.initializers.GlorotUniform(),
                                          trainable=True)
                          )

        super(GCN, self).build(input_shape)

    def call(self, x):
        # input is a tuple (A, X)
        filtres = x[0]
        X = x[1]

        # X = tf.cast(X, tf.float64)
        batch_size = X.shape[0]
        in_size = X.shape[1]
        in_weights = X.shape[2]
        out_weights = self.F_prime

        # tf.print(tf.sparse.to_dense(self.A), summarize=-1)

        # tf.print(X, summarize=-1)
        X = tf.reshape(X, [-1, in_weights])
        # tf.print(X, summarize=-1)

        output = []

        # do convolution
        for i in range(len(self.w)):
            hidden = tf.matmul(X, self.w[i])

            # self.filtres[i] = self.convert_sparse_matrix_to_sparse_tensor(self.filtres[i])
            hidden = tf.sparse.sparse_dense_matmul(filtres[i], hidden)

            # tf.print(hidden, summarize=-1)
            hidden = tf.reshape(hidden, [-1, in_size, out_weights])
            # tf.print(hidden, summarize=-1)

            output.append(hidden)
        # tf.print(tf.add_n(output), summarize=-1)
        return tf.tuple([filtres, tf.keras.activations.relu(tf.add_n(output))])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.F_prime)


class SimplePool(keras.layers.Layer):

    def __init__(self, batch_size, mode, **kwargs):
        assert mode == "max" or mode == "mean", "GCNPool must have 'max' or 'mean' as mode"

        self.mode = mode

        super(SimplePool, self).__init__(**kwargs)

    def build(self, input_shape):
        # input is a tuple of tensors (A, X)
        self.F_prime = input_shape[1][2]
        self.in_size = input_shape[1][1]
        self.batch_size = input_shape[1][0]

        super(SimplePool, self).build(input_shape)

    def call(self, x):
        # input is a tuple (A, X)
        filtres = x[0]
        X = x[1]

        segment_ids = np.array([], dtype=np.int32).reshape(0, self.in_size)

        for b in range(self.batch_size):
            index = np.repeat(b, self.in_size)
            segment_ids = np.concatenate((segment_ids, index), axis=None)

        # tf.print(x, summarize=-1)
        X = tf.reshape(X, [-1, self.F_prime])
        # tf.print(x, summarize=-1)

        if self.mode == "max":
            X = tf.math.segment_max(X, segment_ids)
        else:
            X = tf.math.segment_mean(X, segment_ids)

        # tf.print(x, summarize=-1)

        return tf.tuple([filtres, X])

    def compute_output_shape(self, input_shape):
        return (input_shape[0] / self.in_size, self.F_prime)


class DiffPool(keras.layers.Layer):

    def __init__(self, max_clusters, cheb, **kwargs):
        self.max_clusters = max_clusters
        self.cheb = cheb

        super(DiffPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embed = GCN(features=input_shape[1][2], cheb=self.cheb)
        self.pool = GCN(features=self.max_clusters, cheb=self.cheb)

        self.batch_size = input_shape[1][0]

        super(DiffPool, self).build(input_shape)

    def call(self, x):
        # input is a tuple (A, X)
        filtres = x[0]
        X = x[1]
        num_features = X.shape[2]

        (_, S) = self.pool(x)
        (_, Z) = self.embed(x)

        # Z = tf.reshape(Z, [-1, num_features])

        S = tf.keras.activations.softmax(S, axis = 1)
        S_trans = tf.linalg.transpose(S)

        coarse_X = tf.matmul(S_trans, Z)

        # tf.print(S, summarize=-1)
        S = tf.reshape(S, [-1, self.max_clusters])
        tf.print(S, summarize=-1)
        # tf.print(S_trans, summarize=-1)
        S_trans = tf.reshape(S_trans, [self.max_clusters, -1])
        # tf.print(S_trans, summarize=-1)

        # TODO how do I deal with ChebNet's two filtres?
        # TODO coarse_A is not sparse anymore
        coarse_A = tf.sparse.sparse_dense_matmul(filtres[0], S)
        tf.print(coarse_A, summarize=-1)
        coarse_A = tf.matmul(S_trans, coarse_A)
        tf.print(coarse_A, summarize=-1)

        return (coarse_A, coarse_X)

    def compute_output_shape(self, input_shape):
        return (self.max_clusters, input_shape[0][2])