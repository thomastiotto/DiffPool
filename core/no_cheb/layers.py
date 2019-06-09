import numpy as np
import tensorflow as tf
from tensorflow import keras


class GCN(keras.layers.Layer):

    def __init__(self, features, dropout=0, **kwargs):
        self.F_prime = features
        self.dropout = dropout

        super(GCN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name="W",
                                 shape=(input_shape[1][1], self.F_prime),
                                 initializer=tf.initializers.GlorotUniform(),
                                 trainable=True)

        super(GCN, self).build(input_shape)

    def call(self, x):
        filtre = x[0]
        X = x[1]
        node_indicator = x[2]

        if self.dropout:
            X = tf.nn.dropout(X, rate=self.dropout)

        # do convolution
        hidden = tf.matmul(X, self.w)

        hidden = tf.matmul(filtre, hidden)

        return tf.tuple([filtre, tf.keras.activations.relu(hidden), node_indicator])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.F_prime)


class SimplePool(keras.layers.Layer):

    def __init__(self, mode, **kwargs):
        assert mode == "max" or mode == "mean", "GCNPool must have 'max' or 'mean' as mode"

        self.mode = mode

        super(SimplePool, self).__init__(**kwargs)

    def build(self, input_shape):

        super(SimplePool, self).build(input_shape)

    def call(self, x):
        filtre = x[0]
        X = x[1]
        node_indicator = x[2]

        if self.mode == "max":
            X = tf.math.segment_max(X, node_indicator)
        else:
            X = tf.math.segment_mean(X, node_indicator)

        return tf.tuple([filtre, X, node_indicator])

    def compute_output_shape(self, input_shape):
        return (input_shape[0] / self.in_size, self.F_prime)


class DiffPool(keras.layers.Layer):

    def __init__(self, max_clusters, **kwargs):
        self.max_clusters = max_clusters

        super(DiffPool, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pool = GCN(features=self.max_clusters)
        self.embed = GCN(features=input_shape[1][1])

        super(DiffPool, self).build(input_shape)

    def call(self, x):
        import scipy

        filtre = x[0]
        X = x[1]
        node_indicator = x[2]

        (_, S, _) = self.pool(x)
        (_, Z, _) = self.embed(x)

        S = tf.keras.activations.softmax(S, axis=1)

        # split tensors into the component graphs to be able to pool properly
        _, counts = np.unique(node_indicator, return_counts=True)
        # split to execute operations separately for each input graph as they all have different sizes
        split_S = tf.split(S, counts, axis=0)
        split_Z = tf.split(Z, counts, axis=0)
        split_A_0 = tf.split(filtre, counts, axis=0)

        # split the input adjacencies over rows and then columns
        split_A = []
        for i, a in enumerate(split_A_0):
            split_A.append(tf.split(a, counts, axis=1)[i])

        coarse_X_list = []
        for s, z in zip(split_S, split_Z):
            coarse_X_list.append(tf.matmul(s, z, transpose_a=True))

        coarse_A_list = []
        for a, s in zip(split_A, split_S):
            coarse_a_tmp = tf.matmul(a, s)
            coarse_A_list.append(tf.matmul(s, coarse_a_tmp, transpose_a=True))

        # put the output back to the same input format
        coarse_A = scipy.sparse.block_diag(coarse_A_list).todense().astype(np.float32)
        coarse_X = tf.concat(coarse_X_list, axis=0)

        # keeps track of which graph the new clusters belong to, as I'm not using an extra batch dimension
        coarse_node_indicator = []
        for i in range(len(counts)):
            coarse_node_indicator.append(np.full((self.max_clusters, 1), i))
        coarse_node_indicator = np.vstack(coarse_node_indicator).ravel()

        return tf.tuple([coarse_A, coarse_X, coarse_node_indicator])

    def compute_output_shape(self, input_shape):
        return (self.max_clusters, input_shape[1][2])


# Dense() knows nothing of my data format so bring it back to something it can understand
class ReshapeForDense(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ReshapeForDense, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeForDense, self).build(input_shape)

    def call(self, x):
        filtre = x[0]
        X = x[1]
        node_indicator = x[2]

        _, counts = np.unique(node_indicator, return_counts=True)
        self.num_clusters = counts[0]

        # flatten input per graph
        X = tf.reshape(X, [-1, self.num_clusters * X.shape[1]])

        return X

    def compute_output_shape(self, input_shape):
        return (self.num_clusters * input_shape[1][1])



def convert_sparse_matrix_to_sparse_tensor(X):
    import tensorflow as tf

    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.SparseTensor(indices, coo.data, coo.shape)