import tensorflow as tf
from tensorflow import keras
import numpy as np

class GCN(keras.layers.Layer):

	def __init__(self, A_hat, F_prime, **kwargs):
		self.A = self.convert_sparse_matrix_to_sparse_tensor(A_hat)
		self.F_prime = F_prime

		super(GCN, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight(name='W',
                                shape=(input_shape[2], self.F_prime),
	                            initializer='uniform',
	                            trainable=True)

		super(GCN, self).build(input_shape)

	def call(self, x):
		# x = tf.cast(x, tf.float64)
		batch_size = x.shape[0]
		in_size = x.shape[1]
		in_weights = x.shape[2]
		out_weights = self.W.shape[1]

		x = tf.reshape(x, [-1, in_weights])

		hidden = tf.matmul(x, self.W)

		hidden = tf.sparse.sparse_dense_matmul(self.A, hidden)

		# return tf.expand_dims(hidden, 0)
		return tf.reshape(hidden, [-1, in_size, out_weights])

		# H = tf.sparse.sparse_dense_matmul(self.A, x)

		# return tf.matmul(H, self.W)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.F_prime)

	def convert_sparse_matrix_to_sparse_tensor(self, X):
		coo = X.tocoo()
		indices = np.mat([coo.row, coo.col]).transpose()
		return tf.SparseTensor(indices, coo.data, coo.shape)


class GCNPool(keras.layers.Layer):

	def __init__(self, batch_size, **kwargs):

		self.batch_size = batch_size

		super(GCNPool, self).__init__(**kwargs)

	def build(self, input_shape):
		self.F_prime = input_shape[2]
		self.in_size = input_shape[1]

		super(GCNPool, self).build(input_shape)

	def call(self, x):
		# segment_ids = np.array([], dtype=np.int32).reshape(0, self.img_rows * self.img_cols)
		segment_ids = np.array([], dtype=np.int32).reshape(0, self.in_size)

		for b in range(self.batch_size):
			# index = np.repeat(b, self.img_rows * self.img_cols)
			index = np.repeat(b, self.in_size)
			segment_ids = np.concatenate((segment_ids, index), axis=None)

		x = tf.reshape(x, [-1, self.F_prime])

		return tf.math.segment_max(x, segment_ids)


	def compute_output_shape(self, input_shape):
		return (input_shape[0] / self.in_size, self.F_prime)
		# return (input_shape[0] / (self.img_rows * self.img_cols), self.F_prime)