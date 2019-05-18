import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.sparse as sparse



def load_dataset(img_rows, img_cols):
	from tensorflow.keras.datasets import mnist
	num_classes = 10

	img_rows, img_cols = 28, 28

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return (x_train, y_train), (x_test, y_test)


# make fully connected adjacency matrix
def make_full_adjacency(size):
	import pysal

	A = pysal.lib.weights.lat2W(size, size, rook=False, id_type="int")
	A, _ = A.full()
	A = np.array(A)
	A = A.astype(np.int32)

	return A


def calculate_degree_matrix(A):

	in_degree = np.sum(A, axis=0)
	out_degree = np.sum(A, axis=1)

	# diag = in_degree + out_degree
	diag = in_degree

	D = np.diag(diag) - np.eye(diag.size)

	return D


def reduce_dataset(train, validation, batch_size):
	(x_train, y_train) = train
	(x_test, y_test) = validation

	extra_elements = x_train.shape[0] % batch_size
	if extra_elements:
		x_train = x_train[0:x_train.shape[0] - extra_elements]
		y_train = y_train[0:y_train.shape[0] - extra_elements]

	extra_elements = x_test.shape[0] % batch_size
	if extra_elements:
		x_test = x_test[0:x_test.shape[0] - extra_elements]
		y_test = y_test[0:y_test.shape[0] - extra_elements]

	return (x_train, y_train), (x_test, y_test)


def normalise_adjacency_matrix(A, D):

	D = scipy.linalg.fractional_matrix_power(D, -0.5)
	A = np.linalg.multi_dot([D, A, D])
	A = sparse.csr_matrix(A)

	return A


def batch_adjacency_matrix(A, batch_size):
	batch_A_hat = sparse.kron(sparse.identity(batch_size), A)
	batch_A_hat = batch_A_hat.astype(np.float32)

	return batch_A_hat


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
		return tf.reshape(hidden, [batch_size, in_size, out_weights])

		# H = tf.sparse.sparse_dense_matmul(self.A, x)

		# return tf.matmul(H, self.W)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.F_prime)

	def convert_sparse_matrix_to_sparse_tensor(self, X):
		coo = X.tocoo()
		indices = np.mat([coo.row, coo.col]).transpose()
		return tf.SparseTensor(indices, coo.data, coo.shape)


class GCNPool(keras.layers.Layer):

	def __init__(self, **kwargs):

		super(GCNPool, self).__init__(**kwargs)

	def build(self, input_shape):
		self.F_prime = input_shape[2]
		self.in_size = input_shape[1]
		self.batch_size = input_shape[0]

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



def main():
	batch_size = 64
	epochs = 10
	img_rows, img_cols = 28, 28

	(x_train, y_train), (x_test, y_test) = load_dataset(img_rows, img_cols)
	(x_train, y_train), (x_test, y_test) = reduce_dataset((x_train, y_train), (x_test, y_test), batch_size)

	A = make_full_adjacency(img_rows)
	A_hat = A + np.eye(img_rows * img_cols)
	A_hat = A_hat.astype(np.float32)

	D_hat = calculate_degree_matrix(A_hat)

	A_hat = normalise_adjacency_matrix(A_hat, D_hat)

	batch_A_hat = batch_adjacency_matrix(A_hat, batch_size)

	# plt.subplot(221)
	# plt.spy(A_hat)
	# plt.subplot(222)
	# plt.spy(batch_A_hat)
	# plt.show()

	# plt.matshow(batch_A_hat.todense()[0:784*2, 0:784*2])
	plt.show()

	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense, ReLU

	###################################################

	A_test = batch_adjacency_matrix(A_hat, 2)
	test = GCN(A_test, F_prime=16, input_shape=(img_cols*img_rows, 1))(x_train[0:2])
	test = ReLU()(test)
	test = GCN(A_test, F_prime=2)(test)
	test = ReLU()(test)
	test = GCNPool()(test)
	test = Dense(10, activation='softmax')(test)
	tf.print(test)
	tf.print(y_train[0])
	tf.print(y_train[1])

	###################################################

	# model = Sequential()
	# model.add(GCN(batch_A_hat, F_prime=32, batch_size=batch_size, input_shape=(img_cols*img_rows, 1)))
	# model.add(ReLU())
	# model.add(GCN(batch_A_hat, F_prime=32, batch_size=batch_size, input_shape=(img_cols * img_rows, 1)))
	# model.add(GCNPool((img_rows, img_cols), F_prime=32, batch_size=batch_size))
	# model.add(Dense(10, activation='softmax'))
	#
	# model.compile(loss=keras.losses.categorical_crossentropy,
	#               optimizer=keras.optimizers.Adadelta(),
	#               metrics=['accuracy'])
	#
	# model.fit(x_train, y_train,
	#           batch_size=batch_size,
	#           epochs=epochs,
	#           verbose=1,
	#           validation_data=(x_test, y_test))
	# score = model.evaluate(x_test, y_test,
	#                        batch_size=batch_size,
	#                        verbose=1)
	#
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])

	return

main()