import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt

from helper import *
from layers import *


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

	# A_test = batch_adjacency_matrix(A_hat, 2)
	# test = GCN(A_test, F_prime=16, input_shape=(img_cols*img_rows, 1))(x_train[0:2])
	# test = ReLU()(test)
	# test = GCN(A_test, F_prime=2)(test)
	# test = ReLU()(test)
	# test = GCNPool()(test)
	# test = Dense(10, activation='softmax')(test)
	# tf.print(test)
	# tf.print(y_train[0])
	# tf.print(y_train[1])

	###################################################

	model = Sequential()
	model.add(GCN(batch_A_hat, F_prime=32, input_shape=(img_cols*img_rows, 1)))
	model.add(ReLU())
	model.add(GCN(batch_A_hat, F_prime=32))
	model.add(ReLU())
	model.add(GCNPool(batch_size=batch_size))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test,
	                       batch_size=batch_size,
	                       verbose=1)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return

main()