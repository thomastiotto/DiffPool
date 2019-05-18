import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt

from helper import *
from layers import *


def main():
	batch_size = 64
	epochs = 1
	img_rows, img_cols = 28, 28

	(x_train, y_train), (x_test, y_test) = load_dataset(img_rows, img_cols)
	(x_train, y_train), (x_test, y_test) = reduce_dataset((x_train, y_train), (x_test, y_test), batch_size)

	A = make_full_adjacency(img_rows)
	A_hat = A + np.eye(img_rows * img_cols)

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
	from tensorflow.keras.layers import Dense, ReLU, Dropout

	model = Sequential()
	model.add(Dropout(0.5, input_shape=(img_cols*img_rows, 1)))
	model.add(GCN(batch_A_hat, F_prime=16))
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(GCN(batch_A_hat, F_prime=8))
	model.add(ReLU())
	model.add(GCNPool(batch_size=batch_size))
	model.add(Dense(10, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adam(),
	              metrics=['accuracy'])

	history = model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))
	score = model.evaluate(x_test, y_test,
	                       batch_size=batch_size,
	                       verbose=1)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	return

main()