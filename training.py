import tensorflow as tf
import numpy as np
import scipy


def make_batch(dataset, batch_size):
    import math

    x, a, y, ind = dataset

    from sklearn.utils import shuffle
    # x, a, y, ind = shuffle(x, a, y, ind)

    num_batches = math.ceil(len(x) / batch_size)

    for i in range(num_batches):
        batch_x = np.concatenate(x[i * batch_size: (i + 1) * batch_size]).astype(np.float32)
        batch_a = scipy.sparse.block_diag(a[i * batch_size: (i + 1) * batch_size]).todense().astype(np.float32)
        batch_y = np.vstack(y[i * batch_size: (i + 1) * batch_size]).astype(np.float32)
        batch_ind = np.vstack(ind[i * batch_size: (i + 1) * batch_size]).ravel()
        batch_ind -= batch_ind[0] #rescale indicator so as not to have problems in SimplePool

        yield batch_x, batch_a, batch_y, batch_ind


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.losses.Mean(name='train_loss', dtype=tf.float32)
val_loss = tf.keras.losses.Mean(name='val_loss', dtype=tf.float32)
test_loss = tf.keras.losses.Mean(name='test_loss', dtype=tf.float32)

train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


def train_step(model, batch):
    x_batch_train, a_batch_train, y_batch_train, batch_indicator_train = batch

    with tf.GradientTape() as tape:
        predictions = model((a_batch_train, x_batch_train, batch_indicator_train))
        loss = loss_object(y_batch_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update training metrics
    train_loss(loss)
    train_accuracy(y_batch_train, predictions)


def test_step(model, batch):
    x_batch_val, a_batch_val, y_batch_val, batch_indicator_val = batch

    val_predictions = model((a_batch_val, x_batch_val, batch_indicator_val))
    val_loss = loss_object(y_batch_val, val_predictions)

    # Update val metrics
    val_loss(val_loss)
    val_accuracy(y_batch_val, val_predictions)


def train_model(model, train, validation, test, epochs, batch_size):
    from timeit import default_timer as timer
    from tqdm import tqdm_notebook as tqdm

    x_train, a_train, y_train, ind_train = train
    x_val, a_val, y_val, ind_val = validation
    x_test, a_test, y_test, ind_test = test

    train_data = make_batch(train, batch_size)

    # Iterate over epochs
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset
        for step, batch_train in tqdm(enumerate(train_data), total=len(x_train) / batch_size):
            train_step(model, batch_train)

        # Display metrics at the end of each epoch.
        train_acc = train_accuracy.result()
        train_lss = train_loss.result()
        print("Training accuracy over epoch %s: %s" % (epoch, float(train_acc)))
        print("Training loss over epoch %s: %s" % (epoch, float(train_lss)))

        # Reset training metrics at the end of each epoch
        train_accuracy.reset_states()
        train_loss.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, batch_val in tqdm(enumerate(train_data), total=len(x_val) / batch_size):
            test_step(model, batch_val)

        val_acc = val_accuracy.result()
        val_lss = val_loss.result()

        print("Validation accuracy at end of epoch %s: %s" % (epoch, float(val_acc)))
        print("Validation loss at end of epoch %s: %s" % (epoch, float(val_lss)))

        val_accuracy.reset_states()
        val_loss.reset_state()


        # Run a test loop at the end of each training
        for step, batch_test in tqdm(enumerate(x_test), total=len(x_test) / batch_size):
            test_step(model, batch_test)

        test_acc = test_accuracy.result()
        test_lss = test_loss.result()

        print("Test accuracy at end of training: %s" % (float(test_acc)))
        print("Validation loss at end of training: %s" % (float(test_lss)))

        test_accuracy.reset_states()
        test_loss.reset_state()
