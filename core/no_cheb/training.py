import tensorflow as tf
import numpy as np
import scipy


def make_batch(dataset, batch_size):
    import math

    x, a, y, ind = dataset

    from sklearn.utils import shuffle
    x, a, y, ind = shuffle(x, a, y, ind)

    num_batches = math.ceil(len(x) / batch_size)

    for i in range(num_batches):
        batch_begin = i * batch_size
        batch_end = (i + 1) * batch_size

        batch_ind = ind[batch_begin: batch_end]
        batch_x = x[batch_begin: batch_end]
        batch_a = a[batch_begin: batch_end]
        batch_y = y[batch_begin: batch_end]

        # sort graphs based on indicator function
        batch_ind, batch_x, batch_a, batch_y = list(map(list, zip(*sorted(zip(batch_ind, batch_x, batch_a, batch_y),
                                                                          key=lambda x: x[0][0]))))

        # scale indicator
        for i, el in enumerate(batch_ind):
            batch_ind[i] = el - el + i

        batch_x = np.concatenate(batch_x).astype(np.float32)
        batch_a = scipy.sparse.block_diag(batch_a).todense().astype(np.float32)
        batch_y = np.vstack(batch_y).astype(np.float32)
        batch_ind = np.vstack(batch_ind).ravel()

        yield batch_x, batch_a, batch_y, batch_ind


loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)

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


def val_step(model, batch):
    x_batch_val, a_batch_val, y_batch_val, batch_indicator_val = batch

    predictions = model((a_batch_val, x_batch_val, batch_indicator_val))
    loss = loss_object(y_batch_val, predictions)

    # Update metrics
    val_loss(loss)
    val_accuracy(y_batch_val, predictions)


def test_step(model, batch):
    x_batch_test, a_batch_test, y_batch_test, batch_indicator_test = batch

    predictions = model((a_batch_test, x_batch_test, batch_indicator_test))
    loss = loss_object(y_batch_test, predictions)

    # Update metrics
    test_loss(loss)
    test_accuracy(y_batch_test, predictions)


def train_model(model, train, validation, test, epochs, batch_size):
    from datetime import datetime
    from tqdm import tqdm_notebook as tqdm

    printing_steps = 10

    start_training = datetime.now()

    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        # if epoch % printing_steps == 0 or epoch == epochs:
            # print("Epoch %d " % (epoch), end="\t")

        start_time = datetime.now()

        train_data = make_batch(train, batch_size)
        val_data = make_batch(validation, batch_size)
        test_data = make_batch(test, batch_size)

        # Iterate over the batches of the dataset
        for step, batch_train in enumerate(train_data):
            train_step(model, batch_train)

        # Run a validation loop at the end of each epoch.
        for step, batch_val in enumerate(val_data):
            val_step(model, batch_val)
        # if epoch % printing_steps == 0 or epoch == epochs:
            # print(f"Acc: {float(val_accuracy.result() * 100):.2f}  Loss: {float(val_loss.result() * 100):.2f}", end="  ")

        # Reset metrics at the end of each epoch
        train_accuracy.reset_states()
        train_loss.reset_states()
        val_accuracy.reset_states()
        val_loss.reset_states()

        end_time = datetime.now()
        # if epoch % printing_steps == 0 or epoch == epochs:
        #     print(f"Time: {(end_time - start_time).total_seconds():.2f} seconds")

    # Run a test loop at the end of training
    for step, batch_test in enumerate(test_data):
        test_step(model, batch_test)

    # print(f"Test accuracy at end of training: {float(test_accuracy.result() * 100):.2f}")
    # print(f"Test loss at end of training: {float(test_loss.result() * 100):.2f}")

    # test_accuracy.reset_states()
    # test_loss.reset_states()

    training_time = datetime.now() - start_training

    return test_accuracy.result(), test_loss.result(), training_time.total_seconds()


def k_fold_validation(training, k=10):
    acc_list = []
    loss_list = []
    time_list = []

    for i in range(k):
        acc, lss, tim = training()
        acc_list.append(acc)
        loss_list.append(lss)
        time_list.append(tim)

    return np.mean(acc_list), np.mean(loss_list), np.mean(time_list)
