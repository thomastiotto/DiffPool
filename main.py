from tensorflow import keras

from helper import *
from layers import *

cheb = False


def main():
    batch_size = 100
    epochs = 20
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = load_dataset(img_rows, img_cols)
    (x_train, y_train), (x_test, y_test) = reduce_dataset((x_train, y_train), (x_test, y_test), batch_size)

    A = make_full_adjacency(img_rows)
    A_hat = A + np.eye(img_rows * img_cols)

    A_hat = normalise_adjacency_matrix(A_hat)

    if cheb:
        filtres = chebyshev_polynomials(A_hat)
    else:
        filtres = [A_hat]
    batch_filtres = [make_batch_filtres(i, batch_size) for i in filtres]

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, Flatten

    # model = Sequential()
    # model.add(Dropout(0.5, input_shape=(img_cols * img_rows, 1)))
    # model.add(GCN(features=128))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # model.add(GCN(features=128))
    # model.add(LeakyReLU(alpha=0.3))
    # # model.add(SimplePool(batch_size=batch_size, mode="mean"))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(10, activation='softmax'))

    x = GCN(features=2, cheb=cheb, dropout=0.5, input_shape=(img_cols * img_rows, 1))((batch_filtres, x_train))
    x = GCN(features=2, cheb=cheb)(x)

    model = Model(inputs=(batch_filtres, x_train), outputs=x)

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
