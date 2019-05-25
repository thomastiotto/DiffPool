import networkx as nx

from helper_nocheb import *
from layers_nocheb import *


def main():
    batch_size = 2
    epochs = 1
    img_rows, img_cols = 3, 3

    dataset = read_graphfile("datasets", "PROTEINS", max_nodes=None)

    load_mnist(28, 28)


    A_hat = A + np.eye(img_rows * img_cols)
    A_hat = normalise_adjacency_matrix(A_hat)

    A_hat = np.repeat(A_hat[np.newaxis, :, :], batch_size, axis=0)

    X = np.ones(img_cols * img_rows * 2, dtype=np.float32).reshape(batch_size, img_cols * img_rows, 1)

    from tensorflow.keras.layers import Dense, Flatten

    ###################################################

    test = GCN(features=2, input_shape=(img_cols * img_rows, 1))([A_hat, X])
    # test = SimplePool(mode="max")(test)
    test = DiffPool(max_clusters=1)(test)
    test = Dense(10, activation="softmax")(test[1])

    tf.print(test, summarize=-1)
    # test = Flatten()(test)
    # tf.print(test[1], summarize=-1)
    # test = Dense(10, activation='softmax')(test)
    # tf.print(test, summarize=-1)

    ###################################################

    return


main()
