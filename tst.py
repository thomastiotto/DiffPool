import networkx as nx

from helper import *
from layers import *

cheb = True


def main():
    batch_size = 2
    epochs = 1
    img_rows, img_cols = 3, 3

    A = make_full_adjacency(img_rows)
    A_hat = A + np.eye(img_rows * img_cols)

    G = nx.from_numpy_matrix(np.array(A))
    nx.draw(G, with_labels=True)
    # plt.show()

    A_hat = normalise_adjacency_matrix(A_hat)

    if cheb:
        filtres = chebyshev_polynomials(A_hat)
    else:
        filtres = [A_hat]
    batch_filtres = [batch_adjacency_matrix(i, batch_size) for i in filtres]

    X = np.ones(img_cols * img_rows * 2, dtype=np.float32).reshape(batch_size, img_cols * img_rows, 1)

    from tensorflow.keras.layers import Dense, Flatten

    ###################################################

    test = GCN(features=2, cheb=cheb, input_shape=(img_cols * img_rows, 1))((batch_filtres, X))
    test = GCN(features=2, cheb=cheb)(test)

    # test = DiffPool()

    tf.print(test[1], summarize=-1)
    # test = Flatten()(test)
    # tf.print(test[1], summarize=-1)
    # test = Dense(10, activation='softmax')(test)
    # tf.print(test, summarize=-1)

    ###################################################

    return


main()
