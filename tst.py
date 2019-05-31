import networkx as nx

from helper_nocheb import *
from layers_nocheb import *
from training import make_batch

from tensorflow.keras.layers import Dense


def main():
    batch_size = 2

    dataset = read_graphfile("datasets", "ENZYMES", max_nodes=None)

    train, val, test, num_classes = make_train_test_val(dataset)
    avg_num_nodes = calculate_avg_nodes(dataset)

    train, val, test = preprocess_dataset(train, val, test)

    batch = make_batch(train, batch_size)

    for x, a, y, ind in batch:

        test = GCN(features=4, dropout=0.5)((a, x, ind))
        test = DiffPool(max_clusters=2)(test)
        test = GCN(features=4, dropout=0.5)(test)
        test = DiffPool(max_clusters=1)(test)
        # print("A:")
        # tf.print(test[0], summarize=-1)
        # print("data:")
        # tf.print(test[1], summarize=-1)
        # print("indicator:")
        # tf.print(test[2], summarize=-1)

        test = ReshapeForDense()(test)
        print("data reshaped:")
        tf.print(test, summarize=-1)

        test = Dense(num_classes, activation="softmax")(test)
        print("Classification:")
        tf.print(test, summarize=-1)

        return


main()
