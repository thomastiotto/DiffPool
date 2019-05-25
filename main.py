from helper_nocheb import *
from training import *
from models import *

import networkx as nx

def main():
    batch_size = 100
    epochs = 1


    dataset = read_graphfile("datasets", "PROTEINS", max_nodes=None)
    # node_indicator = make_node_indicator(dataset)
    train, val, test, num_classes = make_train_test_val(dataset)
    # train, val, test = reduce_dataset(train, val, test, batch_size, node_indicator)
    train, val, test = preprocess_dataset(train, val, test)

    model = GCNMaxPool(num_classes, batch_size)

    train_model(model, train, val, test, epochs, batch_size)

    # TODO grafici

    return


main()


