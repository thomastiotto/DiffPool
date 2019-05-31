import matplotlib.pylab as plt
import scipy
from tensorflow import keras
import networkx as nx
import numpy as np
import os
import re
from random import shuffle


def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph['label'] = graph_labels[i - 1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))

    return graphs


def make_train_test_val(dataset):

    num_classes = len(np.unique([get_graph_label(i) for i in dataset]))

    shuffle(dataset)

    train, validate, test = np.split(dataset, [int(.6*len(dataset)), int(.8*len(dataset))])

    graph_counter = 0

    x_train = []
    a_train = []
    y_train = []
    ind_train = []
    for g in train:
        x_train.append(get_graph_node_features(g))
        a_train.append(get_graph_adjacency(g))
        y_train.append(keras.utils.to_categorical(get_graph_label(g), num_classes))
        ind_train.append(make_node_indicator(g, graph_counter))
        graph_counter += 1

    x_val = []
    a_val = []
    y_val = []
    ind_val = []
    for g in validate:
        x_val.append(get_graph_node_features(g))
        a_val.append(get_graph_adjacency(g))
        y_val.append(keras.utils.to_categorical(get_graph_label(g), num_classes))
        ind_val.append(make_node_indicator(g, graph_counter))
        graph_counter += 1

    x_test = []
    a_test = []
    y_test = []
    ind_test = []
    for g in test:
        x_test.append(get_graph_node_features(g))
        a_test.append(get_graph_adjacency(g))
        y_test.append(keras.utils.to_categorical(get_graph_label(g), num_classes))
        ind_test.append(make_node_indicator(g, graph_counter))
        graph_counter += 1

    return (x_train, a_train, y_train, ind_train), (x_val, a_val, y_val, ind_val), (x_test, a_test, y_test, ind_test), num_classes


def make_node_indicator(g, i):

    return np.full((len(g.nodes()), 1), i)


def get_graph_label(g):

    return g.graph["label"].astype("float32")


def get_graph_node_features(g):

    features = np.array([], dtype=np.float32)

    for n in g.nodes:
        feat = g.nodes[n]["feat"]
        features = np.vstack([features, feat]) if features.size else feat

    return features.astype("float32")


def get_graph_adjacency(g):

    return nx.adjacency_matrix(g).astype("float32")


def calculate_avg_nodes(dataset):

    return np.average( [len(i.nodes) for i in dataset] )


def print_dataset_stats(dataset, train, val, test, dataset_name, avg_num_nodes, num_classes):
    import math

    print(f"################ {dataset_name} ################")
    print(f"Number of graphs \t\t\t {len(dataset)}")
    print(f"Number of classes \t\t\t {num_classes}")
    print(f"Average number of nodes \t {math.ceil(avg_num_nodes)}")
    print(f"Train \t\t\t\t\t\t {len(train[0])}")
    print(f"Val \t\t\t\t\t\t {len(val[0])}")
    print(f"Test \t\t\t\t\t\t {len(test[0])}")

    return


def preprocess_dataset(train, validation, test):
    for i in range(len(train[1])):
        train[1][i] = train[1][i] + scipy.sparse.eye(train[1][i].shape[0])
        train[1][i] = normalise_adjacency_matrix(train[1][i])

    for i in range(len(validation[1])):
        validation[1][i] = validation[1][i] + scipy.sparse.eye(validation[1][i].shape[0])
        validation[1][i] = normalise_adjacency_matrix(validation[1][i])

    for i in range(len(test[1])):
        test[1][i] = test[1][i] + scipy.sparse.eye(test[1][i].shape[0])
        test[1][i] = normalise_adjacency_matrix(test[1][i])

    return train, validation, test


def load_mnist(img_rows, img_cols):
    from tensorflow.keras.datasets import mnist
    num_classes = 10

    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'ORIGINAL train samples')
    print(x_test.shape[0], 'ORIGINAL test samples')

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
    out_degree = np.sum(A, axis=1)
    out_degree = out_degree.A1

    D = np.diag(out_degree)

    return D


def reduce_dataset(train, validation, test, batch_size, indicator):
    (x_train, adj_train, y_train) = train
    (x_val, adj_val, y_val) = validation
    (x_test, adj_test, y_test) = test

    print(len(x_train), 'ORIGINAL train samples')
    print(len(x_val), 'ORIGINAL validation samples')
    print(len(x_test), 'ORIGINAL test samples')

    extra_elements = len(x_train) % batch_size
    if extra_elements:
        x_train = x_train[:-extra_elements or None]
        adj_train = adj_train[:-extra_elements or None]
        y_train = y_train[:-extra_elements or None]
        ind_train = indicator[:len(x_train)-extra_elements]

    extra_elements = len(x_val) % batch_size
    if extra_elements:
        x_val = x_val[:-extra_elements or None]
        adj_val = adj_val[:-extra_elements or None]
        y_val = y_val[:-extra_elements or None]

    extra_elements = len(x_test) % batch_size
    if extra_elements:
        x_test = x_test[:-extra_elements or None]
        adj_test = adj_test[:-extra_elements or None]
        y_test = y_test[:-extra_elements or None]

    print(len(x_train), 'REDUCED train samples')
    print(len(x_val), 'REDUCED validation samples')
    print(len(x_test), 'REDUCED test samples')

    return train, validation, test


def normalise_adjacency_matrix(A):
    D = calculate_degree_matrix(A)
    D = scipy.linalg.fractional_matrix_power(D, -0.5)
    D = scipy.sparse.csr_matrix(D)
    # A = np.linalg.multi_dot([D, A, D])
    A = A.dot(D).transpose().dot(D)

    return A.astype(np.float32)


def make_batch_filtres(filtre, size):
    # batch_A_hat = scipy.sparse.kron(scipy.sparse.identity(batch_size), filtre)
    # batched = scipy.sparse.kron(np.ones(size), filtre)
    batched = np.repeat(filtre[np.newaxis, :, :], size, axis=0)

    batched = batched.astype(np.float32)

    return batched


def plot_data(X, Y):
    plt.imshow(X.reshape(28, 28), cmap='Greys')
    plt.title("Ground truth: " + str(np.where(Y == 1)[0][0]))
    plt.show()

    return


def convert_sparse_matrix_to_sparse_tensor(X):
    import tensorflow as tf

    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()

    return tf.SparseTensor(indices, coo.data, coo.shape)
