import matplotlib.pylab as plt
import numpy as np
import scipy
from tensorflow import keras


def load_dataset(img_rows, img_cols):
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
    in_degree = np.sum(A, axis=0)
    out_degree = np.sum(A, axis=1)

    # diag = in_degree + out_degree
    diag = out_degree

    # D = np.diag(diag) - np.eye(diag.size)
    D = np.diag(diag)

    return D


def reduce_dataset(train, validation, batch_size):
    (x_train, y_train) = train
    (x_test, y_test) = validation

    extra_elements = x_train.shape[0] % batch_size
    if extra_elements:
        x_train = x_train[0:x_train.shape[0] - extra_elements]
        y_train = y_train[0:y_train.shape[0] - extra_elements]

    extra_elements = x_test.shape[0] % batch_size
    if extra_elements:
        x_test = x_test[0:x_test.shape[0] - extra_elements]
        y_test = y_test[0:y_test.shape[0] - extra_elements]

    print(x_train.shape[0], 'REDUCED train samples')
    print(x_test.shape[0], 'REDUCED test samples')

    return (x_train, y_train), (x_test, y_test)


def normalise_adjacency_matrix(A):
    D = calculate_degree_matrix(A)
    D = scipy.linalg.fractional_matrix_power(D, -0.5)
    # A = np.linalg.multi_dot([D, A, D])
    A = A.dot(D).transpose().dot(D)
    return scipy.sparse.csr_matrix(A)


def batch_adjacency_matrix(filtre, batch_size):
    batch_A_hat = scipy.sparse.kron(scipy.sparse.identity(batch_size), filtre)
    batch_A_hat = batch_A_hat.astype(np.float32)

    return convert_sparse_matrix_to_sparse_tensor(batch_A_hat)


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



def chebyshev_polynomials(A):
    from scipy.sparse.linalg.eigen.arpack import eigsh

    laplacian = scipy.eye(A.shape[0]) - A
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.eye(A.shape[0])

    cheb = []
    cheb.append(scipy.sparse.csr_matrix(scipy.eye(A.shape[0])))
    cheb.append(scipy.sparse.csr_matrix(scaled_laplacian))

    return cheb
