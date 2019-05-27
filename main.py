from helper_nocheb import *
from training import *
from models import *

from datetime import datetime


def main():
    batch_size = 3
    epochs = 2

    start = datetime.now()
    dataset = read_graphfile("datasets", "PROTEINS", max_nodes=None)

    train, val, test, num_classes = make_train_test_val(dataset)

    train, val, test = preprocess_dataset(train, val, test)
    end = datetime.now()

    print(f"\nPreprocessing time: {(end - start).total_seconds()} seconds")

    print(" ############## MAX POOL ############## ")
    train_model(GCNMaxPool(num_classes), train, val, test, epochs, batch_size)

    print(" ############## MEAN POOL ############## ")
    train_model(GCNMeanPool(num_classes), train, val, test, epochs, batch_size)

    print(" ############## DIFF POOL ############## ")
    train_model(GCNDiffPool(num_classes), train, val, test, epochs, batch_size)

    # TODO grafici

    return

main()


