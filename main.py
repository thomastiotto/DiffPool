from helper_nocheb import *
from training import *
from models import *

from datetime import datetime
from prettytable import PrettyTable
import math


def print_results(maxpool_acc, maxpool_loss, maxpool_time, meanpool_acc, meanpool_loss, meanpool_time, diffpool_acc,
                  diffpool_loss, diffpool_time):
    t = PrettyTable()

    t.title = "Results"
    t.field_names = ["Pooling", "Accuracy (%)", "Loss", "Time (s)"]
    t.add_row( ["MaxPool", round( maxpool_acc * 100, 2 ), round( maxpool_loss, 2 ), round( maxpool_time, 2 )] )
    t.add_row( ["MeanPool", round( meanpool_acc * 100, 2 ), round( meanpool_loss, 2 ), round( meanpool_time, 2 )] )
    t.add_row( ["DiffPool", round( diffpool_acc * 100, 2 ), round( diffpool_loss, 2 ), round( diffpool_time, 2 )] )
    print( t )


def main():
    batch_size = 32
    epochs = 100
    dataset_name = "PROTEINS"  # PROTEINS, ENZYMES, DD
    k_validation = 10

    start = datetime.now()

    dataset = read_graphfile( "datasets", dataset_name, max_nodes=None )
    avg_num_nodes = calculate_avg_nodes( dataset )

    train, val, test, num_classes = make_train_test_val( dataset )

    # print_dataset_stats(dataset, train, val, test, dataset_name, avg_num_nodes, num_classes)

    t = PrettyTable( header=False )
    t.title = dataset_name
    # t.field_names = ["Pooling"]
    t.add_row( ["Number of graphs", len( dataset )] )
    t.add_row( ["Number of classes", num_classes] )
    t.add_row( ["Average number of nodes", math.ceil( avg_num_nodes )] )
    t.add_row( ["Training samples (60%)", len( train[0] )] )
    t.add_row( ["Validation samples (20%)", len( val[0] )] )
    t.add_row( ["Testing samples (20%)", len( test[0] )] )
    print( t )

    train, val, test = preprocess_dataset( train, val, test )
    end = datetime.now()

    print( f"\nPreprocessing time: {(end - start).total_seconds()} seconds" )

    print( "GCN32Max" )
    maxpool_acc, maxpool_loss, maxpool_time = k_fold_validation( lambda:
                                                                 train_model( GCN32Max( num_classes ),
                                                                              train,
                                                                              val,
                                                                              test,
                                                                              epochs,
                                                                              batch_size ),
                                                                 k=k_validation )

    print( "GCN32Mean" )
    meanpool_acc, meanpool_loss, meanpool_time = k_fold_validation( lambda:
                                                                    train_model( GCN32Mean( num_classes ),
                                                                                 train,
                                                                                 val,
                                                                                 test,
                                                                                 epochs,
                                                                                 batch_size ),
                                                                    k=k_validation )

    print( "GCN32Diff" )
    diffpool_acc, diffpool_loss, diffpool_time = k_fold_validation( lambda:
                                                                    train_model(
                                                                            GCN32Diff( num_classes, avg_num_nodes ),
                                                                            train,
                                                                            val,
                                                                            test,
                                                                            epochs,
                                                                            batch_size ),
                                                                    k=k_validation )

    print_results( maxpool_acc, maxpool_loss, maxpool_time, meanpool_acc, meanpool_loss, meanpool_time, diffpool_acc,
                   diffpool_loss, diffpool_time )

    print( "GCN3232Max" )
    maxpool_acc, maxpool_loss, maxpool_time = k_fold_validation( lambda:
                                                                 train_model( GCN3232Max( num_classes ),
                                                                              train,
                                                                              val,
                                                                              test,
                                                                              epochs,
                                                                              batch_size ),
                                                                 k=k_validation )

    print( "GCN3232Mean" )
    meanpool_acc, meanpool_loss, meanpool_time = k_fold_validation( lambda:
                                                                    train_model( GCN3232Mean( num_classes ),
                                                                                 train,
                                                                                 val,
                                                                                 test,
                                                                                 epochs,
                                                                                 batch_size ),
                                                                    k=k_validation )

    print( "GCN3232Diff" )
    diffpool_acc, diffpool_loss, diffpool_time = k_fold_validation( lambda:
                                                                    train_model(
                                                                            GCN3232Diff( num_classes, avg_num_nodes ),
                                                                            train,
                                                                            val,
                                                                            test,
                                                                            epochs,
                                                                            batch_size ),
                                                                    k=k_validation )

    print_results( maxpool_acc, maxpool_loss, maxpool_time, meanpool_acc, meanpool_loss, meanpool_time, diffpool_acc,
                   diffpool_loss, diffpool_time )

    print( "GCN3264Max" )
    maxpool_acc, maxpool_loss, maxpool_time = k_fold_validation( lambda:
                                                                 train_model( GCN3264Max( num_classes ),
                                                                              train,
                                                                              val,
                                                                              test,
                                                                              epochs,
                                                                              batch_size ),
                                                                 k=k_validation )

    print( "GCN3264Mean" )
    meanpool_acc, meanpool_loss, meanpool_time = k_fold_validation( lambda:
                                                                    train_model( GCN3264Mean( num_classes ),
                                                                                 train,
                                                                                 val,
                                                                                 test,
                                                                                 epochs,
                                                                                 batch_size ),
                                                                    k=k_validation )

    print( "GCN3264Diff" )
    diffpool_acc, diffpool_loss, diffpool_time = k_fold_validation( lambda:
                                                                    train_model(
                                                                            GCN3264Diff( num_classes, avg_num_nodes ),
                                                                            train,
                                                                            val,
                                                                            test,
                                                                            epochs,
                                                                            batch_size ),
                                                                    k=k_validation )

    print_results( maxpool_acc, maxpool_loss, maxpool_time, meanpool_acc, meanpool_loss, meanpool_time, diffpool_acc,
                   diffpool_loss, diffpool_time )

    print( "GCN3232Reshape" )
    reshape_acc, reshape_loss, reshape_time = k_fold_validation( lambda:
                                                                 train_model(
                                                                         GCN3232Reshape( num_classes ),
                                                                         train,
                                                                         val,
                                                                         test,
                                                                         epochs,
                                                                         batch_size ),
                                                                 k=k_validation )
    print_results( maxpool_acc, maxpool_loss, maxpool_time, meanpool_acc, meanpool_loss, meanpool_time, reshape_acc,
                   reshape_loss, reshape_time )

    return


main()


