import tensorflow as tf
from layers_nocheb import *
from tensorflow.keras.layers import Dense, Flatten, Input, ReLU


class GCN32Max( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN32Max, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.pool_1 = SimplePool( mode="max" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN32Mean( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN32Mean, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.pool_1 = SimplePool( mode="mean" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN32Diff( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN32Diff, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.pool_1 = DiffPool( max_clusters=1 )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3232Max( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN3232Max, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=32, dropout=0.5 )
        self.pool_1 = SimplePool( mode="max" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3232Mean( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN3232Mean, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=32, dropout=0.5 )
        self.pool_1 = SimplePool( mode="max" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3232Reshape( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN3232Reshape, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=32, dropout=0.5 )
        self.pool_1 = ReshapeForDense()
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN32632Diff( tf.keras.Model ):

    def __init__(self, num_classes, avg_num_nodes):
        import math

        super( GCN32632Diff, self ).__init__()

        #         math.ceil(avg_num_nodes/10)
        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.pool_1 = DiffPool( max_clusters=math.ceil( avg_num_nodes / 10 ) )
        self.conv_2 = GCN( features=32, dropout=0.5 )
        self.pool_2 = DiffPool( max_clusters=1 )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.pool_1( x )
        x = self.conv_2( x )
        x = self.pool_2( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3264Max( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN3264Max, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=64, dropout=0.5 )
        self.conv_3 = GCN( features=32, dropout=0.5 )
        self.conv_4 = GCN( features=64, dropout=0.5 )
        self.pool_1 = SimplePool( mode="max" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.conv_3( x )
        x = self.conv_4( x )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3264Mean( tf.keras.Model ):

    def __init__(self, num_classes):
        super( GCN3264Mean, self ).__init__()

        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=64, dropout=0.5 )
        self.conv_3 = GCN( features=32, dropout=0.5 )
        self.conv_4 = GCN( features=64, dropout=0.5 )
        self.pool_1 = SimplePool( mode="mean" )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.conv_3( x )
        x = self.conv_4( x )
        x = self.pool_1( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x


class GCN3264Diff( tf.keras.Model ):

    def __init__(self, num_classes, avg_num_nodes):
        import math

        super( GCN3264Diff, self ).__init__()

        #         math.ceil(avg_num_nodes/10)
        self.conv_1 = GCN( features=32, dropout=0.5 )
        self.conv_2 = GCN( features=64, dropout=0.5 )
        self.pool_1 = DiffPool( max_clusters=math.ceil( avg_num_nodes / 10 ) )
        self.conv_3 = GCN( features=32, dropout=0.5 )
        self.conv_4 = GCN( features=64, dropout=0.5 )
        self.pool_2 = DiffPool( max_clusters=1 )
        self.classifier = Dense( num_classes, activation="softmax" )

    def call(self, inputs):
        filtre = inputs[0]
        X = inputs[1]
        node_indicator = inputs[2]

        x = self.conv_1( inputs )
        x = self.conv_2( x )
        x = self.pool_1( x )
        x = self.conv_3( x )
        x = self.conv_4( x )
        x = self.pool_2( x )
        x = Dense( 512, activation='relu' )( x[1] )
        x = self.classifier( x )

        return x