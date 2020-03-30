from keras import layers, models, optimizers

class ShallowNeuralNetwork(object):
    def __init__(self, input_size):
        self.input_size = input_size
    
    def create_shallow_NN(self):
        # create input layer 
        input_layer = layers.Input((self.input_size, ), sparse=True)
        
        # create hidden layer
        hidden_layer = layers.Dense(100, activation="relu")(input_layer)
        
        # create output layer
        output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

        classifier = models.Model(inputs = input_layer, outputs = output_layer)
        classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        return classifier 
    

class DeepNeuralNetwork(object):

    def __init__(self,word_index,embedding_matrix):
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix

    def create_cnn(self):
        # Add an Input Layer
        input_layer = layers.Input((70, ))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        return model

    def create_rnn_lstm(self):
        # Add an Input Layer
        input_layer = layers.Input((70, ))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.LSTM(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        return model

    def create_rnn_gru(self):
        # Add an Input Layer
        input_layer = layers.Input((70, ))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the GRU Layer
        lstm_layer = layers.GRU(100)(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        return model

    def create_bidirectional_rnn(self):
        # Add an Input Layer
        input_layer = layers.Input((70, ))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        # Add the LSTM Layer
        lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        return model

    def create_rcnn(self):
        # Add an Input Layer
        input_layer = layers.Input((70, ))

        # Add the word embedding Layer
        embedding_layer = layers.Embedding(len(self.word_index) + 1, 300, weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        
        # Add the recurrent layer
        rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
        
        # Add the convolutional Layer
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

        # Add the pooling Layer
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

        # Add the output Layers
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
        
        return model