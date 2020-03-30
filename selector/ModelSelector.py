from enum import Enum
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
from CustomLibs.sklearn_neural import ShallowNeuralNetwork, DeepNeuralNetwork
import xgboost

class AvailableMLModels(Enum):
    NaiveBayes = 1
    LogisticRegression = 2
    SVM = 3
    RandomForest = 4
    Boosting = 5

class AvailableShallowNeuralNetworkModels(Enum):
    SNN = 6

class AvailableDeepNeuralNetworkModels(Enum):
    CNN = 7
    RNN_LSTM=8
    RNN_GRU=9
    RNN_BiDirectional = 10
    RCNN = 11

class MLModel(object):
    def __init__(self,modelType):
        self.modelType = modelType
    
    def getClassifier(self):
        if self.modelType == AvailableMLModels.NaiveBayes:
            return naive_bayes.MultinomialNB()
        elif self.modelType == AvailableMLModels.LogisticRegression:
            return linear_model.LogisticRegression()
        elif self.modelType == AvailableMLModels.SVM:
            return svm.SVC()
        elif self.modelType == AvailableMLModels.RandomForest:
            return ensemble.RandomForestClassifier()
        elif self.modelType == AvailableMLModels.Boosting:
            return xgboost.XGBClassifier()
        else:
            raise ValueError("Invalid modelType parameter")

class ShallowNeuralNetworkModel(object):
    def __init__(self, modelType, input_size):
        self.modelType = modelType
        self.input_size = input_size
    
    def getClassifier(self):
        obj = ShallowNeuralNetwork(self.input_size)
        if self.modelType == AvailableShallowNeuralNetworkModels.SNN:
            return obj.create_shallow_NN()
        else:
            raise ValueError("Invalid modelType parameter")

class DeepNeuralNetworkModel(object):
    def __init__(self, modelType, word_index, embedding_matrix):
        self.modelType = modelType
        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
    
    def getClassifier(self):
        obj = DeepNeuralNetwork(self.word_index, self.embedding_matrix)
        if self.modelType == AvailableDeepNeuralNetworkModels.CNN:
            return obj.create_cnn()
        elif self.modelType == AvailableDeepNeuralNetworkModels.RNN_LSTM:
            return obj.create_rnn_lstm()
        elif self.modelType == AvailableDeepNeuralNetworkModels.RNN_GRU:
            return obj.create_rnn_gru()
        elif self.modelType == AvailableDeepNeuralNetworkModels.RNN_BiDirectional:
            return obj.create_bidirectional_rnn()
        elif self.modelType == AvailableDeepNeuralNetworkModels.RCNN:
            return obj.create_rcnn()
        else:
            raise ValueError("Invalid modelType parameter")

            




