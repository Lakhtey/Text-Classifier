from sklearn import preprocessing
import numpy as np

class Encoder(object):

    def __init__(self, series, fileName):
        self.series = series
        self.fileName = fileName

    def encode(self):
        encoder = preprocessing.LabelEncoder()
        data_encoded = encoder.fit_transform(self.series)
        np.save(self.fileName, encoder.classes_)
        return data_encoded

    def decode(self):
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = np.load(self.fileName, allow_pickle=True)
        return encoder.inverse_transform(self.series)
