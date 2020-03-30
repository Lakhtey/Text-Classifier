from feature_engineering.FeatureEngineer import getCountVectorizer, getTfIdfVectorizer
from enum import Enum

class AvailableVectorTypes(Enum):
    Count = 1
    Word = 2
    NGram = 3
    Character = 4

class Vectorizer(object):
    def __init__(self,vectorType, series):
        self.vectorType = vectorType
        self.series = series

    def getVectorizer(self):
        if self.vectorType == AvailableVectorTypes.Count:
            return getCountVectorizer(self.series)
        elif self.vectorType == AvailableVectorTypes.Word:
            return getTfIdfVectorizer(self.series)
        elif self.vectorType == AvailableVectorTypes.NGram:
            return getTfIdfVectorizer(self.series,level="ng")
        elif self.vectorType == AvailableVectorTypes.Character:
            return getTfIdfVectorizer(self.series,level="c")
