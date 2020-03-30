from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing import text, sequence
import numpy

def getCountVectorizer(series, analyzer='word', token_pattern=r'\w{1,}'):
    # create a count vectorizer object 
    count_vect = CountVectorizer(analyzer=analyzer, token_pattern=token_pattern)
    count_vect.fit(series)

    # transform the training and validation data using count vectorizer object
    return  count_vect.transform(series)

def getTfIdfVectorizer(series,level="w", token_pattern=r'\w{1,}',max_features=5000, ngram_range=(2,3)):
    """Create and return tfidf vectorizer. By default it does 
        Parameters
        ----------
        series : pandas.core.series.Series
            series to be vectorized.
        level : str, optional
            vectorizer level. It can have values as 'w' or 'ng' or 'c'. (The default is w)
        token_pattern : str, optional
            pattern to tokenize the words. (the default is '\w{1,}') 
        max_features : int, optional
            maximum features to collect. (the default is 5000)   
        ngram_range : tuple, optional
            ngram n value. (the default is (2,3))         
    """
    tfidf_vect = {}
    if level=="w":
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=token_pattern, max_features=max_features)
    elif level=="ng":
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=token_pattern, ngram_range=ngram_range, max_features=max_features)
    elif level=='c':
        tfidf_vect = TfidfVectorizer(analyzer='char', token_pattern=token_pattern, ngram_range=ngram_range, max_features=max_features)
    else:
        raise ValueError("Invalid level parameter: {}, should be 'w','ng' or 'c'".format(level))

    tfidf_vect.fit(series)
    train_tfidf =  tfidf_vect.transform(series)

    return train_tfidf

class WordEmbedder(object):
    def __init__(self,series,wordEmbedVectorPath):
        self.series = series
        self.wordEmbedVectorPath = wordEmbedVectorPath
        self.embeddings_index = self.getEmbeddingIndex()
        self.token = {}

    def getEmbeddingIndex(self):
        # load the pre-trained word-embedding vectors
        embeddings_index = {}
        for i, line in enumerate(open(self.wordEmbedVectorPath, encoding="utf8")):
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
        return embeddings_index

    def createTokenizer(self):
        # create a tokenizer 
        self.token = text.Tokenizer()
        self.token.fit_on_texts(self.series)

    def getsquenceTokens(self):
        # convert text to sequence of tokens and pad them to ensure equal length vectors 
        train_seq = sequence.pad_sequences(self.token.texts_to_sequences(self.series), maxlen=70)
        return train_seq

    def getWordEmbeddings(self):
        word_index = self.token.word_index
        # create token-embedding mapping
        embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix




    