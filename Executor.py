from builders.ModelBuilder import Builder
from preprocessing.PrepareData import getData, readPdfText
from selector.ModelSelector import AvailableMLModels,MLModel
from selector.VectorSelector import AvailableVectorTypes, Vectorizer
from encoder.Encoder import Encoder
from helper.Util import importConfig
import json
import pickle
import pandas as pd
import numpy as np

def train(model,vectorType):
    #load configurations
    config = importConfig()

    #load data
    df = getData(config["_index"],config["_elasticSearchUrl"])

    #encode labels
    labels = Encoder(df['label'],config["_encodingPath"]).encode()

    #getting required vectorizer
    vectorType = vectorType
    vect = Vectorizer(vectorType,df['text']).getVectorizer()

    if model == AvailableMLModels.Boosting:
       vect = vect.tocsc()

    #getting required classifier
    model = model
    classifier = MLModel(model).getClassifier()

    #creating saved file name
    fileName = "models/" + model.name + "_" + vectorType.name + ".sav"

    #training the model
    classifier = Builder().train_model(classifier,vect,labels,fileName)

    # save the model to disk
    pickle.dump(classifier, open(fileName, 'wb'))

    #return response
    return "Training Completed"

def predict(model,vectorType,fileName): 
    #load configurations
    config = importConfig()

    #load pdf text and transform it into data series
    text = readPdfText(fileName)
    df_text = pd.Series(np.array([text]))

    #getting required vectorizer type
    vectorType = vectorType
    vect = Vectorizer(vectorType,df_text).getVectorizer()

    if model == AvailableMLModels.Boosting:
       vect = vect.tocsc()

    #getting required classifier
    model = model

    #creating saved file name
    modelFileName = "models/" + model.name + "_" + vectorType.name + ".sav"

    #load model
    classifier = pickle.load(open(modelFileName, 'rb'))

    #predict label
    encoded_labels = Builder().predict_model(classifier,vect,fileName)

    #decode labels
    decoded_labels = Encoder(pd.Series(encoded_labels),config["_encodingPath"]).decode()

    return decoded_labels.tolist()


#model = AvailableMLModels.NaiveBayes
#vectorType = AvailableVectorTypes.Word
#train(model,vectorType)
#print(predict(model,vectorType,"D:\\predictions\\NFPA 1-2009 Edition.pdf"))





