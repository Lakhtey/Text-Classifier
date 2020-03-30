from sklearn import metrics

class Builder(object):

    def train_model(self, classifier, feature_vector_train, label, fileName):
        # fit the training dataset on the classifier
        classifier.fit(feature_vector_train, label)
        
        return classifier
        

    def predict_model(self, classifier, feature_vector_valid, fileName , is_neural_net=False):
        # predict the labels on validation dataset
        predictions = classifier.predict(feature_vector_valid)
        
        if is_neural_net:
            predictions = predictions.argmax(axis=-1)
        
        #return metrics.accuracy_score(predictions, valid_y)

        return predictions