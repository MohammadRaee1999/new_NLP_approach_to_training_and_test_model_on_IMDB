from keras.layers import Dense
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras import Sequential
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, accuracy_score, jaccard_score
from sklearn.svm import LinearSVC
import pandas as pd



class Classifier:

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    #scale data
    def _scale(self) :
        ss = StandardScaler()
        ss.fit(self.x_train)
        self.x_test = ss.transform(self.x_test)
        self.x_train = ss.transform(self.x_train)
    
    #train mlp mpdel
    def _mlp(self,y_train, x_train):
        model = Sequential()
        model.add(Dense(200, input_dim=x_train[0], activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(y_train[0], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(np.array(x_train), np.array(y_train), batch_size=30, epochs=30)
        return model
    #input : x_test
    #job : give x_test to mlp and set treshold on output of model to take read prediction
    #output : prediction of x_test whit mlp
    def _pred_mlp(self, model, x_test):
        fired_list = []
        for pdata in model.predict(np.array(x_test)):
            temp = []
            for i in pdata:
                if i > 0.4:
                    temp.append(1)
                else:
                    temp.append(0)
            fired_list.append(temp)
        return fired_list
    
    #first set classifire whit leanear svm ,then fit this with x_train, y_train, then pass x_test to it for prediction and pass list of this to output
    def _svm_classifier(self, x_train, y_train, x_test):
        return list(MultiOutputClassifier(LinearSVC()).fit(x_train, y_train).predict(x_test))
    #input : trust lable , predicted lable
    #output : number of lable that predicted true / total lable
    #first set classifire whit KNeighbors,then fit this with x_train, y_train, then pass x_test to it for prediction and pass list of this to output
    def _kn_classifier(self, x_train, y_train, x_test):
        return list(MultiOutputClassifier(KNeighborsClassifier()).fit(x_train, y_train).predict(x_test))
    
    def _acc(self, trust, pred):
        length = len(trust[0])
        total_lenth = len(trust)
        for p in range(len(trust)):
            for q in range(length):
                if trust[p][q] != pred[p][q]:
                    total_lenth -= 1
                    break
        return total_lenth / len(trust)
    
    def run(self):
        self._scale()
        model = self._mlp(self.x_train, self.y_train)
        print("acc of mlp: " + str(jaccard_score(self.y_test,self._pred_mlp(model, self.x_test), average='samples')))
        print("acc of svm_classifier: " + str(jaccard_score(self.y_test, self._svm_classifier(self.x_train, self.y_train, self.x_test), average='samples')))
        print("acc of kn_classifier: " + str(jaccard_score(self.y_test, self._kn_classifier(self.x_train, self.y_train, self.x_test), average='samples')))