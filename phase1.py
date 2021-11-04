import ast
import spacy
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
import pickle
from ClassifierOfData import Classifier
import pandas as pd
from sklearn.model_selection import train_test_split


class Word2vec:

    def __init__(self,test, train):
        self.test = test
        self.train = train
        self.nlp = spacy.load('en_core_web_md')

    def _load(self):
        with open("data/phase1/x_train.pickle", "rb") as train:
            x_train = pickle.load(train)
        with open("data/phase1/y_train.pickle", "rb") as train:
            y_train = pickle.load(train)
        with open("data/phase1/x_test.pickle", "rb") as test:
            x_test = pickle.load(test)
        with open("data/phase1/y_test", "rb") as test:
            y_test = pickle.load(test)
        return x_train, y_train, x_test, y_test
        
    #convert data to usefull for classifire
    # input : genress & vector of each film , dictionary of all genres
    #output : vector of a film(vector of a file is vector of overview) and lable of it(length of lable if length of all diffrent genres if a genres 
    # is in film this cell is 1  else 0)
    def _split_to_data_lable(self,genres, list_vector):
        labels = []
        datum = []
        for info in list_vector:
            datum.append(info[1])
            temp = []
            for num in genres.keys():
                T = True
                for option in info[0]:
                    if num in dict(option).values():
                        T = False
                        temp.append(1)
                        break
                if T:
                    temp.append(0)
            labels.append(temp)
        return datum, labels
    #input:list vector from train data
    #output: dictionary af diffrent genres list
    def _get_id_and_genres(self, list_vector):
        id_name = {}
        for list_data in list_vector:
            for data in list_data[0]:
                id_name.setdefault(data['id'], data['name'])
        return id_name
    #input: an overview of film , output:vector of overview
    def _vector_from_band(self, band: str):
        list_vector = [0] * len(self.nlp(band.replace('\n', ' '))[0].vector)
        for word in self.nlp(band.replace('\n', ' ')):
            list_vector += word.vector
        list_vector /= len(self.nlp(band.replace('\n', ' ')))
        return list_vector
    #input:train.df 
    # output:a list that i'th cell including genres of i'th film and vector of overwiew of i'th film
    def _list_vector(self, input_data):
        list_vector = []
        for number, row in input_data.iterrows():
            list_vector.append([ast.literal_eval(row['genres']), self._vector_from_band(str(row['overview']))])
        return list_vector 
    
    def _pre_processing(self):
        # output:a list that i'th cell including genres of i'th film and vector of overwiew of i'th film
        list_vector1 = self._list_vector(self.train_df)
        #output: dictionary af diffrent genres list
        genres = self._get_id_and_genres(list_vector1)
        x_train, y_train = self._split_to_data_lable( genres,list_vector1)

        list_vector2 = self.list_vector(self.test_df)
        x_test, y_test = self._split_to_data_lable( genres,list_vector2)

        with open("data/phase1/x_train.pickle", "wb") as train:
            pickle.dump(x_train, train)
        with open("data/phase1/y_train.pickle", "wb") as train:
            pickle.dump(y_train, train)
        with open("data/phase1/x_test.pickle", "wb") as test:
            pickle.dump(x_test, test)
        with open("data/phase1/y_test", "wb") as test:
            pickle.dump(y_test, test)

    def run(self):
        self._preprocessing()
        x_train, y_train, x_test, y_test = self._load()
        Classifier(x_train, y_train, x_test, y_test).run()

# main
if __name__ == "__main__":

    Word2vec(pd.read_csv('dataset/test.csv', encoding='windows-1252'),pd.read_csv('dataset/train.csv', encoding='windows-1252')).run()