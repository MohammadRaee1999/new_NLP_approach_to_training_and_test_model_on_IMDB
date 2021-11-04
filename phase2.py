from collections import Counter
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.layers import Dense
from sklearn.datasets import make_multilabel_classification
import pickle
import math
from GeneralClassifier import Classifier


class Cbow:

    def __init__(self,test, train):
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.max_length = 2000000
        self.train_df = train
        self.test_df = test
    
    def _load(self):
        with open("data/phase2/x_train.pickle", "rb") as train:
            x_train = pickle.load(train)
        with open("data/phase2/y_train.pickle", "rb") as train:
            y_train = pickle.load(train)
        with open("data/phase2/x_test.pickle", "rb") as test:
            x_test = pickle.load(test)
        with open("data/phase2/y_test", "rb") as test:
            y_test = pickle.load(test)
        return x_train, y_train, x_test, y_test

    #input : list_indext(cell i include genres of i'th film & overview of i'th film)
    #output : dictionary of genres
    def _get_id_and_genres(self, list_vector):
        id_name = {}
        for list_data in list_vector:
            for data in list_data[0]:
                id_name.setdefault(data['id'], data['name'])
        return id_name

    #calculate tf_idf for train data
    def _impotance_whit_tfidf(self,dictionary,l1, word: str, index, index_list):
        tf = 0
        num = Counter(l1[dictionary.get(word)])
        val = list(l1[dictionary.get(word)].values())
        keys = num.keys()
        for ind, data in enumerate(keys):
            if data == index:
                tf = val[ind]
                break
        return tf * math.log(len(index_list) / len(set(l1[dictionary.get(word)])))

    #_exist_train_data_for_classifier
    #input :
    # list_indext(cell i include genres of i'th film & overview of i'th film)
    # dictionary(dictionary of words)
    # in_list(index of film for each word)
    # genres(dictionary of genres)
    #output : x_train , y_train
    def _train_usefull_data(self, in_list, genres,list_index, dictionary):
        x_train = []
        y_train = []
        for number, text in enumerate(list_index):#number = number of film , text[0] = genres & id , text[1] = overview
            temp = [0] * len(dictionary.keys()) #length of temp = length of dictionary
            for word in self.nlp(text[1]):#for each word in overview
                temp[int(dictionary.get(str(word)))] = self._impotance_whit_tfidf(dictionary, in_list, str(word), number, list_index) #temp[word] = tfidf
            x_train.append(temp.copy()) #temp is representation of film's overview with verctor, that each word's value is tfidf
            temp = []
            for id_g in genres.keys():
                sw = True
                for dict_value in text[0]:
                    if id_g in dict(dict_value).values():
                        temp.append(1)
                        sw = False
                        break
                if sw:
                    temp.append(0)
            y_train.append(temp) #temp size = dictionary of genres , if i'th genre in dictionary of genres , was in this film's genre --> assigne 1 to temp[i] else 0
        return x_train, y_train
    
    #_exist_test_data_for_classifier
    def _test_usefull_data(self, list_index, genres, num, dictionary, in_list):
        x_test = []
        y_test = []
        for number, text in enumerate(list_index):#number = number of film , text[0] = genres & id , text[1] = overview
            temp = [0] * len(dictionary.keys()) #length of temp = length of dictionary
            try:
                for word in self.nlp(text[1]):#for each word in overview
                    string = dictionary.get(str(word))
                    if string is None:
                        continue
                    temp[int(string)] += math.log(num / len(set(in_list[int(string)])))

            except Exception as e:
                pass
            x_test.append(temp.copy()) #temp is representation of film's overview with verctor, that each word's value is tfpdf
            temp = []
            for id_g in genres.keys():
                sw = True
                for dict_value in text[0]:
                    if id_g in dict(dict_value).values():
                        temp.append(1)
                        sw = False
                        break
                if sw:
                    temp.append(0)
            y_test.append(temp) #temp size = dictionary of genres , if i'th genre in dictionary of genres , was in this film's genre --> assigne 1 to temp[i] else 0
        return x_test, y_test

    #input : train.df
    #output : list_indext(cell i include genres of i'th film & overview of i'th film) , text(cell i include i'th film overview)
    def _text_index_of_data(self, df):
        list_index = []
        text = []
        for film_id, band in df.iterrows():
            text.append(band['overview'])
            list_index.append([ast.literal_eval(band['genres']), band['overview']])
        return list_index, text

    #input :  list of overview
    #output : dictionary (dictionary of all word),in_list(index of film for each word)
    def _dictionary_whit_index_list(self, band):
        in_list = []
        dictionary = {}
        num = 0
        for number, text in enumerate(band):#number : number of overview(number of film) , text : overview
            try:
                datum = self.nlp(text)
            except Exception:
                continue
            for data in datum:#for each word in overview
                if str(data) not in dictionary:#if word is'nt in dictionary :
                    in_list.append([number])
                    dictionary.setdefault(str(data), num)
                    num += 1
                else:#if word is in dictionary :
                    in_list[dictionary.get(str(data))].append(number)
        return dictionary,in_list

    def _pre_processing(self):
        #output : list_indext(cell i include genres of i'th film & overview of i'th film) , text(cell i include i'th film overview)
        list_index, text = self._text_index_of_data(self.train_df)
        #output : dictionary (dictionary of all word),in_list(index of film for each word)
        dictionary,in_list = self._dictionary_whit_index_list(text)
        #output : dictionary of genres
        genres = self._get_id_and_genres(list_index)
        x_train, y_train = self._train_usefull_data(in_list, genres,list_index, dictionary)

        list_index_test, text_test = self._text_index_of_data(self.test_df)
        x_test, y_test = self._test_usefull_data(list_index_test, genres,len(list_index), dictionary, in_list)

        with open("data/phase2/x_train.pickle", "wb") as train:
            pickle.dump(x_train, train)
        with open("data/phase2/y_train.pickle", "wb") as train:
            pickle.dump(y_train, train)
        with open("data/phase2/x_test.pickle", "wb") as test:
            pickle.dump(x_test, test)
        with open("data/phase2/y_test", "wb") as test:
            pickle.dump(y_test, test)

    def run(self):
        self._preprocessing()
        x_train, y_train, x_test, y_test = self._load()
        Classifier(x_train, y_train, x_test, y_test).run()

# main
if __name__ == "__main__":

    Cbow(pd.read_csv('dataset/test.csv', encoding='windows-1252'),pd.read_csv('dataset/train.csv', encoding='windows-1252')).run()
