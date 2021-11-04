from collections import Counter
import ast
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from GeneralClassifier import Classifier
import spacy
from sklearn.cluster import KMeans
from keras import Sequential
import math
import pandas as pd


class improve_algorithm:

    def __init__(self,test, train):
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.max_length = 2000000
        self.train_df = train
        self.test_df = test
    
    def _dictionary_whit_index_list(self, band):
        in_list = []
        dictionary = {}
        num = 0
        for number, text in enumerate(band):
            datum = self.nlp(text)
            for data in datum:
                if str(data) not in dictionary:
                    in_list.append([number])
                    dictionary.setdefault(str(data), num)
                    num += 1
                else:
                    in_list[dictionary.get(str(data))].append(number)
        return dictionary,in_list
    
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
    def _train_usefull_data(self, list_index, dictionary, l1, genres):
        x_train = []
        y_train = []
        for number, text in enumerate(list_index):#number = number of film , text[0] = genres & id , text[1] = overview
            temp = [0] * len(dictionary.keys()) #length of temp = length of dictionary
            for word in self.nlp(text[1]):#for each word in overview
                temp[int(dictionary.get(str(word)))] = self._impotance_whit_tfidf(dictionary, l1, str(word), number, list_index) #temp[word] = tfidf
            x_train.append(temp.copy()) #temp is representation of film's overview with verctor, that each word's value is tfpdf
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
        
        tmp = [0] * self.nlp("help")[0].vector
        output = []
        for _ in range(len(x_train)):
            output.append(tmp.copy())

        for index1 in range(len(dictionary)):
            vector = self.nlp(list(dictionary.keys())[index1])[0].vector
            summ = 0
            x = 0
            for index2, row in enumerate(x_train):
                x = index2
                if row[index1] == 0:
                    continue
                output[index2] += vector * row[index1]
                summ += row[index1]
            if summ == 0:
                summ = 1
            output[x] /= summ
        return output, y_train


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
            x_test.append(temp.copy()) #temp is representation of film's overview with verctor, that each word's value is tfidf
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
        
        tmp = [0] * self.nlp("help")[0].vector
        output = []
        for _ in range(len(x_test)):
            output.append(tmp.copy())

        for index1 in range(len(dictionary)):
            vector = self.nlp(list(dictionary.keys())[index1])[0].vector
            summ = 0
            x = 0
            for index2, row in enumerate(x_test):
                x = index2
                if row[index1] == 0:
                    continue
                output[index2] += vector * row[index1]
                summ += row[index1]
            if summ == 0:
                summ = 1
            output[x] /= summ
        return output, y_test

    def _text_index_of_data(self, df):
        list_index = []
        text = []
        for id, band in df.iterrows():
            text.append(band['overview'])
            list_index.append([ast.literal_eval(band['genres']), band['overview']])
        return list_index, text

    def _get_id_and_genres(self, list_vector):
        id_name = {}
        for list_data in list_vector:
            for data in list_data[0]:
                id_name.setdefault(data['id'], data['name'])
        return id_name

    def _pre_processing(self):
        train_list_index, train_text = self._text_index_of_data(self.train_df)
        dictionary, in_list = self._dictionary_whit_index_list(train_text)
        genres = self._get_id_and_genres(train_list_index)
        x_train, y_train = self._train_usefull_data(train_list_index, dictionary, in_list, genres)
      
        test_list_index, test_text = self._text_index_of_data(self.test_df)
        x_test, y_test = self._test_usefull_data(test_list_index,genres,len(train_list_index), dictionary, in_list)
        return x_train, y_train, x_test, y_test

    def run(self):
        x_train, y_train, x_test, y_test = self._preProcessing()
        Classifier(x_train, y_train, x_test, y_test).run()

if __name__ == "__main__":

    improve_algorithm(pd.read_csv('dataset/test.csv', encoding='windows-1252'),pd.read_csv('dataset/train.csv', encoding='windows-1252')).run()