import numpy as np
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import pickle
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn import preprocessing
from gensim.models.fasttext import FastText


def wordTransformer(str):
    return ViTokenizer.tokenize(removeChar(str) ).split(' ')

def removeChar(string):
    ret = ""
    for i in string:
        if (
                i != ',' and i != '.' and i != '!' and i != '=' and i != '#' and i != ';' and i != '%' and i != '?' and i != '"' and i != '\n'):
            ret = ret + i
    return ret

def stringConvert(arr):
    ret = ""
    for i in range(len(arr)):
        if len(ret) == 0:

            ret = ret + arr[i]
        else:
            ret = ret + " " + arr[i]
    return removeChar( ret)

def readFileText(url):
    myFile = open(url)
    ret = []
    label = []
    for i in myFile:
        string  = i.split(' ')
        label.append(string[0])
        tmp = string[1:len(string)]
        ret.append(stringConvert(tmp))
    return ret,label


def extraFeature(data,model):
    omega = []
    for row in data:
        # print( CvD.wordTransformer( CvD.removeChar( row.lower())))
        # print(row)
        omega.append(row.split(' '))

    ret = []
    for row in omega:
        h = model.wv[row]
        R, C = h.shape
        tmp = []
        for i in range(C):
            S = 0
            collum = h[:, i]
            for j in collum:
                S = S + j
            tmp.append(S / R)
        ret.append(np.array(tmp))
    return ret

def splitLabel(str):
    return int( float (str[9:len(str)] ))


def convert_layble(data):
    ret = []
    for row in data:
        ret.append(splitLabel(row))
    ret = np.array(ret)
    return ret

def word_count(data):
    counts = dict()
    for i in data:
        words = i.split(' ')
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts

def choose_words(data):
    C = word_count(data)
    arr = []
    for i in C:
        tmp = []
        tmp.append(i)
        tmp.append(C[i])
        arr.append(tmp)

    arr.sort(key=sortSecond)

    i = 0
    Return = []

    while i < 25000:
        Return.append(arr[i][0])
        i = i + 1
    return Return

def changeText(X):
    arr= choose_words(X)
    #print(arr)
    Return = []
    for i in X:
        words = i.split(' ')
        tmp = ''
        for word in words:
            if word in arr:
                tmp = tmp + ' '+ word
            #print(word)
        print(tmp)
        Return.append(tmp)
    return Return
def sortSecond(val):
    return -val[1]

def tf_idf(X,data2):
    print("trainning tf-idf")
    XX = []
    for i in X:
        XX.append(i)
    for i in data2:
        XX.append(i)
    #XX = changeText(X)
    tf_idf_vector = TfidfVectorizer()
    tf_idf_vector.fit(XX)
    dataArray = tf_idf_vector.transform(X)
    pickle.dump(tf_idf_vector, open('tf_idf_model', 'wb'))
    return dataArray.toarray()

def split_data():
    print("loading data ...")
    data, label = readFileText("./data/train.txt")
    data2,laybel2 = readFileText("./data/test.txt")

    print("load data done")
    # data = data[:10000]
    # label = label[:10000]
    #data = changeText(data)
    model = Word2Vec.load("word2vec.model")

    data = tf_idf(data,data2)

    print("save done")

    label = convert_layble(label)
    print(label[0])
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data)
    data = min_max_scaler.transform(data)

    print(data[0])
    print(label[0])
    pickle.dump(encoder, open('encoder', 'wb'))
    pickle.dump(min_max_scaler, open('min_max_scaler', 'wb'))

    print('return ')
    return data,label

if __name__ == '__main__':
#    X_train, X_validation, y_train, y_validation =  split_data()
    X_train,y_train =  split_data()
    print('feature done')
    count = [0,0,0,0,0,0,0]
    for i in y_train:
       count[i] = count[i] + 1
    print(count)

    print('begin trainning ')
    params_grid = [
    
        {'C': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

    svm_model = LinearSVC(C=0.5)
    #svm_model = GridSearchCV(LinearSVC(), params_grid, cv=5)
    svm_model.fit(X_train, y_train)
    print('trainning done')
    my_file = open('ressult','w')
    #print('Best score for training data:', svm_model.best_score_, "\n")
    # my_file.write('Best score for training data: ', str(svm_model.best_score_), "\n")
    # # View the best parameters for the model found using grid search
    #print('Best C:', svm_model.best_estimator_.C, "\n")
    # my_file.write('Best C: ', str(svm_model.best_estimator_.C ), "\n")
    #print('Best Kernel:', svm_model.best_estimator_.kernel, "\n")
    # my_file.write('Best Kernel:', str(svm_model.best_estimator_.kernel), "\n")
    # print('Best Gamma:', svm_model.best_estimator_.gamma, "\n")
    # my_file.write('Best Gamma: ', svm_model.best_estimator_.gamma, "\n")
    # encoder = preprocessing.LabelEncoder()
    #final_model = svm_model.best_estimator_

    # Y_pred = svm_model.predict(X_validation)
    # print(confusion_matrix(y_validation, Y_pred))
    # my_file.write(str( confusion_matrix(y_validation, Y_pred) ))
    # print("\n")
    # my_file.write("\n")
    pickle.dump(svm_model, open("linearsvm_model", 'wb'))
    # print(classification_report(y_validation, Y_pred))
    # my_file.write(str(classification_report(y_validation, Y_pred)))
    print("Training set score for SVM: %f" % svm_model.score(X_train, y_train))
    # my_file.write("Training set score for SVM: %f" % svm_model.score(X_train, y_train))
    # print("Testing  set score for SVM: %f" % svm_model.score(X_validation, y_validation))
    # my_file.write("Testing  set score for SVM: %f" % svm_model.score(X_validation, y_validation))
    #
    # svm_model.score




