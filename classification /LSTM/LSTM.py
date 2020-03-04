from functools import partial

from pyvi import ViTokenizer, ViPosTagger, ViUtils
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from gensim.models import Word2Vec
from keras.layers import Dense
import  keras as K
from sklearn import preprocessing
from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, LSTM, Bidirectional
import keras
from keras import optimizers
import time
import pickle
from gensim.models.fasttext import FastText
from sympy import yn


def wordTransformer(str):
    return ViTokenizer.tokenize(removeChar(str) ).split(' ')

def removeChar(string):
        ret = ""
        for i in string:
            if (i != ',' and i != '.' and i != '!' and i != '=' and i != '#' and i != ';' and i != '%' and i != '?' and i != '"' and i != '\n'):
                ret = ret + i
        return ret

def stringConvertHaveDictionary(arr,dic):
    ret = ""
    for i in range(len(arr)):
        if len(ret) == 0:
            if arr[i] in dic:
                ret = ret + arr[i]
        else:
            if arr[i] in dic:
                ret = ret + " " + arr[i]

    return removeChar(ret)

def stringConvert(arr):
    ret = ""
    for i in range(len(arr)):
        if len(ret) == 0:

            ret = ret + arr[i]
        else:
            ret = ret + " " + arr[i]
    return removeChar(ret)

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

def splitLabel(str):
    return int( float (str[9:len(str)] ))
def extraFeature_use_fasttext(train_data,maxlenght = 17,size = 128):
    model_fasttext = FastText(size=size, window=10, min_count=2, workers=4, sg=1)
    model_fasttext.build_vocab(train_data)
    model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
    ret = []
    pad = []
    for i in range(size):
        pad.append(0)
    for row in train_data:
        h = model_fasttext.wv[row]
        R, C = h.shape
        tmp = []
        for i in range(R):
            tmp.append(h[i])
        i = R
        while i < maxlenght:
            tmp.append(pad)
            i = i + 1

        ret.append(np.array(tmp))
    return np.array(ret)
def split_data():
    print("loading data ...")
    data, label = readFileText("train.txt")
    #data2,laybel2 = readFileText("./data/test.txt")

    print("load data done")
    # data = data[:10000]
    # label = label[:10000]
    #data = changeText(data)
    model = Word2Vec.load("model_w2v_128.model")

    data = extraFeature(data,model)

    print("save done")

    label = convert_layble(label)
    print(label[0])
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)

    '''min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data)
    data = min_max_scaler.transform(data)

    print(data[0])
    print(label[0])
    pickle.dump(encoder, open('encoder', 'wb'))
    pickle.dump(min_max_scaler, open('min_max_scaler', 'wb'))
    '''
    print('return ')
    return data,label

def extraFeature(data,model):
    omega = []
    s  = set()
    for row in data:
        omega.append( row.split(' '))

    pad = []
    for i in range(256):
        pad.append(0)
    # pad = np.array(pad)
    ret = []
    for row in omega:
        h = model.wv[row]
        R, C = h.shape
        #print(C)
        tmp = []
        for i in range(R):
            tmp.append(h[i])
        i = R
        while  i < 17:
            tmp.append(pad)
            i = i + 1

        ret.append(tmp)
    print('done ')
    return np.array(ret)

def oneHotCoding(data):
    ret = []
    for row in data:
        ret.append(splitLabel(row))
    ret = np.array(ret)
    onehot_encoder = OneHotEncoder(sparse=False)
    ret = ret.reshape(len(ret), 1)
    ret = onehot_encoder.fit_transform(ret)
    return ret

def createModel(input_dim,output_dim):
    model = Sequential()
    model.add(K.layers.Dense(units=32, input_dim=input_dim, activation='relu', use_bias = True))
    #model.add(Dropout(0.2))
    model.add(K.layers.Dense(units=16, activation='relu', use_bias = True))
    #model.add(Dropout(0.2))
    model.add(K.layers.Dense(units = 8, activation='relu', use_bias = True))

    model.add(K.layers.Dense(units=output_dim, activation='softmax', use_bias = False))

    model.get_layer(index=1)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model(n_class, input_dim):
    print('n_class ', n_class )
    model = Sequential()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.0001,beta_1=0.9, beta_2=0.9999, amsgrad=False)  #l0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.add(LSTM(64, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.3))
    model.add(LSTM(32))

    model.add(Dense(n_class, activation="softmax"))

    model.compile(loss='categorical_crossentropy',               # keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("loading data ...")
    x_train,y_train = readFileText("train.txt")
    x_test, y_test = readFileText("test.txt")
    print("load data done")
    # convert = []
    # max  = 0
    # for i in dataset:
    #
    #     convert.append( wordTransformer(i))
    #     t = len( wordTransformer(i) )
    #     if max < t:
    #         max = t
    # print(max)

    print("Feature data raw to vector 17*256 dim")
    model = Word2Vec.load('word2vec.model')
    x_train = extraFeature(x_train,model)
    x_test = extraFeature(x_test, model)


    y_train = oneHotCoding(y_train)
    y_test = oneHotCoding(y_test)
    epochs = 64
    millis1 = int(round(time.time() * 1000))
    model = build_model( input_dim = (x_train.shape[1],x_train.shape[2]) , n_class= y_train[0].shape[0] )
    history = model.fit(x_train,
                        y_train,
                        epochs = epochs,
                        batch_size= 1024,
                        validation_data = (x_test, y_test)
                        )
    millis2 = int(round(time.time() * 1000))
    print("time trainning is ", millis2 - millis1," mili")
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

