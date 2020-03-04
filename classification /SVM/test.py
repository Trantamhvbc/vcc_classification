from gensim.models import Word2Vec
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report




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

def loadmodel():
    print('loading model ')
    model_w2v = Word2Vec.load("word2vec.model")
    model_svm = pickle.load(open('linearsvm_model', 'rb'))
    min_max_scaler = pickle.load(open('min_max_scaler', 'rb'))
    encoder = pickle.load(open('encoder', 'rb'))
    return model_svm,model_w2v,min_max_scaler,encoder



def convert_layble(data):
    ret = []
    for row in data:
        ret.append(splitLabel(row))
    ret = np.array(ret)
    return ret
def splitLabel(str):
    return int( float (str[9:len(str)] ))

def extraFeature(data,model):
    omega = []
    for row in data:
        # print( CvD.wordTransformer( CvD.removeChar( row.lower())))
        # print(row)
        t = []
        words = row.split(' ')
        for word in words:
            if word in model:
                t.append(word)
        print(t)
        omega.append(t)
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
def tf_idf(X):
    print("test tf-idf")
    tf_idf_vector = pickle.load(open('tf_idf_model','rb'))
    dataArray = tf_idf_vector.transform(X)
    return dataArray.toarray()

def split_data(model_w2v, min_max_scaler,encoder):
    data, label = readFileText("./data/test.txt")
#    data = data[:3000]
#    label = label[:3000]
    data = tf_idf(data)
    #data = min_max_scaler.transform(data)
    label = convert_layble(label)
    label = encoder.transform(label)
    return data,label



if __name__ =='__main__':

    model_svm, model_w2v, min_max_scaler,encoder = loadmodel()
    data,label = split_data(model_w2v,min_max_scaler,encoder)
    print('load done')
    Y_pred = model_svm.predict(data)

    print(confusion_matrix(label, Y_pred))
    print(classification_report(label, Y_pred))
    print("Test set score for SVM: %f" % model_svm.score(data, label))
    print("save done")
