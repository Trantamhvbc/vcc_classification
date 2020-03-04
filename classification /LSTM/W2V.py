from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
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

def extraFeature(data):
    omega = []
    for row in data:
        # print( CvD.wordTransformer( CvD.removeChar( row.lower())))
        # print(row)
        omega.append(row.split(' '))
    model = Word2Vec(
        omega, size = 256,
        window=16,
        min_count=1,
        workers=8,
        iter = 1024
    )
    model.save('word2vec.model')

def extraFeature_use_fasttext(train_data):
    model_fasttext = FastText(size=256, window=16, min_count=2, workers=8, sg=1,iter=256)
    model_fasttext.build_vocab(train_data)
    model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
    model_fasttext.save("model_fasttext.bin")
    print('save done')
    # ret = []
    # pad = []
    # for i in range(size):
    #     pad.append(0)
    # for row in train_data:
    #     h = model_fasttext.wv[row]
    #     R, C = h.shape
    #     tmp = []
    #     for i in range(R):
    #         tmp.append(h[i])
    #     i = R
    #     while i < maxlenght:
    #         tmp.append(pad)
    #         i = i + 1
    #
    #     ret.append(np.array(tmp))
    # return np.array(ret)


if __name__ =='__main__':
    data, label = readFileText("train.txt")
    data2, l = readFileText("test.txt")
    for i in data2:
        data.append(i)
    data = extraFeature(data)
    print("save done")
