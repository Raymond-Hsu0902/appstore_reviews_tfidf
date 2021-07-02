# -*- coding: UTF-8 -*-
import pickle
import jieba

#停用词列表
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
 
#加载停用词
stopwords = stopwordslist("chineseStopWords.txt")
jieba.load_userdict('userDict.txt')

# 載入Model
with open('app/model/tfidf_model.pkl','rb') as f: 
    tfidfModel = pickle.load(f)
    model = tfidfModel['nb'] # 通过字典获取模型
    categories_labal_dict = tfidfModel['categories_label'] # 获取类别名称-编码关系
    tfidf = tfidfModel['tfidfVectorizer'] # 获取文本特征向量转换器

def get_features(x):
    return tfidf.transform(x)

def predict(text):
    words = [word for word in jieba.lcut(text) if len(word)>=2 and word not in stopwords]
    print('words = ',words)
    data = " ".join(words)
    feat = get_features([data])# 使用加载模型后的tfidf 文本提取器
    pred = model.predict(feat)[0] # 使用加载后的model
    return pred