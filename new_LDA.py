import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import pyLDAvis.gensim_models
import re
from util import Read_file_list,combine2gram,combine3gram,remove_stopwords,test_topic
from collections import Counter
import jieba
from collections import defaultdict
from gensim.models import LdaModel
import pandas as pd
from gensim.corpora import Dictionary
from gensim import corpora, models
import csv
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import random

path_list = Read_file_list(r".\txt")
#获取超过500字的段落
para_list = []
para_label = []
id = 1
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        for l in text:
            if len(l)>=500:
                para_list.append(l)
                para_label.append(id)
        id = id + 1
#print(para_label)
#print(len(para_label))

# 均匀抽取200个段落
text_ls = []
text_label = []
random_indices = random.sample(range(len(para_list)), 200)
text_ls.extend([para_list[i] for i in random_indices])
text_label.extend([para_label[i] for i in random_indices])
text_ls = remove_stopwords(text_ls)

def print_formatted_topics(lda_model, num_words=10, num_decimals=4):
    for idx, topic in lda_model.print_topics(num_words=num_words):
        print("Topic #{}:".format(idx))
        words = topic.split("+")
        for word in words:
            word = word.strip()
            word_prob = float(word.split("*")[0])
            word_prob_formatted = format(word_prob, '.{}f'.format(num_decimals))
            word_text = word.split("*")[1]
            print("{} ({})".format(word_text, word_prob_formatted))
        print("\n")

if __name__ == "__main__":
    # 分词，分别以字和词为基本单位
    tokens_word = []  # 以词文单位
    tokens_word_label = []

    for i, text in enumerate(text_ls):
            words = [word for word in jieba.lcut(sentence=text)]
            token_2gram = []
            token_2gram += combine2gram(words)
            tokens_word.append(token_2gram)
            tokens_word_label.append(text_label[i])
    # 构造词典,为文档中的每个词分配一个独一无二的整数编号
    dictionary_word = gensim.corpora.Dictionary(tokens_word)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus_word = [dictionary_word.doc2bow(tokens) for tokens in tokens_word]

    # 训练lda模型，num_topics设置主题的个数
    results_perplexity_word = []
    results_cv_word = []
    results_perplexity_char = []
    results_cv_char = []
    num_topics_list = range(12, 13, 1)
    for num_topics in num_topics_list:
        # 以“词”作为基本单元
        lda_word = gensim.models.ldamodel.LdaModel(corpus=corpus_word, id2word=dictionary_word, num_topics=num_topics,
                                                   passes=30, alpha='auto', eta='auto')

        #print(lda_word.print_topics(num_topics=num_topics, num_words=100, formatted=True, format='%.5f'))
        print_formatted_topics(lda_word, num_words=10, num_decimals= 5)

        perplexity_word = -lda_word.log_perplexity(corpus_word)
        cv_model_word = gensim.models.CoherenceModel(model=lda_word, texts=tokens_word, dictionary=dictionary_word,
                                                     coherence='c_v')   # 一致性
        results_perplexity_word.append(perplexity_word)
        results_cv_word.append(cv_model_word.get_coherence())


        for i in range(20):
            test_ls = test_topic(para_list,para_label)
            doc_topics = lda_word.get_document_topics(test_ls)
            for j in doc_topics:
                print(j)




    '''
    print("results_perplexity_word : " + str(results_perplexity_word))
    print("results_cv_word : " +str(results_cv_word))

    # 创建画布
    fig, axes = plt.subplots(nrows=1, ncols=1)

    # 在第一个小区域中绘制第一条曲线
    plt.plot(num_topics_list, results_perplexity_word, label='word')
    plt.title('perplexity')
    plt.legend()
    plt.show()

    # 在第二个小区域中绘制第二条曲线
    fig, axes = plt.subplots(nrows=1, ncols=1)
    plt.plot(num_topics_list, results_cv_word, label='word')
    plt.title('coherence')
    plt.legend()
    plt.show()


    # 在第三个小区域中绘制第三条曲线
    fig, axes = plt.subplots(nrows=1, ncols=1)
    plt.plot(num_topics_list, results_cv_word, label='coherence')
    plt.title('word')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=1)
    plt.plot(num_topics_list, results_perplexity_word, label='perplexity')
    plt.title('word')
    plt.legend()
    plt.show()
    '''
