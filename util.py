import os
import re
from collections import Counter
import random
import jieba
import gensim

def Read_file_list(dict_name):
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name)
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)
        except :
            result_txt.append(temp_name)
    return result_txt

def combine2gram(cutword_list):
    if len(cutword_list) == 1:
        return []
    res = []
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i]  + cutword_list[i+1])#+ " "
    return res

def combine3gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        res.append(cutword_list[i] + cutword_list[i+1] + " " + cutword_list[i+2] )
    return res

def remove_stopwords(text):
    with open("cn_stopwords.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    for j in range(len(text)):
        for line in lines:
            text[j] = text[j].replace(line, "")
            text[j] = text[j].replace(" ", "")
    regex_str = ".*?([^\u4E00-\u9FA5]).*?"
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】〖〗《》？“”‘’！[\\]^_`{|}~]+'
    symbol = []
    for j in range(len(text)):
        text[j] = re.sub(english, "", text[j])
        symbol += re.findall(regex_str, text[j])
    count_ = Counter(symbol)
    count_symbol = count_.most_common()
    noise_symbol = []
    for eve_tuple in count_symbol:
        if eve_tuple[1] < 200:
            noise_symbol.append(eve_tuple[0])
    noise_number = 0
    for line in text:
        for noise in noise_symbol:
            line = line.replace(noise, "")
            noise_number += 1
    return text

def test_topic(para_list,para_label):
    test_ls = []
    test_label = []
    random_indices = random.sample(range(len(para_list)), 1)
    test_ls.extend([para_list[i] for i in random_indices])
    test_label.extend([para_label[i] for i in random_indices])
    test_tokens_word = []
    test_tokens_word_label = []

    for i, text in enumerate(test_ls):
            words = [word for word in jieba.lcut(sentence=text)]
            token_2gram = []
            token_2gram += combine2gram(words)
            test_tokens_word.append(token_2gram)
            test_tokens_word_label.append(test_label[i])

    # 构造词典,为文档中的每个词分配一个独一无二的整数编号
    dictionary_word = gensim.corpora.Dictionary(test_tokens_word)
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus_word = [dictionary_word.doc2bow(tokens) for tokens in test_tokens_word]
    #print(corpus_word)

    return corpus_word
