import re
import string
import numpy as np
import os
# import unidecode
import itertools
from nltk import ngrams
from tqdm import tqdm
NGRAM = 5

def open_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
    return lines
    # print(len(lines))

# def removed_accent(text):
    # print(unidecode.unidecode(text))
    # return unidecode.unidecode(text)
    # pass
# removed_accent('Tổ chức này chú trọng đến: tiêu chuẩn, giáo dục và các vấn đề về chính sách')

def extract_phrase(text):
    # print(re.findall(r'\w[\w ]+', text))
    return re.findall(r'\w[\w ]+', text)
# extract_phrase('Tổ chức này chú trọng đến: tiêu chuẩn, giáo dục và các vấn đề về chính sách')
# open_file('train_data.txt')

def get_phrases():
    lines = open_file('train_data.txt')
    phrases = itertools.chain.from_iterable(extract_phrase(text) for text in lines)
    phrases = [p.strip() for p in phrases if len(p.strip()) > 1]
    return phrases

def gen_ngrams(words, n = 5):
    # print (ngrams(words.split(' '), n))
    return ngrams(words.split(' '), n)

def list_ngrams():
    list_ngrams = []
    phrases = get_phrases()
    # phrases = ["chúng ta sẽ làm việc trong hôm nay"]
    for phrase in tqdm(phrases):
        for ngrams in gen_ngrams(phrase, NGRAM):
            if(len(" ".join(ngrams)) < 30):
                list_ngrams.append(" ".join(ngrams))
    del phrases
    list_ngrams = list(set(list_ngrams))
    with open('list_ngrams.txt', 'w') as f_w:
        for list_ngram in list_ngrams:
            f_w.write(list_ngram + '\n')
    # print(list_ngrams[1])
    # return list_ngrams

list_ngrams()


# with open('list_ngrams.txt', 'a') as f_w:

# list_ngrams()
# gen_ngrams("Tổ chức này chú trọng đến: tiêu chuẩn, giáo dục và các vấn đề về chính sách")