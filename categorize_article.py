#%%

from nltk.corpus import stopwords
stopword_set = set(stopwords.words('english'))

#%%

from gensim import corpora, models

dictionary_path = 'dictionary.model'
lda_path = 'lda.model'

dictionary = corpora.Dictionary.load(dictionary_path)
lda_model = models.LdaModel.load(lda_path)

#%%

import re
import numpy as np

def get_article_topic(article):
    global dictionary, lda_model, stopword_set
    temp = article.lower()

    words = re.findall(r'\w+', temp, flags = re.UNICODE)

    important_words = []
    important_words = filter(lambda x: x not in stopword_set, words)

    ques_vec = []
    ques_vec = dictionary.doc2bow(important_words)

    topic_vec = []
    topic_vec = lda_model[ques_vec]

    word_count_array = np.empty((len(topic_vec), 2), dtype = np.object)
    for i in range(len(topic_vec)):
        word_count_array[i, 0] = topic_vec[i][0]
        word_count_array[i, 1] = topic_vec[i][1]

    idx = np.argsort(word_count_array[:, 1])
    idx = idx[::-1]
    word_count_array = word_count_array[idx]
    return word_count_array[0:3, 0]

#%%

import pandas as pd
data = pd.read_csv('data_test.tsv', sep='\t', encoding='utf8')

#%%

topic_category_file = open('topic_category.txt', 'r', encoding='utf8')
lines = topic_category_file.readlines()
topic_dict = {}
for line in lines:
    splits = line.split('\t')
    topic_dict[int(splits[0])] = splits[1]

#%%

def topics_to_categories(topics):
    global topic_dict
    categories = []

    for topic in topics:
        categories.append(topic_dict[topic])
    return categories

#%%

topics = []
for article in data['Content']:
    topics.append(topics_to_categories(get_article_topic(article)))

#%%
    
for i in range(len(topics)):
    for j in range(len(topics[0])):
        topics[i][j] = topics[i][j].replace('\n', '')

count = 0
count_3 = 0
categories_true = data.iloc[:, 3]
for i in range(len(topics)):
    if categories_true[i].replace(' ', '') == topics[i][0]:
        count += 1
    else:
        print(len(topics[i][0]), len(categories_true[i]))
    if categories_true[i].replace(' ', '') in topics[i]:
        count_3 += 1