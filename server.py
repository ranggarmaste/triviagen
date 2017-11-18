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
topic_category_file = open('topic_category.txt', 'r', encoding='utf8')
lines = topic_category_file.readlines()
topic_dict = {}
for line in lines:
    splits = line.split('\t')
    topic_dict[int(splits[0])] = splits[1][:-1]

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

def topics_to_categories(topics):
    global topic_dict
    categories = []

    for topic in topics:
        categories.append(topic_dict[topic])
    return categories

#%%

def get_article_topics(article):
    return topics_to_categories(get_article_topic(article))

#%%

import json
from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/article', methods=['POST'])
def hello_world():
    data = json.loads(request.data)

    resp = {}
    resp['topics'] = get_article_topics(data['content'])
    return json.dumps(resp)
