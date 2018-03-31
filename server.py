#%%
import pandas as pd
import numpy as np
import sys, os
import glob

from nltk.corpus import stopwords
from mitie import *
from collections import defaultdict
stopword_set = set(stopwords.words('english'))

#%%

from gensim import corpora, models

dictionary_path = 'dictionary.model'
lda_path = 'lda.model'

dictionary = corpora.Dictionary.load(dictionary_path)
lda_model = models.LdaModel.load(lda_path)
ner = named_entity_extractor('./MITIE-models/english/ner_model.dat')

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

def get_relations(content):
    facts = []
    tokens = tokenize(content)
    entities = ner.extract_entities(tokens)
    rel_classifier_names = glob.glob("./MITIE-models/english/binary_relations/*.svm")
    for rel_classifier_name in rel_classifier_names:
        rel_detector = binary_relation_detector(rel_classifier_name)
        relation_type = rel_classifier_name.split(".")[-2]
        neighboring_entities = [(entities[i][0], entities[i+1][0]) for i in xrange(len(entities)-1)]
        neighboring_entities += [(r,l) for (l,r) in neighboring_entities]
        for first_entity, second_entity in neighboring_entities:
            fact = []
            rel = ner.extract_binary_relation(tokens, first_entity, second_entity)
            score = rel_detector(rel)
            if (score > 0.5):
                first_entity_text     = " ".join(tokens[i].decode("utf-8")  for i in first_entity)
                second_entity_text = " ".join(tokens[i].decode("utf-8")  for i in second_entity)
                fact.append(first_entity_text)
                fact.append(relation_type)
                fact.append(second_entity_text)
                facts.append(fact)
    return facts

#%%

def get_article_topics(article):
    return topics_to_categories(get_article_topic(article))

#%%

import json
from flask import Flask
from flask import request
app = Flask(__name__)

@app.route('/article', methods=['GET'])
def hello_world():
    return "hello, world!"

@app.route('/article', methods=['POST'])
def process_article():
    data = request.json

    resp = {}
    resp['title'] = data['title']
    resp['topics'] = get_article_topics(data['content'])
    resp['mitie_relations'] = get_relations(data['content'])
    return json.dumps(resp)
