#%%

import numpy as np
import pandas as pd

data = pd.read_csv('data_test.tsv', sep='\t')
#%%

import requests

extracted = {}

for j in range(len(data)):
    r = requests.post('http://localhost:9000', data=data.iloc[j, 1].encode('utf-8'))
    lines = r.text.split('\n')
    lines.pop()
    for i in range(len(lines)):
        lines[i] = lines[i][5:]
        if 'Context' in lines[i]:
            lines[i] = lines[i].split(':')[1]
        lines[i] = lines[i][1:]
        lines[i] = lines[i][:-1]
        lines[i] = lines[i].replace('T:', '')
        lines[i] = lines[i].replace('L:', '')
        
    splits = []
    for i in range(len(lines)):
        splits.append(lines[i].split('; '))
        relation = splits[i][1:len(splits[i])-1]
        relation = ' '.join(relation)
        splits[i] = [splits[i][0], relation, splits[i][len(splits[i])-1]]
    extracted[data.iloc[j, 0]] = splits
    print(j)


#%%

id_map = {}
for i in range(len(data)):
    id_map[data.iloc[i, 0]] = i+1

title_data = {'id': [x+1 for x in range(len(data))], 'article': data.iloc[:, 0].tolist(), 'category': data.iloc[:, 3].tolist()}
trivia_data = {'id_article': [], 'article': [], 'entity1': [], 'relation': [], 'entity2': []}

for k, v in extracted.items():
    for t in v:
        if t[2] != '':
            trivia_data['id_article'].append(id_map[k])
            trivia_data['article'].append(k)
            trivia_data['entity1'].append(t[0])
            trivia_data['relation'].append(t[1])
            trivia_data['entity2'].append(t[2])
            
#%%

title_df = pd.DataFrame(data=title_data)
title_df = title_df[['id', 'article', 'category']]
trivia_df = pd.DataFrame(data=trivia_data)
trivia_df = trivia_df[['id_article', 'article', 'entity1', 'relation', 'entity2']]

#%%

title_df.to_csv('article.tsv', sep='\t', index=False)
trivia_df.to_csv('openie.tsv', sep='\t', index=False)