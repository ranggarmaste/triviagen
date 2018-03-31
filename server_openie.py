#%%
import json
import requests
from flask import Flask
from flask import request
app = Flask(__name__)

#%%
@app.route('/openie', methods=['POST'])
def process_article():
    data = request.json
    
    r = requests.post('http://localhost:9000', data['content'].encode('utf-8'))
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

    resp = {}
    resp['title'] = data['title']
    resp['openie_relations'] = splits
    return json.dumps(resp)
