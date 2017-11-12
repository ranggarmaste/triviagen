file = open('titles.txt', 'r', encoding='utf8')
lines = file.readlines()

data = []
import wikipedia

for i, line in enumerate(lines):
    try:
        page = wikipedia.page(line)
        article = {}
        article['summary'] = page.summary
        article['content'] = page.content
        article['title'] = line
        data.append(article)
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
        print('Error:', line)
    if i % 100 == 0:
        print(i)

out = open('data.tsv', 'a', encoding='utf8')
for article in data:
    out.write(article['title'].replace('\n', ''))
    out.write('\t')
    out.write(article['summary'].replace('\n', ' '))
    out.write('\t')
    out.write(article['content'].replace('\n', ' '))
    out.write('\n')
out.close()
file.close()

import pandas as pd
data_pd = pd.read_csv('data.tsv', sep='\t')
