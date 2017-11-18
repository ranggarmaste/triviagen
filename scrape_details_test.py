#%%
file = open('test_list.txt', 'r', encoding='utf8')
lines = file.readlines()

#%%
import wikipedia
data = []

for i in range(len(lines)):
    if i % 11 == 0:
        current_category = lines[i]
        print(current_category)
    else:
        try:
            page = wikipedia.page(lines[i])
            article = {}
            article['summary'] = page.summary
            article['content'] = page.content
            article['title'] = lines[i]
            article['category'] = current_category
            data.append(article)
        except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
            print('Error:', lines[i])
            
#%%
out = open('data_test.tsv', 'w', encoding='utf8')
for article in data:
    out.write(article['title'].replace('\n', ''))
    out.write('\t')
    out.write(article['summary'].replace('\n', ' '))
    out.write('\t')
    out.write(article['content'].replace('\n', ' '))
    out.write('\t')
    out.write(article['category'].replace('\n', ' '))
    out.write('\n')
out.close()
file.close()

#%%
import pandas as pd
data_pd = pd.read_csv('data_test.tsv', sep='\t')
