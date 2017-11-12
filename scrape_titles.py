import requests
import json

categories = ['Art', 'Culture', 'Events', 'Games', 'Geography', 'Health', 'History', 'Humanities', 'Law', 'Life', 'Mathematics', 'Matter', 'Nature', 'People', 'Philosopy', 'Politics', 'Reference works', 'Religion', 'Science and tehcnology', 'Society', 'Sports', 'World']
articles = []
for i in range(200):
    r = requests.get('https://en.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=max')
    data = json.loads(r.text)
    articles = articles + [x['title'] for x in data['query']['random']]
    print(i)
    
articles_set = set(articles)
print('Titles retrieved:', len(articles_set))

titles = open('titles.txt', 'w', encoding='utf8')
for article in articles_set:
    titles.write(article)
    titles.write('\n')
titles.close()