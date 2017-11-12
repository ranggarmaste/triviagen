# Initialization
import pandas as pd
import numpy as np

data = pd.read_csv('data.tsv', sep='\t')
data = data.dropna()

summaries = data.iloc[:, 2].values

# Pre-processing
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stopword_set = set(stopwords.words('english'))
summaries_edited = []
for i, summary in enumerate(summaries):
    summary = re.sub('[^a-zA-Z]', ' ', summary)
    summary = summary.lower()
    summary = summary.split()
    summary = [x for x in summary if x not in stopword_set]
    summary = ' '.join([lemmatizer.lemmatize(x) for x in summary])
    summaries_edited.append(summary)
    
# Create TF-IDF matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000000, use_idf=True, ngram_range=(1,2))
tfidf_matrix = tfidf_vectorizer.fit_transform(summaries_edited)

# LDA
from gensim import corpora, models 
dictionary = corpora.Dictionary([x.split() for x in summaries_edited])

num_topics = n_ideal = 25
corpus = [dictionary.doc2bow(text) for text in [x.split() for x in summaries_edited]]
lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=0)

doc_topic = list() 
for doc in corpus: 
    doc_topic.append(lda.__getitem__(doc, eps=0))
    
for i, d in enumerate(doc_topic):
    for j, d_t in enumerate(d):
        doc_topic[i][j] = d_t[1]
doc_topic = np.array(doc_topic)

# If necessary: Cleaning
threshold = 0.5
# indices = list(range(len(doc_topic)))
indices = np.amax(doc_topic, axis=1) > threshold
doc_topic = doc_topic[indices]
clusters = doc_topic.argmax(1)

# Create new data frame
data_new_values = { 'Title': data['Title'].values[indices], 'Summary': data['Summary'].values[indices], 'Content': data['Content'].values[indices], 'Cluster': clusters }
data_new = pd.DataFrame(data_new_values)
data_new['Cluster'].value_counts()

# See titles
cluster_title_dict = {}
for article in data_new.iloc[:, :].values:
    if article[0] not in cluster_title_dict:
        cluster_title_dict[article[0]] = []
    cluster_title_dict[article[0]].append(article[3])
    
# Visualization for LDA
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(doc_topic)
xs, ys = tsne_lda[:, 0], tsne_lda[:, 1]

from matplotlib.pyplot import cm 
colors = cm.rainbow(np.linspace(0,1,n_ideal))
cluster_names = {}
for i in range(0, 25):
    terms = [dictionary[x[0]] for x in lda.get_topic_terms(i)[:8]]
    cluster_names[i] = ', '.join(terms)
    
titles = data['Title'].values
titles = titles[indices]
df = pd.DataFrame(dict(x=xs, y=ys, label=doc_topic.argmax(1), title=titles)) 
df = df.sample(frac=0.005, random_state=0)
groups = df.groupby('label')

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

for i in df.index:
   ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=1, prop={'size': 15})
plt.show()

# Save LDA Model
lda.save('lda.model')
lda = models.LdaModel.load('lda.model')