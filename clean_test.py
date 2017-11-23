#%%
import pandas as pd

data = pd.read_csv('data_test.tsv', sep='\t')

#%%
for i in range(len(data)):
    data.iloc[i, 2] = data.iloc[i, 2].replace('==', '')
    
data.to_csv('data_test.tsv', sep='\t', index=False)

#%%

from nltk.tokenize import wordpunct_tokenize

for i in range(len(data)):
    title = data.iloc[i, 0]
    tokens = wordpunct_tokenize(data.iloc[i, 2])
    for j in range(len(tokens)):
        tokens[j] = title if tokens[j] == 'He' else tokens[j]
        tokens[j] = title if tokens[j] == 'he' else tokens[j]
        tokens[j] = title if tokens[j] == 'He' else tokens[j]
        tokens[j] = title if tokens[j] == 'she' else tokens[j]
        tokens[j] = title if tokens[j] == 'She' else tokens[j]
        tokens[j] = title if tokens[j] == 'it' else tokens[j]
        tokens[j] = title if tokens[j] == 'It' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'his' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'His' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'her' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'Her' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'its' else tokens[j]
        tokens[j] = title + "'s" if tokens[j] == 'Its' else tokens[j]
        tokens[j] = title if tokens[j] == 'him' else tokens[j]
        tokens[j] = title if tokens[j] == 'Him' else tokens[j]
    data.iloc[i, 2] = ' '.join(tokens)
    print (i)
    
data.to_csv('data_test_coreference.tsv', sep='\t', index=False)

#%%
# Iseng
summary = pd.read_csv('mitie_summary.tsv', sep='\t')