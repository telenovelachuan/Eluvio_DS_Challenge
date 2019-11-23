import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


client = Client(processes=False)
print client.__repr__

df = dd.read_csv("../data/raw/Eluvio_DS_Challenge.csv", blocksize=64000000)
print df.dtypes
df.time_created = df.time_created.astype('uint32')
df.up_votes = df.up_votes.astype('uint16')
df.down_votes = df.down_votes.astype('uint16')

df = client.persist(df)
print df.dtypes


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


N = 15
titles = df['title']
top_N_words = get_top_n_words(titles.compute(), n=N)
print top_N_words

fig, ax = plt.subplots(figsize=(16, 8))
words = [x[0] for x in top_N_words]
ax.bar(range(N), [x[1] for x in top_N_words])
ax.set_xticks(range(N))
ax.set_xticklabels(words, rotation='vertical')
ax.set_title('Top words in titles (excluding stop words)')
ax.set_xlabel('Word')
ax.set_ylabel('Number of occurences')
plt.show()



