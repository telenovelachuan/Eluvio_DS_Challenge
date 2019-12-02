import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def load_raw_data(client):
    df = dd.read_csv("../data/raw/Eluvio_DS_Challenge.csv", blocksize=64000000)
    df.time_created = df.time_created.astype('uint32')
    df.up_votes = df.up_votes.astype('uint16')
    df.down_votes = df.down_votes.astype('uint16')
    df["date_created"] = df["date_created"].map_partitions(pd.to_datetime, format='%Y/%m/%d', meta=('datetime64[ns]'))

    df = client.persist(df)
    print df.dtypes
    return df


client = Client(processes=False)
df = load_raw_data(client)


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


def plot_top_n_words(top_words, title=None, rotation=45):
    count = len(top_words)
    fig, ax = plt.subplots(figsize=(16, 8))
    words = [x[0] for x in top_words]
    ax.bar(range(count), [x[1] for x in top_words])
    ax.set_xticks(range(count))
    ax.set_xticklabels(words, rotation='vertical')
    chart_title = title or 'Top {} words in title (excluding stop words)'.format(count)
    ax.set_title(chart_title)
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=rotation)
    plt.show()


plot_top_n_words(top_N_words, rotation=0)

print "number of titles by year"
plt.figure(figsize=(10, 5))
yearly_num = df['date_created'].dt.year.value_counts().compute()

ax = sns.lineplot(x=yearly_num.index.values, y=yearly_num.values)
ax = plt.title('Number of titles by year')

print "number of titles by month"
years = df['date_created'].dt.year.unique().compute()
plt.figure(figsize=(18,7))
for year in years:
    df_year = df[df['date_created'].dt.year == year]
    monthly_nums = df_year['date_created'].dt.month.value_counts().compute()
    _ax = sns.lineplot(x=monthly_nums.index.values,y=monthly_nums.values, legend='full', label=year)

plt.title('Number of titles by month')

print "number of titles by day"
daily_num = df['date_created'].dt.day.value_counts().compute()
plt.figure(figsize=(10,5))
_ax = sns.lineplot(x=daily_num.index.values,y=daily_num.values)
plt.title('Number of titles by day')


def plot_wordcloud(words, title):
    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)
    plt.figure(figsize=(20,20))
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off")
    plt.title(title, fontsize=60)
    plt.show()


df_2008 = df[df['date_created'].dt.year == 2008]
all_text = " ".join(df_2008.title.compute())
plot_wordcloud(all_text, "2008 title words")

print "What's happening on China in 2008?"
df_china = df_2008[df_2008.title.str.match("China", case=False)]
top_11_words = get_top_n_words(df_china.title.compute(), n=11)
print top_11_words[1:]
plot_top_n_words(top_11_words[1:], title="Top 10 words on China in 2008", rotation=0)

df_2016 = df[df['date_created'].dt.year == 2016]
all_text = " ".join(df_2016.title.compute())
plot_wordcloud(all_text, "2016 title words")

print "What's happening on China in 2016?"
df_china=df_2016[df_2016.title.str.match("China", case=False)]
top_11_words = get_top_n_words(df_china.title.compute(), n=11)
print top_11_words[1:]
plot_top_n_words(top_11_words[1:], title="Top 10 words on China in 2016", rotation=0)

topic = "global warming"
title = "mentions of \"{}\" over time".format(topic)
print title
df_topic=df[df.title.str.match(topic, case=False)]

plt.figure(figsize=(10,5))
yearly_num = df_topic['date_created'].dt.year.value_counts().compute()
ax = sns.lineplot(x=yearly_num.index.values,y=yearly_num.values)
ax = plt.title(title)



