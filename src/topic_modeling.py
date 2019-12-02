import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from collections import Counter
import datetime as dt
import pickle

client = Client(processes=False)
df = dd.read_csv("../data/raw/Eluvio_DS_Challenge.csv", blocksize=64000000)
df.time_created = df.time_created.astype('uint32')
df.up_votes = df.up_votes.astype('uint16')
df.down_votes = df.down_votes.astype('uint16')
df["date_created"] = df["date_created"].map_partitions(pd.to_datetime, format='%Y/%m/%d', meta=('datetime64[ns]'))
# df.index = df['date_created']
# df['year'] = df.index.dt.year

df = client.persist(df)
print df.dtypes;
MODEL_TYPES = {
    'LDA': LatentDirichletAllocation,
    'LSA': TruncatedSVD
}
TOPIC_COLOR_MAP = np.array([
    "#1f77b5", "#aec7e9", "#ff7f0f", "#ffbb79", "#2ca02d", "#98df8b", "#d62729", "#ff9897", "#9467be", "#c5b0d6",
    "#8c564c", "#c49c95", "#e377c3", "#f7b6d3", "#7f7f80", "#c7c7c8", "#bcbd23", "#dbdb8e", "#17bed0", "#9edae6"
])
count_vectorizer_limited = CountVectorizer(stop_words='english', max_features=40000)


def keys_to_counts(keys):
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


def get_keys(topic_matrix):
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys


def get_top_n_words(n, n_topics, document_term_matrix, keys, count_vectorizer):
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
        top_word_indices.append(top_n_word_indices)
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
            temp_word_vector[:, index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))
    return top_words


class Topic_Model:
    def __init__(self, input_df, model_type='LDA', max_features=None, sample_rate=0.05, **args):
        self.df = input_df
        self.sample_rate = sample_rate
        self.n_components = args['n_components'] if 'n_components' in args else 0
        self.sampled_df = self.df.sample(frac=sample_rate)
        self.sampled_titles = self.sampled_df['title'].compute().values
        self.sampled_up_votes = self.sampled_df['up_votes'].compute().values
        self.sampled_document_term_matrix = count_vectorizer_limited.fit_transform(self.sampled_titles)

        self.model_type = model_type
        if model_type.upper() not in MODEL_TYPES.keys():
            self.model_type = MODEL_TYPES.keys()[0]
        model_func = MODEL_TYPES[self.model_type]
        print "initialing & fitting {} model...".format(self.model_type)
        self.model = model_func(**args)
        self.topic_matrix = self.model.fit_transform(self.sampled_document_term_matrix)

        self.keys = get_keys(self.topic_matrix)
        self.categories, self.counts = keys_to_counts(self.keys)
        model_filename = "../models/{}_with_sr_{}.model".format(self.model_type, self.sample_rate)
        with open(model_filename, 'wb') as f:
            pickle.dump(self.model, f)
        print "Model initialization done. Model saved to disk."

    def get_top_n_topics(self, count_vectorizer, n=10):
        top_n_words = get_top_n_words(n, self.n_components, self.sampled_document_term_matrix, self.keys, count_vectorizer)
        print "top {} words for each topic by {}:".format(n, self.model_type)
        for idx, topic in enumerate(top_n_words):
            print("Topic {}: ".format(idx + 1), topic)

    def plot_num_title_of_each_topic(self, count_vectorizer):
        top_3_words = get_top_n_words(3, self.n_components, self.sampled_document_term_matrix, self.keys, count_vectorizer)
        print "self.categories:{}".format(self.categories)
        print "top_3_words:{}".format(top_3_words)
        labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in self.categories]
        fig, ax = plt.subplots(figsize=(16, 8))
        print "Frequency of the {} topics by {}:".format(self.n_components, self.model_type)
        ax.bar(self.categories, self.counts)
        ax.set_xticks(self.categories)
        plt.xticks(rotation=45)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Number of titles')
        ax.set_title('{} topic frequency'.format(self.model_type))
        plt.show()

    def _get_mean_topic_vectors(self, keys, two_dim_vectors):
        '''
        returns a list of centroid vectors from each predicted topic category
        '''
        centroid_topic_vectors = []
        for t in range(self.n_components):
            articles_in_that_topic = []
            for i in range(len(keys)):
                if keys[i] == t:
                    articles_in_that_topic.append(two_dim_vectors[i])

            articles_in_that_topic = np.vstack(articles_in_that_topic)
            centroid_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
            centroid_topic_vectors.append(centroid_article_in_that_topic)
        return centroid_topic_vectors

    def visualize_tnse_2_dimension(self, count_vectorizer):
        print "use t-SNE to visualize {} high dimensional dataset".format(self.model_type)
        tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
        tsne_vectors = tsne_model.fit_transform(self.topic_matrix)
        mean_topic_vectors = self._get_mean_topic_vectors(self.keys, tsne_vectors)

        colormap = TOPIC_COLOR_MAP[:self.n_components]
        proportion = 20;
        sampled_up_votes_sizes = self.sampled_up_votes / proportion;
        top_3_words = get_top_n_words(3, self.n_components, self.sampled_document_term_matrix, self.keys, count_vectorizer)
        fig, ax = plt.subplots(figsize=(16, 16))
        plt.scatter(x=tsne_vectors[:, 0], y=tsne_vectors[:, 1],
                    color=colormap[self.keys],
                    marker='o', s=sampled_up_votes_sizes, alpha=0.5)
        for t in range(self.n_components):
            plt.text(mean_topic_vectors[t][0],
                     mean_topic_vectors[t][1],
                     top_3_words[t], color=colormap[t],
                     horizontalalignment='center', weight='bold')
        plt.show()


print "Sampling dataset for modeling comparison & determining number of topics"
count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
sampled_df = df.sample(frac=0.1)
sampled_titles = sampled_df['title'].compute().values
sampled_up_votes = sampled_df['up_votes'].compute().values
sampled_document_term_matrix = count_vectorizer.fit_transform(sampled_titles)
print "Done."


print "determining the number of topics"
def tokenize_titles(titles):

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    texts = []
    for title in titles:
        raw = title.decode('utf-8').strip().lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [token for token in tokens if not token in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    return texts

words = tokenize_titles(sampled_titles)
dictionary = corpora.Dictionary(words)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in words]


def compute_plot_coherence_values(doc_term_matrix, stop, start=2, step=1):

    coherence_values = []
    model_list = []
    num_options = range(start, stop, step)
    for num_topics in num_options:
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=words, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    print "coherence scores: {}".format(coherence_values)
    print "plotting coherence score for different number of topics"
    plt.plot(num_options, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence scores"), loc='best')
    plt.show()


compute_plot_coherence_values(doc_term_matrix, 50, 2, 1)

print "setting num of topics to be 14"
n_topics = 14
lsa_model = Topic_Model(df, model_type='LSA', max_features=40000, n_components=n_topics)
lsa_model.get_top_n_topics(count_vectorizer_limited, 10)

lsa_model = Topic_Model(df, model_type='LSA', max_features=40000, n_components=14)
lsa_model.plot_num_title_of_each_topic(count_vectorizer_limited)

lsa_model = Topic_Model(df, model_type='LSA', max_features=40000, n_components=14)
lsa_model.visualize_tnse_2_dimension(count_vectorizer_limited)

print "Try LDA"
lda_model = Topic_Model(df, model_type='LDA', max_features=40000, learning_method='online', n_components=14, random_state=0, verbose=0)
lda_model.get_top_n_topics(count_vectorizer_limited, 10)

lda_model = Topic_Model(df, model_type='LDA', max_features=40000, learning_method='online', n_components=14, random_state=0, verbose=0)
lda_model.plot_num_title_of_each_topic(count_vectorizer_limited)

lda_model = Topic_Model(df, model_type='LDA', max_features=40000, learning_method='online', n_components=14, random_state=0, verbose=0)
lda_model.visualize_tnse_2_dimension(count_vectorizer_limited)

print "Scale up to the entire dataset"
print "lazy loading dataset..."
entire_df = df
entire_titles = entire_df['title'].compute().values
entire_up_votes = entire_df['up_votes'].compute().values
print "Done."


print "initializing LDA model..."
entire_lda_model = Topic_Model(entire_df, sample_rate=0.15, model_type='LDA', max_features=100000, learning_method='online', n_components=14, random_state=0, verbose=0)
entire_lda_model.visualize_tnse_2_dimension(count_vectorizer_limited)


entire_count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
print "count_vectorizer initialized"
entire_document_term_matrix = entire_count_vectorizer.fit_transform(entire_titles)
print "entire_document_term_matrix initialized"
entire_lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=0, verbose=1)
print "LDA model initialized"
entire_lda_model.fit(entire_document_term_matrix);
print "training on the entire dataset done."


years = df['date_created'].dt.year.compute()
start_year, end_year = min(years), max(years)
yearly_topics = {}
for year in range(start_year, end_year + 1):
    print "tackling {} topics...".format(year)
    yearly_ds = df[df['date_created'].dt.year == year]
    yearly_ds_titles = yearly_ds['title'].compute().values
    count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
    document_term_matrix = count_vectorizer.fit_transform(yearly_ds_titles)
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', verbose=0)
    topic_matrix = lda_model.fit_transform(document_term_matrix)
    keys = get_keys(topic_matrix)
    categories, counts = keys_to_counts(keys)
    top_n_words = get_top_n_words(1, 10, document_term_matrix, keys, count_vectorizer)
    yearly_topics[year] = top_n_words
for key in sorted(yearly_topics.keys()):
    print "{} top 10 topic words: {}".format(key, yearly_topics[key])

