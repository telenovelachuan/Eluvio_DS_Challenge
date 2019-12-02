A data science project on the news title headline dataset to explore title feature information and discover yearly topics using natural language processing models.

# Feature explorative ideas

1. Aggregation

- top n words in title, excluding stop words
- number of titles by year
- number of titles by month for each year
- number of titles by each day in month

2. Word Cloud and category deep dive
- title word cloud for 2008
- What's happening on China in 2008

	- It can be inferred that the 2008 Olympic Games in China(Beijing) and the gigantic earthquake topped the focus in 2008.

- title word cloud for 2016

	- While in 2016, news in China focused its attention on south sea affairs and space exploration progress.

- trend of mentions on global warming 

	- Discussions on the topic of "global warming" piked in 2013, and the topic gets more attention afterwards than before 2013.


# LSA & LDA Modeling
Use LSA & LDA model to cluster and generate yearly topics for each year. 

1. Sampling the dataset with a sample rate of 0.05.
2. Determine the number of topics using coherence value.
	- tokenize titles and exclude stop words
	- generate LSA models in the option range of 2 to 50, and plot the coherence score on different number of topics
	- choose number of topics to be 14
3. Generate top 14 topics on the sampled dataset using LSA, each topic containing 10 words.
4. Plot the topic frequencies. Bias observed since the first topic cluster largely overwhelmed the others. The hottest topic clusters in sampled dataset are:
	- attack and assaults in Syria
	- south sea affair in China
	- South and north Korea
	- nuclear issues in Iran
	- issues between Russia and Ukraine
5. Use t-SNE to reduce and visualize the dimension of topic clusters by LSA.
	- utilize the "up_votes" feature here to determine the size of dot radius, so that titles with much up votes will display a larger impact on the scattered plot.
	- the scattered dots in two-dimension plane looks generally organized, but some inter-cluster overlaps and intra-cluster dispersal exist.
6. Generate top 14 topics on the sampled dataset using LDA, each topic containing 10 words.
7. Plot the topic frequencies generate by LDA. It can be observed that LDA generates far more balanced topic clusters.
8. Use t-SNE to reduce and visualize the dimension of topic clusters by LDA.
	- LDA performs better: the clusters are distributed more aggregatively and recognizably, and with less overlap
9. Apply LDA model to the entire dataset.
	- use Dask library to handle and lazy load the big dataset input.
	- due to the resource limit of my computer, I sampled 15% of the entire dataset
10. Generate yearly topic clusters using LDA. The final yearly topics are:
	