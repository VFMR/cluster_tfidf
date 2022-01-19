import math
import random
RND = 42
random.seed(RND)

from tqdm import tqdm
import numpy as np
np.random.seed(RND)
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from .utils import clean_term, count_file_rows
from .clustering import EmbeddingCluster


def get_df(idf, n_docs):
    """idf(t) = log(N / document_frequency) 
       -> document_frequency = exp(idf) / N

    Args:
        idf ([type]): [description]
        n_docs ([type]): [description]

    Returns:
        [type]: [description]
    """
    n_by_df_1 = math.exp(idf)
    df_1 = n_by_df_1/n_docs
    return df_1 - 1

def get_cluster_idf(idf_array, n_docs):
    """Aggregation of idf for the cluster.
    It is not quite clear how to aggregate: summing over individual
    idfs would assume words are always separate, taking the max would assume they
    are mentioned together

    Args:
        idf_array ([type]): [description]
        n_docs ([type]): [description]

    Returns:
        [type]: [description]
    """
    # use max() to approximate the df!
    df = max([get_df(x, n_docs) for x in idf_array])
    return math.log(n_docs/(df+1))



class TfidfCounter:
    def __init__(self, tfidfvectorizer):
        self.tfidfvectorizer = tfidfvectorizer
        self.cleaner = tfidfvectorizer[0]
        self.vocabulary = tfidfvectorizer[-1].vocabulary_
        self.analyzer = tfidfvectorizer[-1].analyzer
        self.counter = CountVectorizer(analyzer=self.analyzer, vocabulary=self.vocabulary)

        self.pipe = self._make_pipeline()

    def _make_pipeline(self):
        
        last_step = ('vectorizer', self.counter)

        if isinstance(self.tfidfvectorizer, sklearn.pipeline.Pipeline):
            pipe = Pipeline(steps=[x for x in self.tfidfvectorizer.steps[:-1]])
            pipe.steps.append(last_step)
        else:
            pipe = Pipeline(steps=[last_step])

        return pipe

    def fit(self, X=None):
        self.pipe.fit(X)

    def transform(self, X):
        return self.pipe.transform(X)

    def fit_transform(self, X):
        return self.pipe.fit_transform(X)

    def inverse_transform(self, X):
        return self.counter.inverse_transform(X)


class ClusterTfidf:
    def __init__(self, 
                 vectorizer,
                 embeddings,
                 load_clustering=False,
                 load_clustering_dir=None,
                 load_clustering_name='clustertfidf',
                 embedding_dim=300,
                 n_top_clusters=7):
        """
        Class for computing Cluster TfIdf.
        on a cluster level.

        Args:
            vectorizer ([type]): sklearn.feature_extraction.text.TfidfVectorizer instance.
                if refit=False, it a fitted instance must be provided
            refit (bool): whether or not the TfidfVectorizer shall be refitted
            embeddings (dict): Embedding lookup
            clustermethod (str): {agglomerative} Method for clustering
            distance_threshold (float): Distance threshold for agglomerative clustering
        """
        # inputs:
        self.vectorizer = vectorizer
        self.counter = TfidfCounter(self.vectorizer)
        self.n_top_clusters = n_top_clusters

        self.clustering  = EmbeddingCluster(embeddings=embeddings, vectorizer=vectorizer, distance_threshold=0.5, n_words=40000)
        
        self.load_clustering = load_clustering
        if load_clustering:
            self.clustering.load(dir=load_clustering_dir, name=load_clustering_name)
        self.embedding_dim = embedding_dim


        # if not self.refit:
        #     self.index2word = self.clustering.index2word
        #     self.word2index = self.clustering.word2index


    def fit(self, X=None):
        """Method to fit the Tfidf if self.refit and to compute the clusters on
            on an array of words.

        Args:
            X (iterable): Array of strings, i.e. non-tokenized texts.
        """
        self.clustering.fit()
        self.clustering.save(dir='../../Temp')
        return self


    def _multi_cluster_func(self, array):
        """Input an array of predicted clusters for different words and
        return a deduplicated array of all the clusters that appear more than once
        

        Args:
            array ([type]): [description]

        Returns:
            [type]: [description]
        """
        return array.unique()


    def input_cleanup(self, X):
        new_X = []
        for row in tqdm(X, desc='Cleaning input'):
            if isinstance(row, str):
                new_X.append(clean_term(row))
            else:
                new_X.append('')
        return pd.Series(new_X)


    def _find_vectorizer_instance(self):
        if isinstance(self.vectorizer, sklearn.pipeline.Pipeline):
            vectorizer = self.vectorizer[-1]
        else:
            vectorizer = self.vectorizer
        return vectorizer


    def _get_idf(self):
        vect = self._find_vectorizer_instance()
        return vect.idf_

    def predict(self, X):
        # tfidf:
        X = self.input_cleanup(X)

        print('Vectorize texts')
        vects = self.vectorizer.transform(X)
        counts = self.counter.transform(X)
        idf = self._get_idf()

        print('Count documents')
        n_docs = count_file_rows(path='../../Output/Data/prepared_data_all.csv')
        
        embeddings = self.clustering.embeddings
        index2cluster = self.clustering.index2cluster
        index2word = self.clustering.index2word
        mc_func = self._multi_cluster_func
        np_array = np.array
        pd_Series = pd.Series

        n_clustered_rows = 0

        result = np.zeros( (len(X), self.embedding_dim) )
        for row_index, row in enumerate(tqdm(vects)):

            do_reporting = False
            
            vect_array = row.toarray()[0]
            count_array = counts[row_index].toarray()[0]
            indices = [str(x) for x in list( np.where(vect_array != 0)[0])]
            words = [index2word[x] for x in indices]

            # get values for aggregation:
            row_idf = np_array([idf[int(x)] for x in indices])
            row_embedding = np_array([embeddings[word] for word in words])
            clusters = pd_Series([index2cluster[x] for x in indices])
            nonzero_counts = pd.Series([count_array[int(x)] for x in indices])

            unique_clusters = mc_func(clusters)
            cluster_vectors = []
            cluster_weights = []
            append_v = cluster_vectors.append
            append_w = cluster_weights.append
            # all_cluster_tf = get_cluster_tf(clusters)

            # TODO: This is ugly and inefficient because I do this loop twice. Find better solution
            max_count = nonzero_counts.max()
            for c in unique_clusters:
                cluster_ix = [i for i, cl in enumerate(clusters) if cl==c]
                clustersum = sum(nonzero_counts[cluster_ix])
                if clustersum > max_count:
                    max_count = clustersum

            for c in unique_clusters:
                cluster_ix = [i for i, cl in enumerate(clusters) if cl==c]


                cluster_tf = sum(nonzero_counts[cluster_ix]) / max_count
                
                idf_filtered = [x for i, x in enumerate(row_idf) if i in cluster_ix]
                cluster_idf = get_cluster_idf(idf_filtered, n_docs)
                cluster_tfidf = cluster_idf * cluster_tf

                # make linear combination of terms of same cluster.
                # use normalized regular tfidf weights
                weights = [vect_array[int(indices[i])] for i in cluster_ix]
                weights_norm = [x/sum(weights) for x in weights]

                cluster_embeddings = [row_embedding[i] for i in cluster_ix]
                vectors = np.array([e*w for e, w in zip(cluster_embeddings, weights_norm)])
                append_v(vectors)
                append_w(cluster_tfidf)
                
                # Reporting:
                # if len(cluster_ix)>1:
                #     print(X[row_index])
                #     print(words)
                #     print(f'Cluster indices for cluster {c}: {[words[ix] for ix in cluster_ix]}')
                #     print(f'IDF: {cluster_idf}, TFIDF: {cluster_tfidf}, TF: {cluster_tf}')
                #     print(f'Weights: {weights_norm}')
                #     do_reporting = True

            
            # aggregation of  row into embedding_dim-array
            top_vecs = [(w, v) for w, v in zip(cluster_weights, cluster_vectors)]
            top_vecs = sorted(top_vecs, key=lambda x: x[0])

            maxvecs = min(self.n_top_clusters, len(top_vecs))
            top_vecs = top_vecs[:maxvecs]
            
            # normalize weights:
            top_embeds = np.array([x[1][0] for x in top_vecs])
            top_weights = [x[0] for x in top_vecs]
            tw_sum = sum(top_weights)
            top_weights = np.array([x/tw_sum for x in top_weights])
            result[row_index] = top_weights@top_embeds  

            # Reporting
            # if do_reporting:
            #     n_clustered_rows += 1
            #     simils = sorted([(myCosine(embeddings[word], result[row_index]), word) for word in words])[::-1]
            #     for pair in simils:
            #         print(f'Similarity to {pair[1]}: {pair[0]:0.3f}')
            #     print(f'Number of clustered rows: {n_clustered_rows}/{row_index}')
            #     print()
            #     print('****************')

        return result
            