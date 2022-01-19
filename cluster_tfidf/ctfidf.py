import math
import random
from statistics import mean

from base import BaseEmbeddingClass
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
from .base import _BaseEmbeddingClass


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


def get_cluster_idf(idf_array, n_docs, aggregator='max'):
    """Aggregation of idf for the cluster.
    It is not quite clear how to aggregate: summing over individual
    idfs would assume words are always separate, taking the max would assume they
    are mentioned together

    Args:
        idf_array ([type]): [description]
        n_docs ([type]): [description]
        aggregator (str or callable): function to approximate
            the aggregated document frequency from individual ones. 
            Can be {'max', 'min', 'mean',} or callable.
            Defaults to 'max' to use the maximum df.

    Returns:
        [type]: [description]
    """
    if isinstance(aggregator, str) and aggregator in ['min', 'max', 'mean']:
        if aggregator=='max':
            func = max
        elif aggregator=='min':
            func = min
        elif aggregator=='mean':
            func = mean
    elif callable(aggregator):
        func = aggregator
    else:
        raise ValueError('aggregator must be one of {"max", "min", "mean"} or callable')
    
    df = func([get_df(x, n_docs) for x in idf_array])
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


class ClusterTfidf(_BaseEmbeddingClass):
    def __init__(self, 
                 vectorizer,
                 embeddings,
                 n_docs,
                 corpus_path=None,
                 corpus_path_encoding='latin1',
                 load_clustering=False,
                 embedding_dim=None,
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
        if n_docs:
            if isinstance(n_docs, int):
                self.n_docs = n_docs
            else:
                raise ValueError('n_docs must be of type int')
        else:
            if corpus_path:
                self.n_docs = count_file_rows(corpus_path, encoding=corpus_path_encoding)
            else:
                raise ValueError('either n_docs or corpus_path must be specified')

        self.vectorizer = vectorizer
        self.counter = TfidfCounter(self.vectorizer)
        self.n_top_clusters = n_top_clusters

        self.clustering  = EmbeddingCluster(embeddings=embeddings, vectorizer=vectorizer, distance_threshold=0.5, n_words=40000)
        
        self.load_clustering = load_clustering
        if load_clustering:
            self.clustering.load(load_clustering)
        
        # Get embedding dimensionality by checking against some Word.
        if embedding_dim:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self._get_embedding_dim(checkterm='test')


        # if not self.refit:
        #     self.index2word = self.clustering.index2word
        #     self.word2index = self.clustering.word2index


    def _get_embedding_dim(self, checkterm='test'):
        array = self.embeddings[checkterm]
        if len(array)==1:
            embedding_dim = len(array[0])
        else:
            embedding_dim = len(array)
        return embedding_dim


    # HACK: fit method is not required
    # Only here to make user interface simpler.
    # This may be solvable with better name for cluster_vocab method.
    def fit(self, savedir=None, savename='clustertfidf'):
        """Convenience function to have a fit methods.
        Only calls the cluster_vocab method in its place.

        Args:
            savedir ([type], optional): [description]. Defaults to None.
            savename (str, optional): [description]. Defaults to 'clustertfidf'.
        """
        self.cluster_vocab(savedir, savename)


    def cluster_vocab(self, savedir=None, savename='clustertfidf'):
        """Method to fit the Tfidf if self.refit and to compute the clusters on
            on an array of words.

        Args:
            X (iterable): Array of strings, i.e. non-tokenized texts.
        """
        self.clustering.fit()
        if savedir:
            self.clustering.save(dir=savedir, name=savename)
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


    def _input_cleanup(self, X):
        new_X = []
        for row in tqdm(X, desc='Cleaning input'):
            if isinstance(row, str):
                new_X.append(clean_term(row))
            else:
                new_X.append('')
        return pd.Series(new_X)



    def _get_idf(self):
        vect = self._find_vectorizer_instance()
        return vect.idf_

    def predict(self, X):
        # tfidf:
        X = self._input_cleanup(X)

        print('Vectorize texts')
        vects = self.vectorizer.transform(X)
        counts = self.counter.transform(X)
        idf = self._get_idf()

        n_docs = self.n_docs
        
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


            # HACK: This is ugly and inefficient because I do this loop twice.
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
            