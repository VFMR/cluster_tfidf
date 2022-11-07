import math
import random
from statistics import mean

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from .utils import clean_term, count_file_rows
from .clustering import EmbeddingCluster
from .base import _BaseEmbeddingClass



def get_df(idf, n_docs):
    """idf(t) = log(N / document_frequency)
       -> document_frequency = N / exp(idf)

    Args:
        idf ([type]): [description]
        n_docs ([type]): [description]

    Returns:
        [type]: [description]
    """
    e_idf = math.exp(idf)
    df_1 = n_docs / e_idf
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



class TfidfCounter(_BaseEmbeddingClass):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

        xvect = self._find_vectorizer_instance()
        self.vocabulary = xvect.vocabulary_
        self.analyzer = xvect.analyzer
        self.counter = CountVectorizer(analyzer=self.analyzer, vocabulary=self.vocabulary)

        self.pipe = self._make_pipeline()

    def _make_pipeline(self):

        last_step = ('vectorizer', self.counter)

        if isinstance(self.vectorizer, sklearn.pipeline.Pipeline):
            pipe = Pipeline(steps=[x for x in self.vectorizer.steps[:-1]])
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


class ClusterTfidfVectorizer(_BaseEmbeddingClass, TransformerMixin):
    def __init__(self,
                 vectorizer,
                 embeddings,
                 n_docs,
                 corpus_path=None,
                 corpus_path_encoding='latin1',
                 load_clustering=False,
                 embedding_dim=None,
                 checkterm='test',
                 n_top_clusters=7,
                 cluster_share=0.2,
                 clustermethod='agglomerative',
                 distance_threshold=0.5,
                 n_words=40000,
                 n_jobs=-1):
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
        self.base_args = [
            'vectorizer'
            'embeddings'
            'n_docs'
            'corpus_path'
            'corpus_path_encoding'
            'load_clustering'
            'embedding_dim'
            'checkterm'
            'n_top_clusters'
            'clustermethod'
            'distance_threshold'
            'n_top_words'
            'cluster_share'
        ]

        # checks
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

        self.clustering  = EmbeddingCluster(embeddings=embeddings,
                                            vectorizer=vectorizer,
                                            clustermethod=clustermethod,
                                            distance_threshold=distance_threshold,
                                            n_words=n_words,
                                            cluster_share=cluster_share,
                                            checkterm=checkterm,
                                            n_jobs=n_jobs)

        self.load_clustering = load_clustering
        if load_clustering:
            self.clustering.load(load_clustering)

        # Get embedding dimensionality by checking against some Word.
        if embedding_dim:
            self._embedding_dim = embedding_dim
        else:
            self._embedding_dim = self._get_embedding_dim(embeddings, checkterm=checkterm)

        self.n_jobs = n_jobs

    def set_params(self, **kwargs):
        print(self.__dict__)
        own_params = {key: value for key, value in kwargs.items() if key in self.__dict__}
        model_params = {key: value for key, value in kwargs.items() if key not in self.__dict__}
        for key, value in own_params.items():
            self.__dict__.update({key: value})

        self.clustering.set_params(**model_params)


    def get_params(self):
        params = self.__dict__
        params.update(self.clustering.get_params())

        # exclude "private" parameters:
        params = {key: value for key, value in params.items() if not key.startswith('_')}
        return params


    def fit(self, X=None, y=None, savedir=None, savename='clustertfidf'):
        """Convenience function to have a fit methods.
        Only calls the cluster_vocab method in its place.

        Args:
            savedir ([type], optional): [description]. Defaults to None.
            savename (str, optional): [description]. Defaults to 'clustertfidf'.
        """
        self.clustering.fit()
        if savedir:
            self.clustering.save(dir=savedir, name=savename)
        return self


    def save(self, dir, name='clustertfidf'):
        self.clustering.save(dir=dir, name=name)


    def load(self, path):
        self.clustering.load(path=path)


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
        for row in X:
            if isinstance(row, str):
                new_X.append(clean_term(row))
            else:
                new_X.append('')
        return pd.Series(new_X)


    def _get_idf(self):
        vect = self._find_vectorizer_instance()
        return vect.idf_


    def transform(self, X, aggregate_word_level=True):
        X = self._input_cleanup(X)

        vects = self.vectorizer.transform(X)
        counts = self.counter.transform(X)
        idf = self._get_idf()

        n_docs = self.n_docs

        # setting up some variables for potential minor speed boost:
        embeddings = self.clustering.embeddings
        index2cluster = self.clustering.index2cluster
        index2word = self.clustering.index2word
        mc_func = self._multi_cluster_func
        np_array = np.array
        pd_Series = pd.Series

        n_clustered_rows = 0
        if aggregate_word_level:
            result = np.zeros( (len(X), self._embedding_dim) )
        else:
            result = {
                'vectors': [],
                'weights': []  #np.zeros( (len(X), self._n_top_clusters) )
                }
        for row_index, row in enumerate(vects):

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
            if aggregate_world_level:
                result[row_index] = top_weights@top_embeds
            else:
                result['weights'].append(top_weights)
                result['vectors'].append(top_embeds)


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
