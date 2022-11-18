import random
import json
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from .base import _BaseEmbeddingClass
from .utils import clean_term


class EmbeddingCluster(_BaseEmbeddingClass):
    def __init__(self,
                 embeddings,
                 vectorizer,
                 clustermethod='agglomerative',
                 distance_threshold=0.4,
                 n_words=False,
                 cluster_share=0.2,
                 checkterm='test',
                 n_jobs=-1,
                 **kwargs):
        """[summary]

        Args:
            embeddings ([type]): [description]
            vectorizer ():
            clustermethod (str, optional): [description]. Defaults to 'agglomerative'.
            distance_threshold (float, optional): [description]. Defaults to 0.3.

        Raises:
            ValueError: if clustermethod not valid
        """
        super().__init__(embeddings=embeddings,
                         vectorizer=vectorizer,
                         checkterm=checkterm)
        # input values:
        self.clustermethod = clustermethod
        self.distance_threshold = distance_threshold

        self.model = self._get_cluster_model(**kwargs)

        self.index2word = self._get_index2word()
        self.word2index = self._get_word2index(self.index2word)

        if n_words:
            self.n_words = min(n_words, len(self.index2word))
        else:
            self.n_words = len(self.index2word)
        self._n_clusters = int(cluster_share*self.n_words)

        # restrict embeddings to relevant words to save memory
        self.embeddings = {word: self._embedding_lookup(word) for word in self.index2word.values()}

        self.n_jobs = n_jobs

    def _get_cluster_model(self, **kwargs):
        allowed_clustermethods = ['agglomerative', 'kmeans']
        throw_error = False

        if isinstance(self.clustermethod, str):
            if self.clustermethod in allowed_clustermethods:
                if self.clustermethod=='agglomerative':

                    # define standard parameters and update those set by user:
                    model_args = {
                        'n_clusters': None,
                        'affinity': 'cosine',
                        'distance_threshold': self.distance_threshold,
                        'linkage': 'average'
                    }
                    model_args.update(kwargs)

                    model = AgglomerativeClustering(**model_args)
                elif self.clustermethod=='kmeans':
                    model_args = {
                        'n_clusters': self._n_clusters,
                        'n_jobs': self.n_jobs
                    }
                    model_args.update(kwargs)
                    model = KMeans(**model_args)
            else:
                raise ValueError(f"""Inappropriate argument value for 'clustermethod'.
                             Must be one of {allowed_clustermethods}""")

        else:
            try:
                model = self.clustermethod(**kwargs)
            except:
                model = self.clustermethod
                model.set_params(**kwargs)

        return model


    def set_params(self, **kwargs):
        own_params = {key: value for key, value in kwargs.items() if key in self.__dict__}
        model_params = {key: value for key, value in kwargs.items() if key not in self.__dict__}
        for key, value in own_params.items():
            self.__dict__.update({key: value})

        self.model.set_params(**model_params)


    def get_params(self):
        params = self.__dict__
        params.update(self.model.get_params())

        # exclude "private" parameters:
        params = {key: value for key, value in params.items if not key.startswith('_')}
        return params


    def _find_top_words(self):
        vectorizer = self._find_vectorizer_instance()
        idf = vectorizer.idf_
        vocab = {clean_term(term): ix for term, ix in vectorizer.vocabulary_.items()}
        idf_vocab = sorted([ (idf[value], key) for key, value in vocab.items() ])
        top_words = [(x[1], self.word2index[x[1]]) for x in idf_vocab]

        return top_words


    def _cosine_distance(self, a, b):
        return a@b / (np.linalg.norm(a)*np.linalg.norm(b))


    def _multi_cluster_func(self, array):
        counts = pd.Series(array).value_counts()
        multi_clusters = counts[counts>1].index
        return multi_clusters


    def _get_multi_clusters(self):
        return self._multi_cluster_func(self.index2cluster.values())


    def _get_n_clusters(self):
        return len(self._get_multi_clusters())


    def _update_clusters(self, split, cluster=True):
        split_indices = [x[1] for x in split]
        X_embeds = np.array([x[2] for x in split])
        norm = np.linalg.norm
        self._norms = self._norms+[norm(x) for x in X_embeds]

        if cluster:
            clusters = self.model.fit_predict(X_embeds) + self._maxcluster
        else:
            clusters = np.arange(len(split)) + 1 + self._maxcluster

        index2cluster = {ix: c for ix, c in zip(split_indices, clusters)}
        self.index2cluster.update(index2cluster)
        self._maxcluster = max([x for x in self.index2cluster.values()])


    def _fix_missing_clusters(self):
        missing_ix = [ix for ix in self.index2word.keys() if not ix in self.index2cluster.keys()]
        maxcluster = max(self.index2cluster.values())
        for ix in missing_ix:
            maxcluster += 1
            self.index2cluster.update({ix: maxcluster})


    def fit(self, X=None):
        """[summary]

        Args:
            X (iterable): corpus containing
            selection (str, optional): {random, corpus} Method to select word sample.
                'random' will use index2word to retrieve words. Defaults to 'random'.
            n_words (int, optional): Number of randomly selected words to consider for
                the clustering. Note that Agglomerative Clustering Memory is O(n**3).
                Defaults to 10,000
            distance_thresh (float, optional): Distance threshold to predict whether
                vector belongs to one of the clusters. Distance measured via Cosine
                distance.

        Returns
            np.array
        """
        X = self._find_top_words()
        X_top = X[:self.n_words]
        X_bottom = X[self.n_words:]

        random.shuffle(X_top)

        self.index2cluster = {}
        self._maxcluster = 0
        self._norms = []
        excluded = []
        indices = []


        embedded_array = [(x[0], x[1], self._embedding_lookup(x[0])) for x in X_top]

        # remove all terms that have zero-vector, i.e. oov
        excluded = excluded+[x for x in embedded_array if not x[2].any()]
        embedded_array = [x for x in embedded_array if x[2].any()]
        indices = indices+[x[1] for x in embedded_array]
        self._update_clusters(embedded_array)

        # manually add "clusters" for left out terms:
        X_bottom_w_embeddings = [(x[0], x[1], self._embedding_lookup(x[0])) for x in X_bottom]
        for array in [excluded, X_bottom_w_embeddings]:
            self._update_clusters(array, cluster=False)
            indices = indices+[x[1] for x in array]

        # scaling the norms and adding them to lookup dictionary
        norms_scaled = self._scale_norms(self._norms)
        self.index2norm = {indices[i]: norm[0] for i, norm in enumerate(norms_scaled)}

        # HACK: I am not sure what caused this, but a tiny number of terms
        # still did not receive a cluster
        self._fix_missing_clusters()

        return self


    def _scale_norms(self, norms):
        """Scale norms to be in [0,1] range

        Args:
            norms (numpy.array): Array containing the vector norms

        Returns:
            numpy.array: scaled vector norms
        """
        norms = np.array(norms).reshape(-1, 1)
        scaler = MinMaxScaler()
        norms_scaled = scaler.fit_transform(norms)
        return norms_scaled


    def _make_save_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            print('Save folder already exists. Overwriting content.')


    def save(self, dir, name='clustertfidf'):
        """ Save to disk to allow using this model later on.
        Saves multiple files into a folder

        Args:
            dir (str): name of directory to save in.
            name (str): name of the directory that is created to save
        """
        # create save folder
        folder = os.path.join(dir, name)
        self._make_save_folder(folder)

        exports = {
            'index2word': self.index2word,
            'word2index': self.word2index,
            'index2norm': {key: float(value) for key, value in self.index2norm.items()},
            'index2cluster': {key: int(value) for key, value in self.index2cluster.items()}
            }

        for key, value in exports.items():
            filename = os.path.join(folder, key+'.json')
            with open(filename, 'w') as f:
                json.dump(value, f)

    def _load_obj(self, file: str, archive: ZipFile = None):
        if archive is None:
            with open(file+'.json', 'r') as f:
                content = f.read()
                result = json.loads(content)
        else:
            with archive.open(file+'.json', 'r') as f:
                content = f.read()
                result = json.loads(content)
        return result

    def load(self, path: str, archive: ZipFile = None):
        """[summary]

        Args:
            path (str): Name of the directory that holds the results from save method.
        """
        self.index2word = self._load_obj(path+'/index2word', archive=archive)
        self.word2index = self._load_obj(path+'/word2index', archive=archive)
        self.index2norm = self._load_obj(path+'/index2norm', archive=archive)
        self.index2cluster = self._load_obj(path+'/index2cluster', archive=archive)


        # make sure values are numeric:
        self.index2norm = {key: float(value) for key, value in self.index2norm.items()}
        self.index2cluster = {key: int(value) for key, value in self.index2cluster.items()}
