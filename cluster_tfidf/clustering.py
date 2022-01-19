import random
from subprocess import NORMAL_PRIORITY_CLASS
RND = 42
random.seed(RND)
import json
import os

import numpy as np
np.random.seed(RND)
import pandas as pd
from tqdm import tqdm
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
                 checkterm='test'):
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

        # checks:
        allowed_clustermethods = ['agglomerative', 'kmeans']
        if clustermethod not in allowed_clustermethods:
            raise ValueError(f"""Inappropriate argument value for 'clustermethod'. 
                                 Must be one of {allowed_clustermethods}""")

        self.index2word = self._get_index2word()
        self.word2index = self._get_word2index(self.index2word)

        if n_words:
            self._n_words = min(n_words, len(self.index2word))
        else:
            self._n_words = len(self.index2word)
        self._n_clusters = int(cluster_share*self._n_words)

        # restrict embeddings to relevant words to save memory
        self.embeddings = {word: self._embedding_lookup(word) for word in self.index2word.values()}


    def _get_cluster_model(self):
        if self.clustermethod=='agglomerative':
            model = AgglomerativeClustering(n_clusters=None,
                                            affinity='cosine',
                                            distance_threshold=self.distance_threshold,
                                            linkage='average')
        elif self.clustermethod=='kmeans':
            model = KMeans(n_clusters=self._n_clusters, random_state=RND, n_jobs=-1)
        return model


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
        self.model = self._get_cluster_model()
        # X = [(x[0], x[1]) for x in self.vocabulary
        X = self._find_top_words()
        X_top = X[:self._n_words]
        X_bottom = X[self._n_words:]
        
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


    def _make_save_folder(folder):
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


    def _load_obj(self, file):
        with open(file+'.json', 'r') as f:
            content = f.read()
            result = json.loads(content)
        return result


    def load(self, path):
        """[summary]

        Args:
            path (str): Name of the directory that holds the results from save method.
        """
        self.index2word = self._load_obj(path+'/index2word')
        self.word2index = self._load_obj(path+'/word2index')
        self.index2norm = self._load_obj(path+'/index2norm')
        self.index2cluster = self._load_obj(path+'/index2cluster')

        # make sure values are numeric:
        self.index2norm = {key: float(value) for key, value in self.index2norm.items()}
        self.index2cluster = {key: int(value) for key, value in self.index2cluster.items()}
