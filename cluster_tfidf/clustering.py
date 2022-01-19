import random
RND = 42
random.seed(RND)
import json

import numpy as np
np.random.seed(RND)
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from base import BaseEmbeddingClass
from utils import clean_term

class EmbeddingCluster(BaseEmbeddingClass):
    def __init__(self, 
                 embeddings,
                 vectorizer,
                 clustermethod='agglomerative',
                 distance_threshold=0.4,
                 n_words=False,
                 cluster_share=0.2):
        """[summary]

        Args:
            embeddings ([type]): [description]
            vectorizer (): 
            clustermethod (str, optional): [description]. Defaults to 'agglomerative'.
            distance_threshold (float, optional): [description]. Defaults to 0.3.

        Raises:
            ValueError: if clustermethod not valid
        """
        super().__init__(embeddings=embeddings, vectorizer=vectorizer)
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
            self.n_words = n_words
        else:
            self.n_words = len(self.index2word)
        self.n_clusters = int(cluster_share*self.n_words)

        # restrict embeddings to relevant words to save memory
        self.embeddings = {word: embeddings[word] for word in self.index2word.values()}


    def _get_cluster_model(self):
        if self.clustermethod=='agglomerative':
            model = AgglomerativeClustering(n_clusters=None,
                                            affinity='cosine',
                                            distance_threshold=self.distance_threshold,
                                            linkage='average')
        elif self.clustermethod=='kmeans':
            model = KMeans(n_clusters=self.n_clusters, random_state=RND, n_jobs=-1)
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


    def get_multi_clusters(self):
        return self._multi_cluster_func(self.index2cluster.values())


    def get_n_clusters(self):
        return len(self.get_multi_clusters())

    def _clustering_split(self, split, cluster=True):
        split_indices = [x[1] for x in split]
        split_words = [x[0] for x in split]
        X_embeds = np.array([x[2] for x in split])
        norm = np.linalg.norm
        self.norms = self.norms+[norm(x) for x in X_embeds]
        
        if cluster:
            clusters = self.model.fit_predict(X_embeds) + self.maxcluster
        else:
            clusters = np.arange(len(split)) + 1 + self.maxcluster
        
        index2cluster = {ix: c for ix, c in zip(split_indices, clusters)}
        self.index2cluster.update(index2cluster)
        self.maxcluster = max([x for x in self.index2cluster.values()])

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
        X_top = X[:self.n_words]
        X_bottom = X[self.n_words:]
        
        random.shuffle(X_top)
        
        self.index2cluster = {}
        self.maxcluster = 0
        self.norms = []
        excluded = []
        indices = []


        # agglomerative clustering cannot handle all data. Thus make three random splits
        # and cluster each of these
        
        if self.clustermethod=='agglomerative':
            X_splits = np.array_split(X_top, 3)
        else:  # Kmeans can cluster all words at once
            X_splits = [X_top]

        # deactivate this in case it cannot be run!
        X_splits = [X_top]
        
        for split in tqdm(X_splits, desc='Clustering'):
            embedded_split = [(x[0], x[1], self.embeddings[x[0]]) for x in split]
            excluded = excluded+[x for x in embedded_split if not x[2].any()]
            embedded_split = [x for x in embedded_split if x[2].any()]
            indices = indices+[x[1] for x in embedded_split]
            self._clustering_split(embedded_split)

        # manually add "clusters" for left out terms:
        X_bottom_w_embeddings = [(x[0], x[1], self.embeddings[x[0]]) for x in X_bottom]
        for array in [excluded, X_bottom_w_embeddings]:
            self._clustering_split(array, cluster=False)
            indices = indices+[x[1] for x in array]
        
            
        self.norms = np.array(self.norms).reshape(-1, 1)
        # scaling the norms to be in the [0, 1] range:
        scaler = MinMaxScaler()
        norms_scaled = scaler.fit_transform(self.norms)
        self.index2norm = {indices[i]: norm[0] for i, norm in enumerate(norms_scaled)}

        
        self._fix_missing_clusters()
        return self


    def save(self, dir, name='clustertfidf'):
        """ Save to disk to allow using this model later on. 
        Saves multiple files into a folder

        Args:
            dir (str): name of directory to save in.
            name (str): name of the directory that is created to save 
        """
        exports = {
            'index2word': self.index2word,
            'word2index': self.word2index,
            'index2norm': {key: float(value) for key, value in self.index2norm.items()},
            'index2cluster': {key: int(value) for key, value in self.index2cluster.items()}
            }
        for key, value in exports.items():
            print(key)
            with open(f'{dir}/{name}_{key}.json', 'w') as f:
                json.dump(value, f)


    def load_obj(self, file):
        with open(file+'.json', 'r') as f:
            content = f.read()
            result = json.loads(content)
        return result


    def load(self, dir, name='clustertfidf'):
        """[summary]

        Args:
            dir ([type]): [description]
            name (str, optional): [description]. Defaults to 'clustertfidf'.
        """
        self.index2word = self.load_obj(dir+'/'+name+'_'+'index2word')
        self.word2index = self.load_obj(dir+'/'+name+'_'+'word2index')
        self.index2norm = self.load_obj(dir+'/'+name+'_'+'index2norm')
        self.index2cluster = self.load_obj(dir+'/'+name+'_'+'index2cluster')

        # make sure values are numeric:
        self.index2norm = {key: float(value) for key, value in self.index2norm.items()}
        self.index2cluster = {key: int(value) for key, value in self.index2cluster.items()}
