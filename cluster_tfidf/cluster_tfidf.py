import math
import json
import random
RND = 42
random.seed(RND)

from tqdm import tqdm
import numpy as np
np.random.seed(RND)
import pandas as pd
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def clean_term(term):
    replacement_dct = {
        #'\\': '/',
        'ã¤': 'ä',
        'ã¼': 'ü',
        'ã¶': 'ö',
        #'Ã\\x9c': 'Ü',´
        'ã\x9f': 'ß',
        'ã\\x9f': 'ß',
        'Ã\x9c': 'Ü',
        'Ã\\x9c': 'Ü',
        'Ã¼': 'Ü',
        'Ã¶': 'Ö' ,
        'Ã\x96': 'Ö',
        'Ã\\x96': 'Ö',
        'Ã¤': 'Ä',
        r'Ã\x84': 'Ä',
        'Ã\x84': 'Ä',
        'Ã\\x84': 'Ä'
        }

    for key, value in replacement_dct.items():
        term = term.replace(key, value)
    
    return term

def myCosine(a, b):
    return a@b / (np.linalg.norm(a)*np.linalg.norm(b))

# def cosine_sim_words(x, y):
#     a = embeddings[x]
#     b = embeddings[y]
#     return myCosine(a, b)

# def check_pairs(lst):
#     for row in lst:
#         for a in row:
#             for b in row:
#                 if a!=b:
#                     print(f'{a} & {b}: {cosine_sim_words(a, b):0.2f}')


def get_n_docs_from_training_path(path='../../Output/Data/prepared_data_all.csv'):
    with open(path, 'r', encoding='latin1') as f:
        n_docs = sum([1 for _ in f.readlines()])-1
    return n_docs

def get_cluster_tf(cluster_array):
    # TODO: Currently, these are not sorted -> will not work when multiplying with idf!

    # numerator: how often does term appear in doc
    numerator = pd.Series(cluster_array).value_counts()
    print(f'TF numerator: {numerator}')

    # denominator: how often does the most frequent term appear (serves to normalize)
    denominator = numerator.max()

    return numerator/denominator

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


def get_cluster_tfidf(clusters, idf_array, n_docs):
    tf = get_cluster_tf(clusters)
    idf = get_cluster_idf(idf_array, n_docs)
    return tf*idf

def analyze_clusters(clustering, filename):
    index2word = clustering.index2word
    index2cluster = clustering.index2cluster

    multi_clusters = clustering.get_multi_clusters()
    print(f'Final number of clusters: {len(multi_clusters)}')

    words = {y: [index2word[x] for x in [key for key, value in index2cluster.items() if value==y]] for y in multi_clusters}
    with open(filename, 'w') as f:
        for key, value in words.items():
            f.write(f'{key}: {value}\n')
    return words


class BaseEmbeddingClass:
    def __init__(self, embeddings, vectorizer):
        """
        """
        # input
        self.embeddings = embeddings
        self.vectorizer = vectorizer

        # checks
        xvect = self._find_vectorizer_instance()
        if not isinstance(xvect, sklearn.feature_extraction.text.TfidfVectorizer):
            raise ValueError(f"""
                Vectorizer must be either a sklearn.feature_extraction.text.TfidfVectorizer
                instance or an sklearn.pipeline.Pipeline instance with 
                sklearn.feature_extraction.text.TfidfVectorizer being the last step.
                {type(xvect)} was provided instead.
                """)
        self.vocabulary = {clean_term(term): str(ix) for term, ix in xvect.vocabulary_.items()}
        self.vocabulary = xvect.vocabulary_.items()

        # retrieve values:
        self.embedding_dim = len(embeddings['test'])


    def _embed_array_of_words(self, X):
        array = np.zeros( (len(X), self.embedding_dim) )
        for i, word in enumerate(tqdm(X, desc='Embedding words')):
            array[i] = self.embeddings[word]

        # remove OOV Words:
        array = array[~(array==0).all(1)]  
        return array


    def _padding(self, X):
        """Padding to make transform rows with a variable number of words into a matrix.

        Args:
            X (iterable of iterables): The array to pad

        Returns:
            numpy.array: matrix with padded rows
        """
        maxlen = max([len(row) for row in X])
        result_array = np.zeros( (len(X), maxlen) )
        for i, row in X:
            padded_row = np.array(row + [0]*(maxlen-len(row)))
            result_array[i] = padded_row
        return result_array


    def _embedding_aggregation(self, X, weights):
        """compute the weighted aggregation of words given a set of words.

        Args:
            X (iterable of iterable): Array where each row is an array of strings. I.e.
                documents split into words.
            weights ([type]): weights to multiply each word embedding with. Number of
                elements in each sublist must be the same as X

        Returns:
            numpy.array: the (n, embedding_dim) matrix of weighted embeddings
        """
        result = np.zeros( (len(X), self.embedding_dim))
        
        # implemented in a loop because of variable length of rows.
        # Padding instead avoided due to memory concerns.
        # While vectorized operation would be faster, this appears
        # to be sufficiently fast.
        for i, row in enumerate(tqdm(X, desc='Aggregation of embeddings')):

            row_weights = np.array(weights[i])
            embedding_mat = np.array([self.embeddings[x] for x in row])

            aggregate = row_weights@embedding_mat
            result[i] = aggregate
        return result


    def _find_vectorizer_instance(self):
        if isinstance(self.vectorizer, sklearn.pipeline.Pipeline):
            vectorizer = self.vectorizer[-1]
        else:
            vectorizer = self.vectorizer
        return vectorizer


    def _get_idf(self):
        vect = self._find_vectorizer_instance()
        return vect.idf_


    def _get_index2word(self):
        vectorizer = self._find_vectorizer_instance()     
        index2word = {str(i): clean_term(term) for term, i in vectorizer.vocabulary_.items()}

        return index2word


    def _get_word2index(self, index2word):
        return {value: key for key, value in index2word.items()}


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
        # top_words = [x[] for x in idf_vocab]
        # indices = [x['] for x in idf_vocab]
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
        analyze_clusters(self.clustering, '../../Temp/clustering_agglomerative.txt')
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
        n_docs = get_n_docs_from_training_path(path='../../Output/Data/prepared_data_all.csv')
        
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
            