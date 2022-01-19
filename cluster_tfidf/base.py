import numpy as np
import sklearn
from tqdm import tqdm

from .utils import clean_term

class _BaseEmbeddingClass:
    def __init__(self, embeddings, vectorizer, checkterm='test'):
        """
        """
        # input
        self.embeddings = embeddings
        self.vectorizer = vectorizer

        xvect = self._find_vectorizer_instance()
        # BUG: for some reason a error occurs when cleaning terms.
        # However, cleaning should be done
        # self.vocabulary = {clean_term(term): str(ix) for term, ix in xvect.vocabulary_.items()}
        self.vocabulary = xvect.vocabulary_.items()

        # retrieve values:
        self.embedding_dim = self._get_embedding_dim(self.embeddings, checkterm=checkterm)


    def _embedding_lookup(self, term):
        try:
            result = self.embeddings[term]
        except:
            result = np.zeros(self.embedding_dim)
        return result


    def _get_embedding_dim(self, embeddings, checkterm='test'):
        array = embeddings[checkterm]
        if len(array)==1:
            embedding_dim = len(array[0])
        else:
            embedding_dim = len(array)
        return embedding_dim


    def _embed_array_of_words(self, X):
        array = np.zeros( (len(X), self.embedding_dim) )
        for i, word in enumerate(tqdm(X, desc='Embedding lookup')):
            array[i] = self._embedding_lookup(word)

        # remove OOV Words:
        return self._remove_oov(array)


    def _remove_oov(self, array):
        return array[~(array==0).all(1)] 


    # REMOVE This appears to do nothing:
    # def _padding(self, X):
    #     """Padding to make transform rows with a variable number of words into a matrix.

    #     Args:
    #         X (iterable of iterables): The array to pad

    #     Returns:
    #         numpy.array: matrix with padded rows
    #     """
    #     maxlen = max([len(row) for row in X])
    #     result_array = np.zeros( (len(X), maxlen) )
    #     for i, row in X:
    #         result_array[i] = np.array(row + [0]*(maxlen-len(row)))
    #     return result_array


    # REMOVE Not used.
    # def _embedding_aggregation(self, X, weights):
    #     """compute the weighted aggregation of words given a set of words.

    #     Args:
    #         X (iterable of iterable): Array where each row is an array of strings. I.e.
    #             documents split into words.
    #         weights ([type]): weights to multiply each word embedding with. Number of
    #             elements in each sublist must be the same as X

    #     Returns:
    #         numpy.array: the (n, embedding_dim) matrix of weighted embeddings
    #     """
    #     result = np.zeros( (len(X), self.embedding_dim))
        
    #     # implemented in a loop because of variable length of rows.
    #     # Padding instead avoided due to memory concerns.
    #     # While vectorized operation would be faster, this appears
    #     # to be sufficiently fast.
    #     for i, row in enumerate(tqdm(X, desc='Aggregation of embeddings')):

    #         row_weights = np.array(weights[i])
    #         embedding_mat = np.array([self.embeddings[x] for x in row])

    #         aggregate = row_weights@embedding_mat
    #         result[i] = aggregate
    #     return result


    def _is_tfidf(self, obj):
        return isinstance(obj, sklearn.feature_extraction.text.TfidfVectorizer)


    def _find_vectorizer_instance(self):
        vect_error = False
        if self._is_tfidf(self.vectorizer):
            vectorizer = self.vectorizer
        elif isinstance(self.vectorizer, sklearn.pipeline.Pipeline):
            if self._is_tfidf(self.vectorizer[-1]):
                vectorizer  = self.vectorizer[-1]
            else:
                vect_error = True
        else:
            vect_error = True

        if vect_error:
            raise ValueError(f"""
                Vectorizer must be either a sklearn.feature_extraction.text.TfidfVectorizer
                instance or an sklearn.pipeline.Pipeline instance with 
                sklearn.feature_extraction.text.TfidfVectorizer being the last step.
                """)

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
