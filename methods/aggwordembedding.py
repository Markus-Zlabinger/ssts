import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import warnings



class Tokens2Embeddings:

    @staticmethod
    def transform(token_lists, vocabulary):
        if (isinstance(token_lists, list)) is False:
            raise Exception("Should be list of lists")

        if all(isinstance(el, list) for el in token_lists) is False:
            raise Exception("Should be list of lists")

        word_embedding_lists = []
        for token_list in token_lists:
            word_embedding_list = []
            for token in token_list:
                if token not in vocabulary:
                    raise Exception("Following token was not found in the vocabulary:", token)
                else:
                    word_embedding_list.append(vocabulary[token])
            word_embedding_lists.append(word_embedding_list)
        return word_embedding_lists


class TextEmbedding:
    __normalize = None
    __removepc = None
    _isfit = False
    _show_warnings = True

    def __init__(self, normalize=False, removepc=False, show_warnings=True):
        if type(self) == TextEmbedding:
            raise Exception("This class should not be implemented.")
        self.__normalize = normalize
        self.__removepc = removepc
        self._show_warnings = show_warnings

    def fit_transform(self, word_embedding_vocab, tokenlists):
        raise NotImplementedError

    def fit(self, tokenlists):
        raise NotImplementedError

    def transform(self):
        if self._isfit is False:
            raise Exception("The model is not fit yet!")

    def postprocess(self, mat):
        return mat

    def get_method(self):
        return self.__class__.__name__

    @staticmethod
    def normalize(matrix):
        norm = Normalizer(norm="l2", copy=True)
        return norm.transform(matrix)


class AverageEmbedding(TextEmbedding):

    def __init__(self, normalize=False, removepc=False, show_warnings=True):
        self._isfit = True
        super(AverageEmbedding, self).__init__(normalize=normalize, removepc=removepc, show_warnings=show_warnings)

    def fit_transform(self, word_embedding_vocab, tokenlists):
        return self.transform(word_embedding_vocab, tokenlists)

    def fit(self, tokenlists):
        pass

    def transform(self, word_embedding_vocab, tokenlists):
        super(AverageEmbedding, self).transform()

        word_embedding_lists = []
        for tokenlist in tokenlists:
            word_embedding_lists.append([word_embedding_vocab[x] for x in tokenlist])

        sentence_embeddings = [np.mean(x, axis=0) for x in word_embedding_lists]
        return super().postprocess(np.array(sentence_embeddings))


class TfidfWeighted(TextEmbedding):
    tfidf_model = None

    def __init__(self, normalize=False, removepc=False, show_warnings=True):
        super(TfidfWeighted, self).__init__(normalize=normalize, removepc=removepc, show_warnings=show_warnings)

    def fit_transform(self, word_embedding_vocab, tokenlists):
        self.fit(tokenlists)
        return self.transform(word_embedding_vocab, tokenlists)

    def fit(self, tokenlists):
        def dummy_fun(doc):
            return doc

        self.tfidf_model = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None, norm=None)
        self.tfidf_model.fit(tokenlists)
        self._isfit = True

    def transform(self, word_embedding_vocab, tokenlists):
        super(TfidfWeighted, self).transform()

        feature_vocab = self.tfidf_model.vocabulary_
        idf_vocab = self.tfidf_model.idf_

        # Get weighted average vectors
        sentence_embeddings = []
        for i, tokenlist in enumerate(tokenlists):
            tokencount = Counter(tokenlist)

            word_embedding_list = []
            weights = []
            for token, count in tokencount.items():
                word_embedding_list.append(word_embedding_vocab[token])
                weights.append(idf_vocab[feature_vocab[token]] * count)
            sentence_embeddings.append(np.average(word_embedding_list, axis=0, weights=weights))
        return super().postprocess(np.array(sentence_embeddings))


class SifWeighted(TextEmbedding):
    document_frequencies = None
    word2weight = None
    # SIF: a / (a + value/N)  -- where value=term_occurance and N=number_of_terms
    # IDF: a / (a + value/N)  -- where value=document_occurance and N=number_of_documents
    weighting_approach = "SIF"
    n_components = None
    pc = None

    def __init__(self, normalize=False, removepc=True, weighting_approach="SIF", n_components=1, show_warnings=True):
        super(SifWeighted, self).__init__(normalize=normalize, removepc=removepc, show_warnings=show_warnings)
        if weighting_approach not in {"SIF", "IDF"}:
            raise Exception("Unknown weightinig approach")
        self.weighting_approach = weighting_approach
        self.n_components = n_components

    def fit_transform(self, word_embedding_vocab, tokenlists):
        self.fit(word_embedding_vocab, tokenlists)
        return self.transform(word_embedding_vocab, tokenlists)

    def fit(self, word_embedding_vocab, tokenlists):
        self.word2weight = self.get_w2weight_dict(tokenlists)
        mat = self.get_embeddings(word_embedding_vocab, tokenlists)
        self.compute_principal_components(mat)
        self._isfit = True

    def get_embeddings(self, word_embedding_vocab, tokenlists):
        weights = [[] for _ in range(len(tokenlists))]
        word_embedding_lists = [[] for _ in range(len(tokenlists))]
        for i, tokenlist in enumerate(tokenlists):
            for token in tokenlist:
                if token in self.word2weight and token in word_embedding_vocab:
                    weights[i].append(self.word2weight[token])
                    word_embedding_lists[i].append(word_embedding_vocab[token])
                else:
                    if self._show_warnings:
                        warnings.warn("Token was not found in the vocabulary: {}".format(token))

        # Compute Weighted Sentence Embeddings
        sentence_embeddings = []
        for i, word_embedding_list in enumerate(word_embedding_lists):
            sentence_embedding = None
            for j, word_embedding in enumerate(word_embedding_list):
                if j == 0:
                    sentence_embedding = np.array(word_embedding) * weights[i][j]
                else:
                    sentence_embedding += np.array(word_embedding) * weights[i][j]
            sentence_embedding /= len(word_embedding_list)
            sentence_embeddings.append(sentence_embedding)
        sentence_embeddings = np.array(sentence_embeddings)
        return sentence_embeddings

    def transform(self, word_embedding_vocab, tokenlists):
        super(SifWeighted, self).transform()
        sentence_embeddings = self.get_embeddings(word_embedding_vocab, tokenlists)

        if self.n_components > 0:
            sentence_embeddings = self.remove_principal_component(sentence_embeddings, n_components=self.n_components)
        return sentence_embeddings

    def get_w2weight_dict(self, tokenlists, a=1e-3):
        tokens_flat = []

        # Compute term frequency
        for tokenlist in tokenlists:
            if self.weighting_approach == "IDF":
                tokenlist = set(tokenlist)
            tokens_flat.extend(list(tokenlist))
        word2weight = Counter(tokens_flat)

        # Normalize term frequency
        denominator = len(tokens_flat)
        if self.weighting_approach == "IDF":
            denominator = len(tokenlists)

        # Compute weights for each term
        for key, value in word2weight.items():
            word2weight[key] = a / (a + value / denominator)
        return dict(word2weight)

    def compute_principal_components(self, mat, n_components=1):
        if self.pc is None:
            svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=0)
            svd.fit(mat)
            self.pc = svd.components_
        else:
            raise Exception("PCs were already computed!")

    def remove_principal_component(self, mat, n_components=1):
        # Removal of Principal component
        assert self.pc is not None

        if n_components == 1:
            # Mathematical explanation:

            # mat: n x m
            # pc.transpose(): m x 1
            # pc: 1 x m
            # mat.dot(pc.transpose()): n x 1
            # mat.dot(pc.transpose()) * pc: n x m
            mat_pc = mat - mat.dot(self.pc.transpose()) * self.pc
        else:
            mat_pc = mat - mat.dot(self.pc.transpose()).dot(self.pc)
        return mat_pc
