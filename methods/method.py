import os
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import *
from helperfunctions.util import createdir, write_textlist, load_textlist

class Method:
    stop_words = set(stopwords.words('english'))

    def __init__(self, preprocessor, caching=True, model_path=""):
        self.preprocess = preprocessor.preprocess
        self.preprocess_all = preprocessor.preprocess_all
        self.preprocessor = preprocessor
        self.caching = caching
        self.model_path=model_path
        self.vocabulary=None

    def similarity(self, sentence1, sentence2):
        raise NotImplementedError("Implement me plz")

    def _cleanname(self, name):
        name = name.strip()
        if name.endswith(".p"):
            name = name[:-2]
        if name.endswith(".txt"):
            name = name[:-3]
        return name.strip()

    def all_similarity(self, sentence_pairs, datasetname):
        outpath = f"./data/cached_results/{self._cleanname(self.get_caching_name())}/{self._cleanname(datasetname)}/"
        outfile = outpath + "results.txt"
        if self.caching:
            if os.path.isfile(outfile):
                return [float(x) for x in load_textlist(outfile)]

        # Non caching
        print("Create from scratch:", outfile)
        unique_sentences = []
        for sentence_pair in sentence_pairs:
            unique_sentences.extend(sentence_pair)
        unique_sentences = list(set(unique_sentences))
        self.prepare(unique_sentences)
        y_pred = []
        for sen1, sen2 in sentence_pairs:
            y_pred.append(self.similarity(sen1, sen2))

        # Store
        createdir(outpath)
        write_textlist([str(x) for x in y_pred], outfile)
        return y_pred

    def prepare(self, unique_texts):
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_caching_name(self):
        name = self.get_name() + " " + self.get_preprocessor() + " " + self.get_model_name()
        return name.replace("/", "").replace("\\", "").strip()

    def get_preprocessor(self):
        return self.preprocessor.__class__.__name__

    def get_model_name(self):
        return os.path.basename(self.model_path).strip()

    @staticmethod
    def tokenize_base(texts):
        _BM25_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,
                         strip_multiple_whitespaces, remove_stopwords, lambda x: strip_short(x, 2), stem_text]
        return [preprocess_string(text, _BM25_FILTERS) for text in texts]

    @staticmethod
    def tokenize_text(text):
        _BM25_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,
                         strip_multiple_whitespaces, remove_stopwords, lambda x: strip_short(x, 2), stem_text]
        return preprocess_string(text, _BM25_FILTERS)

    @staticmethod
    def remove_stopwords(text):
        _BM25_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation,
                         strip_multiple_whitespaces, remove_stopwords, lambda x: strip_short(x, 2), stem_text]
        return preprocess_string(text, _BM25_FILTERS)