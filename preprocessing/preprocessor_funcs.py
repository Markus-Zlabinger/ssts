from gensim.parsing.preprocessing import *
import gensim

class Preprocessing:
    def __init__(self):
        pass

    def preprocess(self, text):
        raise NotImplementedError("Implement plz")

    def preprocess_all(self, texts):
        return [self.preprocess(t) for t in texts]

class Idendity(Preprocessing):
    def preprocess(self, text):
        return text

class PreprocessGemsin(Preprocessing):
    FILTER = None
    JOIN_TOKENS = False

    def __init__(self):
        super().__init__()

    def preprocess(self, text):
        assert self.FILTER is not None
        tokens = preprocess_string(text, self.FILTER)
        if self.JOIN_TOKENS:
            return " ".join(tokens)
        return tokens


"""
For Tokens
"""

class Tokenize(PreprocessGemsin):
    FILTER = [strip_multiple_whitespaces]

class TokenizeLower(PreprocessGemsin):
    FILTER = [lambda x: x.lower(), strip_punctuation,
              strip_multiple_whitespaces, lambda x: strip_short(x, 2)]


class TokenizeLowerStopwords(PreprocessGemsin):
    FILTER = [lambda x: x.lower(), strip_punctuation,
              strip_multiple_whitespaces, lambda x: strip_short(x, 2), remove_stopwords]


class TokenizeLowerStopwordsStem(PreprocessGemsin):
    FILTER = [lambda x: x.lower(), strip_punctuation,
              strip_multiple_whitespaces, lambda x: strip_short(x, 2), remove_stopwords, stem_text]


"""
For Sentences
"""


class SentenceLower(TokenizeLower):
    JOIN_TOKENS = True


class SentenceLowerStopwords(TokenizeLowerStopwords):
    JOIN_TOKENS = True


class SentenceLowerStopwordsStem(TokenizeLowerStopwordsStem):
    JOIN_TOKENS = True

class SentenceStopwords(Preprocessing):
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    FILTERS = [strip_punctuation,
               strip_multiple_whitespaces, lambda x: strip_short(x, 2)]

    def preprocess(self, text):
        tokens = preprocess_string(text, self.FILTERS)
        tokens = [t for t in tokens if t.lower() not in self.stop_words]
        return ' '.join(tokens)