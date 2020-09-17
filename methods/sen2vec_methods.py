import os
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_serving.client import BertClient
from methods.method import Method
from sklearn.metrics.pairwise import cosine_similarity
import helpers
import methods.aggwordembedding as te
from helperfunctions.util import clean
from sklearn.preprocessing import normalize

class EmbeddingMethod(Method):

    def __init__(self, preprocessor, caching=True, model_path=""):
        super().__init__(preprocessor, caching, model_path)
        self.sen2vec = None
        self.model = None

    def similarity(self, sentence1, sentence2):
        assert self.sen2vec is not None
        v1 = self.sen2vec[sentence1]
        v2 = self.sen2vec[sentence2]
        cosim = cosine_similarity([v1], [v2])[0][0]
        assert cosim < 1.01, f"Cosim: {cosim} {self.get_name()}"
        return cosim

    def prepare(self, unique_texts):
        raise NotImplementedError("implement me")

    def set_sen2vec(self, texts, veclist):
        assert len(texts) == len(set(texts))
        assert len(texts) == len(veclist)
        sen2vec = dict()
        for text, vec in zip(texts, veclist):
            sen2vec[text] = vec
        self.sen2vec = sen2vec


class TFIDFMethod(EmbeddingMethod):
    def prepare(self, unique_texts):
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        tokenlist = self.preprocess_all(unique_texts)
        vecs = vectorizer.fit_transform(tokenlist)
        self.set_sen2vec(unique_texts, vecs.A)


class Sen2VecMethod(EmbeddingMethod):
    def prepare(self, unique_texts):
        if self.model is None:
            import sent2vec
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(self.model_path)

        sentences_preprocessed = self.preprocess_all(unique_texts)
        vecs = self.model.embed_sentences(sentences_preprocessed)
        self.set_sen2vec(unique_texts, vecs)

class USE(EmbeddingMethod):
    def prepare(self, unique_texts):
        if self.model is None:
            import tensorflow_hub as hub
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        sentences_preprocessed = self.preprocess_all(unique_texts)
        vecs = self.model(sentences_preprocessed)
        vecs = vecs.numpy()  # Transform TF to numpy format
        self.set_sen2vec(unique_texts, vecs)


class SIF(EmbeddingMethod):
    def prepare(self, unique_texts):
        if self.vocabulary is None:
            self.vocabulary = helpers.load_pickle(self.model_path)

        model = te.SifWeighted()
        tokenlists = self.preprocess_all(unique_texts)
        tokens_clean = clean(tokenlists, self.vocabulary)
        assert len(tokens_clean) == len(unique_texts)

        # Distractors
        distractor_sentences = helpers.load_pickle("./sts/data/random_bio_sentences.p")
        print("Num Distractor Sentences:", len(distractor_sentences))
        distractor_tokenized = self.preprocess_all(distractor_sentences)
        distractors_clean = clean(distractor_tokenized, self.vocabulary)

        model.fit(word_embedding_vocab=self.vocabulary, tokenlists=distractors_clean + tokens_clean)

        vecs = model.transform(word_embedding_vocab=self.vocabulary, tokenlists=tokens_clean)
        self.set_sen2vec(unique_texts, vecs)

class WeightedW2V(EmbeddingMethod):
    def prepare(self, unique_texts):
        if self.vocabulary is None:
            self.vocabulary = helpers.load_pickle(self.model_path)

        model = te.TfidfWeighted()
        tokenlists = self.preprocess_all(unique_texts)
        tokens_clean = clean(tokenlists, self.vocabulary)
        assert len(tokens_clean) == len(unique_texts)

        # Distractors
        distractor_sentences = helpers.load_pickle("./sts/data/random_bio_sentences.p")
        print("Num Distractor Sentences:", len(distractor_sentences))
        distractor_tokenized = self.preprocess_all(distractor_sentences)
        distractors_clean = clean(distractor_tokenized, self.vocabulary)

        model.fit(tokenlists=distractors_clean + tokens_clean)

        vecs = model.transform(word_embedding_vocab=self.vocabulary, tokenlists=tokens_clean)
        self.set_sen2vec(unique_texts, vecs)

class AvgW2V(EmbeddingMethod):
    def prepare(self, unique_texts):
        if self.vocabulary is None:
            self.vocabulary = helpers.load_pickle(self.model_path)

        model = te.AverageEmbedding()
        tokenlists = self.preprocess_all(unique_texts)
        tokens_clean = clean(tokenlists, self.vocabulary)
        assert len(tokens_clean) == len(unique_texts)

        vecs = model.transform(word_embedding_vocab=self.vocabulary, tokenlists=tokens_clean)
        self.set_sen2vec(unique_texts, vecs)

class BERT(EmbeddingMethod):

    def prepare(self, texts):

        if self.model is None:
            # if "/" not in self.model_path:
            from sentence_transformers import SentenceTransformer, models
            try:
                self.model = SentenceTransformer(self.model_path)
            # else:
            # catch Exception:
            except Exception as e:
                word_embedding_model = models.Transformer(self.model_path)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                               pooling_mode_mean_tokens=True,
                                               pooling_mode_cls_token=False,
                                               pooling_mode_max_tokens=False)
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        texts_preprocessed = self.preprocess_all(texts)
        vecs = self.model.encode(texts_preprocessed)
        self.set_sen2vec(texts, vecs)

class Doc2Vec(EmbeddingMethod):

    def prepare(self, texts):
        start_alpha = 0.01
        infer_epoch = 1000
        if self.model is None:
            import gensim.models as g
            self.model = g.Doc2Vec.load(self.model_path)

        texts_preprocessed = self.preprocess_all(texts)
        # vecs = self.model.encode(texts_preprocessed)
        vecs = []
        for d in texts_preprocessed:
            vecs.append(self.model.infer_vector(d, alpha=start_alpha, steps=infer_epoch))
        self.set_sen2vec(texts, vecs)

def _bertservice_loader(model_name):
    if model_name == "biobert_v1.1_pubmed":
        return BertClient(port=5555, port_out=5556)
    return None

class BERTService(EmbeddingMethod):

    def prepare(self, texts):

        model_name = os.path.basename(self.model_path)
        self.model = _bertservice_loader(model_name)
        assert self.model is not None

        texts_preprocessed = self.preprocess_all(texts)
        vecs = self.model.encode(texts_preprocessed, is_tokenized=True)
        self.set_sen2vec(texts, vecs)

class InferSent(EmbeddingMethod):

    def prepare(self, texts):

        if self.model is None:
            from methods.fbinfersent.models import InferSent
            import torch
            model_version = 2

            INFERSENT_PATH = "/newstorage2/zlabinger/pretrained/infersent/"
            MODEL_PATH = INFERSENT_PATH + "infersent%s.pkl" % model_version
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
            self.model = InferSent(params_model)
            self.model.load_state_dict(torch.load(MODEL_PATH))
            W2V_PATH = INFERSENT_PATH + 'crawl-300d-2M.vec'
            self.model.set_w2v_path(W2V_PATH)

        texts_preprocessed = self.preprocess_all(texts)
        # texts_preprocessed = [t.lower() for t in texts]
        self.model.build_vocab(texts_preprocessed, tokenize=True)
        vecs = self.model.encode(texts_preprocessed, tokenize=True)
        vecs = normalize(vecs)

        self.set_sen2vec(texts, vecs)