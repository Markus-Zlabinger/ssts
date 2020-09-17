import helperfunctions.util as util
from config import *
from scipy.stats.stats import pearsonr, spearmanr
from methods.sen2vec_methods import *
from methods.method_traditional import *
from preprocessing.preprocessor_funcs import *
from tabulate import tabulate
import numpy as np

def evaluate(method, datasetname):
    # Load data
    data = util.load_dataset(datasetname)

    # Compute similarity
    sentence_pairs = list(zip(data["sentence1"].to_list(), data["sentence2"].to_list()))
    y_pred = method.all_similarity(sentence_pairs, datasetname=datasetname)
    y_true = list(data["score"])
    assert len(y_pred) == len(y_true)
    return y_pred, y_true


def run_evaluation():


    usecache = True

    methods = [
        TFIDFMethod(TokenizeLower(), caching=usecache),
        TFIDFMethod(TokenizeLowerStopwords(), caching=usecache),

        LevenstheinMethod(preprocessor=TokenizeLower(), caching=usecache),
        LevenstheinMethod(preprocessor=TokenizeLowerStopwords(), caching=usecache),


        AvgW2V(TokenizeLower(), caching=usecache, model_path=path_bioword2vec()),
        AvgW2V(TokenizeLowerStopwords(), caching=usecache, model_path=path_bioword2vec()),

        WeightedW2V(TokenizeLower(), caching=usecache, model_path=path_bioword2vec()),
        WeightedW2V(TokenizeLowerStopwords(), caching=usecache, model_path=path_bioword2vec()),


        SIF(TokenizeLower(), caching=usecache, model_path=path_bioword2vec()),
        SIF(TokenizeLowerStopwords(), caching=usecache, model_path=path_bioword2vec()),

        Doc2Vec(TokenizeLower(), caching=usecache, model_path="D:/bigdata/doc2vec/enwiki_dbow/doc2vec.bin"),
        Doc2Vec(TokenizeLowerStopwords(), caching=usecache, model_path="D:/bigdata/doc2vec/enwiki_dbow/doc2vec.bin"),


        BERT(Idendity(), caching=usecache, model_path='allenai/scibert_scivocab_uncased'),
        BERT(Idendity(), caching=usecache, model_path='monologg/biobert_v1.1_pubmed'),
        BERT(Idendity(), caching=usecache, model_path='emilyalsentzer/Bio_ClinicalBERT'),

        USE(Idendity(), caching=usecache, model_path="USE"),

        InferSent(Idendity(), caching=True, model_path="InferSent"),

        Sen2VecMethod(SentenceLower(), caching=usecache, model_path=path_biosent2vec()),
        Sen2VecMethod(SentenceLowerStopwords(), caching=usecache, model_path=path_biosent2vec()),
    ]

    methods = sorted(methods, key=lambda x: (x.get_name().lower(), x.get_model_name()))
    method_names = set()
    for method in methods:
        caching_name = method.get_caching_name()
        assert caching_name not in method_names, caching_name
        method_names.add(caching_name)

    datasets = ["biosses", "medsts_all"]
    table = []
    for method in methods:
        d = dict()
        d["Method"] = method.get_name().lower()
        d["Model"] = util.convert_model_name(method.get_model_name().strip())
        d["Preprocessing"] = method.get_preprocessor().lower()
        evals_p = []
        evals_s = []
        for dataset in datasets:
            y_pred, y_true = evaluate(method, dataset)

            pearson = pearsonr(y_pred, y_true)[0]
            spearman = spearmanr(y_pred, y_true)[0] # Could also report Spearman

            d[dataset + "_pear"] = "{:.2f}".format(pearson)
            evals_p.append(pearson)
            evals_s.append(spearman)
        d["AVG"] = "{:.2f}".format(np.mean(evals_p))
        table.append(d)
    print(tabulate(table, headers="keys", tablefmt="github", floatfmt=".2f"))

run_evaluation()