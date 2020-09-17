from collections import Iterable
import os
import pandas as pd
from helperfunctions.namedict import namedict
from config import path_medsts_dataset

def write_textlist(list_of_texts, path):
    assert isinstance(list_of_texts, Iterable)
    assert len(list_of_texts) > 0
    assert type(list_of_texts[0]) == str

    with open(path, "w+") as f:
        f.write("\n".join(list_of_texts))

def createdir(path):
    if not os.path.isdir(path):  # Only create folder if it does not exist yet
        os.makedirs(path)

def load_textlist(path):
    list_of_texts = []
    with open(path, "r") as f:
        for line in f:
            list_of_texts.append(line.strip())
    return list_of_texts

def load_dataset(datasetname="", verbose=False):
    data = None
    if datasetname == "medsts":
        data = _load_medsts()
    if datasetname == "biosses":
        data = _load_biosses()
    if datasetname == "medsts_test":
        data = _load_medsts_test()
    if datasetname == "medsts_all":
        data = _load_medsts_all()
    assert data is not None
    if verbose:
        print("Loaded #Entries:", len(data))
    return data


def _load_biosses():
    path = "./data/biosses/all.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df[["sentence1", "sentence2", "score"]].copy()
    return _prepare_df(df)


def _load_medsts():
    path = path_medsts_dataset() + "clinicalSTS.train.txt"
    df = pd.read_csv(path, sep="\t", names=["sentence1", "sentence2", "score"])
    return _prepare_df(df)


def clean(tokenlists, vocabulary):
    tokenlists_clean = []

    for tokenlist in tokenlists:
        curlist = []
        for token in tokenlist:
            if token in vocabulary:
                curlist.append(token)
        if len(curlist) > 0:
            tokenlists_clean.append(curlist)
    assert len(tokenlists_clean) > 0
    return tokenlists_clean


def _load_medsts_test():
    path = path_medsts_dataset() + "clinicalSTS.test.txt"
    path_scores = path_medsts_dataset() + "clinicalSTS.test.gs.sim.txt"
    df = pd.read_csv(path, sep="\t", names=["sentence1", "sentence2"])
    with open(path_scores) as f:
        scores = f.read().split("\n")
    scores = [float(s) for s in scores]
    assert len(scores) == len(df)
    df["score"] = scores
    return _prepare_df(df)


def _load_medsts_all():
    df1 = _load_medsts()
    df2 = _load_medsts_test()
    return _prepare_df(df1.append(df2))


def _prepare_df(df):
    assert sum(df.duplicated()) == 0, "There are duplicated rows"

    assert len(df.columns) == 3
    for colname in df.columns:
        assert colname in {"sentence1", "sentence2", "score"}

    # Strip strings
    df["sentence1"] = df["sentence1"].str.strip()
    df["sentence2"] = df["sentence2"].str.strip()
    return df.copy()


def get_unique_sentences(df):
    sentences = df["sentence1"].to_list() + df["sentence2"].to_list()
    return list(set(sentences))



def convert_model_name(name):
    endinglist = [".p", ".bin"]
    for tmp in endinglist:
        if name.endswith(tmp):
            return namedict[name[:-len(tmp)]]
    if name == "":
        return "-"
    return namedict[name]

def start_bert_server():
    from bert_serving.server.helper import get_args_parser
    from bert_serving.server import BertServer
    args = get_args_parser().parse_args(['-model_dir', 'YOUR_MODEL_PATH_HERE',
                                         '-port', '5555',
                                         '-port_out', '5556',
                                         '-num_worker',
                                         '-cpu'])
    server = BertServer(args)
    server.start()
