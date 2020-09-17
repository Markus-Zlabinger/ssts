## Implementation of semantic short-text similarity (SSTS) methods
This repository contains the code to reproduce our results on unsupervised semantic short-text similarity (SSTS) methods. You will find 
implementations of following SSTS methods:

##### Traditional Methods:
* Levensthein Distance
* TFIDF-based text similarity
##### Aggregated Word Embeddings:
* Averaged embeddings
* TFIDF-Weighted Average
* Smooth Inverse Frequency (SIF)

##### Contextualized Text Embeddings:
* SenBERT
* Universal Sentence Encoder (USE)
* InferSent
* Sent2Vec

## Experiment on Biomedical Sentence-2-Sentence Similarity

#### Prerequisites
Before running this experiment, go through following steps:
* Populate the *config.py* with your specific configuration. This includes, path to the benchmark dataset MedSTS (which you need to request from the author) and the paths to pretrained models. Note that the benchmark corpus BIOSSES is publicly available at https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html. Make sure to cite the corresponding papers when using either dataset for your research.
* Cached results of our experiments are available in the folder *data/cached_results*. If you want to produce results from scratch, delete the *cached_results* folder.

#### Results
Run the experiments using the *evaluation.py*. You should get results for two biomedical sentence2sentence similarity corpora: BIOSSES and MedSTS (also referred to as ClinicalSTS). The table should look like this:

| Method            | Model         | Preprocessing          |   biosses_pear |   medsts_all_pear |   AVG |
|-------------------|---------------|------------------------|----------------|-------------------|-------|
| avgw2v            | BioWord2Vec   | tokenizelower          |           0.61 |              0.72 |  0.66 |
| avgw2v            | BioWord2Vec   | tokenizelowerstopwords |           0.72 |              0.77 |  0.75 |
| bert              | ClinicalBERT  | idendity               |           0.65 |              0.69 |  0.67 |
| bert              | BioBERT       | idendity               |           0.78 |              0.58 |  0.68 |
| bert              | SciBERT       | idendity               |           0.60 |              0.68 |  0.64 |
| doc2vec           | Doc2Vec       | tokenizelower          |           0.81 |              0.75 |  0.78 |
| doc2vec           | Doc2Vec       | tokenizelowerstopwords |           0.80 |              0.76 |  0.78 |
| infersent         | InferSent 2.0 | idendity               |           0.49 |              0.65 |  0.57 |
| levenstheinmethod | -             | tokenizelower          |           0.55 |              0.64 |  0.60 |
| levenstheinmethod | -             | tokenizelowerstopwords |           0.64 |              0.69 |  0.66 |
| sen2vecmethod     | BioSent2Vec   | sentencelower          |           0.81 |              0.74 |  0.78 |
| sen2vecmethod     | BioSent2Vec   | sentencelowerstopwords |           0.81 |              0.77 |  0.79 |
| sif               | BioWord2Vec   | tokenizelower          |           0.79 |              0.75 |  0.77 |
| sif               | BioWord2Vec   | tokenizelowerstopwords |           0.78 |              0.76 |  0.77 |
| tfidfmethod       | -             | tokenizelower          |           0.74 |              0.70 |  0.72 |
| tfidfmethod       | -             | tokenizelowerstopwords |           0.74 |              0.73 |  0.74 |
| use               | USE 4.0       | idendity               |           0.66 |              0.72 |  0.69 |
| weightedw2v       | BioWord2Vec   | tokenizelower          |           0.73 |              0.75 |  0.74 |
| weightedw2v       | BioWord2Vec   | tokenizelowerstopwords |           0.76 |              0.77 |  0.76 |

## Contact Us
If you have any further questions, open a new issue for this repository.

## Publication
The code from this library is based on two of our papers. The first paper uses unsupervised SSTS methods to compute the similarity between biomedical sentences.
```
@inproceedings{zlabinger2020crowd,
  title={Effective Crowd-Annotation of Participants, Interventions, and Outcomes in the Text of Clinical Trial Reports},
  author={Zlabinger, Markus and Sabou, Marta and Hofstätter, Sebastian and Hanbury, Allan},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP-Findings 2020)},
  year={2020},
}
```
And the second paper uses unsupervised SSTS methods to find similar questions.
```
@inproceedings{zlabinger2019efficient,
  title={Efficient answer-annotation for frequent questions},
  author={Zlabinger, Markus and Rekabsaz, Navid and Zlabinger, Stefan and Hanbury, Allan},
  booktitle={International Conference of the Cross-Language Evaluation Forum for European Languages},
  pages={126--137},
  year={2019}
}
```

