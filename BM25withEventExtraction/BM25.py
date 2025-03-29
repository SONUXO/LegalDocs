import os, sys, json, time, re, string, codecs, random, numpy as np, pandas as pd
import evaluate_at_K
from tqdm.notebook import tqdm_notebook
import pickle as pkl
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class BM25(object):
    def __init__(self, b=0.7, k1=1.6, n_gram:int = 1):
        self.n_gram = n_gram
        self.vectorizer = TfidfVectorizer(max_df=.65, min_df=1,
                                  use_idf=True, 
                                  ngram_range=(n_gram, n_gram))
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        start_time = time.perf_counter()
        print(f"Fitting tf_idf vectorizer")

        y = self.vectorizer.fit_transform(X)  # Combine fit and transform
        self.avdl = y.sum(1).mean()

        print(f"Finished tf_idf vectorizer, time : {time.perf_counter() - start_time:0.3f} sec")

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        if not q.strip():  # Handle empty queries
            return np.zeros(len(X))  

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        
        if q.nnz == 0:  # If query has no tokens in vocabulary
            return np.zeros(len(X))  

        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1
    

# Load Data
candidate_query = pd.read_csv('train_candidates.csv')  # Added `.csv`
corpus = list(candidate_query["text"])  # List of candidate case texts
citation_names = list(candidate_query["id"].astype(str))  # List of case IDs (string format)

train_query = pd.read_csv('train_queries.csv')  # Added `.csv`
query_corpus = list(train_query["text"])  # List of query case texts
query_names = list(train_query["id"].astype(str))  # List of query case IDs (string format)

# Convert `relevant_candidates` into a dictionary
true_labels = {}
for index, row in train_query.iterrows():
    query_id = str(row["id"])  
    relevant_cases = row["relevant_candidates"]

    if isinstance(relevant_cases, str):  
        relevant_cases = ast.literal_eval(relevant_cases)  
    
    if isinstance(relevant_cases, float):  
        relevant_cases = []
    
    true_labels[query_id] = list(map(str, relevant_cases))  

# Train BM25 Model
bm25 = BM25(n_gram=1)
bm25.fit(corpus)

# Compute BM25 scores
bm_25_results_dict = {}
for i in range(len(query_corpus)):
    qu = query_corpus[i]
    qu_n = query_names[i]
    doc_scores = bm25.transform(qu, corpus)
    bm_25_results_dict[qu_n] = {citation_names[j]: doc_scores[j] for j in range(len(doc_scores))}

# Convert results to DataFrame
bm_25_results_list = []
for query_id, scores in bm_25_results_dict.items():
    for doc_id, score in scores.items():
        bm_25_results_list.append([query_id, doc_id, score])
bm_25_results_df = pd.DataFrame(bm_25_results_list, columns=['query_case_id', 'candidate_case_id', 'bm25_score'])
