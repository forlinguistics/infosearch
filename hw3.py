"""bm25"""
from preproc import preproc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse


def preproc_bm25():
    """calculates matrices for further use in bm25 calculation"""
    corpus = []
    with open('q_corpus.txt', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            corpus.append(line[:-1])
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(corpus)  # для индексации запроса
    x_tfidf_vec = tfidf_vectorizer.fit_transform(corpus)  # матрица для idf

    tf = x_tfidf_vec
    idf = tfidf_vectorizer.idf_
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    A = sparse.csr_matrix(tf.multiply(idf) * (k + 1))
    B_1 = (k * (1 - b + b * len_d / avdl))
    for i, j in zip(A.nonzero()[0], A.nonzero()[1]):
        A[i, j] = A[i, j] / (tf[i, j] + B_1[i])
    sparse.save_npz('matr', A, compressed=True)
    with open('bm25.pk', 'wb') as fin:
        pickle.dump(count_vectorizer, fin)


def bm25(query):
    """prints top ten closest matches for query using bm25 metric"""
    matr = sparse.load_npz('matr.npz')
    vectorizer = pickle.load(open('bm25.pk', 'rb'))
    query_count_vec = vectorizer.transform([preproc(query)]).toarray()
    vec = matr @ query_count_vec.T
    with open('questions.txt', encoding='UTF-8') as f:
        answers = f.readlines()
    answers = np.array(answers)
    sorted_scores_indx = np.argsort(vec, axis=0)[::-1]
    ranked = sorted_scores_indx.ravel()[0:10]
    for i in range(10):
        print(f'{i + 1}: {answers[ranked[i]]}')


if __name__ == '__main__':
    print("Enter the query: ")
    inp_string = input()
    bm25(inp_string)
