import argparse
import numpy as np
from preproc import make_corpus, preproc
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from scipy.spatial.distance import cosine


# считаем
def get_index():
    """Creates index from corpus and prints out answers to the homework questions."""
    with open('corpus.txt', 'r', encoding='UTF-8') as f:
        corpus = f.readlines()
    vectorizer = TfidfVectorizer()
    matr = vectorizer.fit_transform(corpus).toarray()
    np.save('matr.npy', matr)
    with open('tfidf.pk', 'wb') as fin:
        pickle.dump(vectorizer, fin)


def query_proc(query, corp_matr):
    vectorizer = pickle.load(open("tfidf.pk", 'rb'))
    query_vec = vectorizer.transform([preproc(query)]).toarray()
    cosine_sims = np.apply_along_axis(lambda x: sim(x, query_vec[0]), 1, corp_matr)
    return (cosine_sims)


def sim(vec1, vec2):
    return (1 - cosine(vec1, vec2))


def tf_idf_sims(query):
    f_names = []
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(curr_dir + '/friends-data/'):
        for name in files:
            f_names.append(name)
    matr = np.load('matr.npy')
    a = list(zip(f_names, query_proc(query, matr)))
    a.sort(key=lambda x: x[1], reverse=True)
    return (a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", help="path to the files to be turned into a corpus")
    args = parser.parse_args()
    if not ('matr.npy' in os.listdir('./') and 'tfidf.pk' in os.listdir('./')):
        if 'corpus.txt' in os.listdir('./'):
            get_index()
        else:
            if args.files_path:
                make_corpus(args.files_path)
            else:
                make_corpus()
        get_index()
    print("Enter the query: ")
    inp_string = input()
    e_list = tf_idf_sims(inp_string)
    for e in e_list[0:10]:
        print(f'{e[0]}  sim:{e[1]:.3f}')
