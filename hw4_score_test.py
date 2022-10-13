"""hw4 part 2"""
import numpy as np
from transformers import AutoTokenizer, AutoModel
from hw4 import bert_preproc
from preproc import preproc
from scipy import sparse
import pickle


def test_bert(top):
    """bert method score test with top parameter for checking top n result"""
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_preproc(tokenizer, model, 'questions.txt', 10000, 'matr_bert_q')
    matr_q = np.load('matr_bert_q.npy')
    matr_a = np.load('matr_bert.npy')
    sims = matr_q @ matr_a.T
    print('multiplied')
    sorted_scores_indx = np.argsort(sims, axis=0)
    r_count = 0
    score = 0
    for col in sorted_scores_indx.T:
        if r_count in col[-top:]:
            score += 1
        r_count += 1
    print(score / 10000)


def test_bm25(top):
    """bm25 method score test with top parameter for checking top n result"""
    corpus = []
    with open('q_corpus.txt', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            corpus.append(preproc(line[:-1]))

    matr = sparse.load_npz('matr.npz')
    vectorizer = pickle.load(open('bm25.pk', 'rb'))
    questions_count_vec = vectorizer.transform(corpus).toarray()
    print('transformed')
    vec = matr @ questions_count_vec.T
    print('multiplied')
    sorted_scores_indx = np.argsort(vec, axis=0)
    r_count = 0
    score = 0
    for col in sorted_scores_indx.T:
        if r_count in col[-top:]:
            score += 1
        r_count += 1
    print(score / 10000)


if __name__ == '__main__':
    test_bm25(10)  # скор почему-то 0,0003
    test_bert(10)  # скор почему-то 0,001
