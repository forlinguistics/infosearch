import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import argparse
from preproc import make_corpus

def get_index():
    """Creates index from corpus and prints out answers to the homework questions."""

    with open('corpus.txt', 'r', encoding='UTF-8') as f:
        corpus = f.readlines()
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(corpus)
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    words = np.array(vectorizer.get_feature_names_out())

    print('Самое частое слово: ' + words[np.argmax(matrix_freq)] + '\n')
    print('Одно из самых редких слов: ' + words[np.argmin(matrix_freq)] + '\n')
    print("Слова, встречающиеся во всех текстах: " + ", ".join(words[np.all(X.toarray(), axis=0)]) + '\n')
    names = ['моника', 'рэйчел', 'чендлер', 'фиби', 'росс', 'джой', 'джоуя']
    print("Частоты имен персонажей:")
    for i in names:
        if i == 'джоуя':
            print(f"джоуи : {matrix_freq[words == i][0]}")
        else:
            print(f"{i}: {matrix_freq[words == i][0]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", help="path to the files to be turned into a corpus")
    args = parser.parse_args()
    if 'corpus.txt' in os.listdir('./'):
        get_index()
    else:
        if args.files_path:
            make_corpus(args.files_path)
        else:
            make_corpus()