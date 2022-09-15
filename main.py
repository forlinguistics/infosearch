import os
import string
from natasha import Doc, Segmenter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pymorphy2
import argparse


def preproc(text):
    """"Preprocesses given file and returns its words as a string."""

    morph = pymorphy2.MorphAnalyzer()
    segmenter = Segmenter()
    st = stopwords.words("russian")
    text_without_punct = text.translate(str.maketrans('', '', string.punctuation))
    doc = Doc(text_without_punct[1:])
    doc.segment(segmenter)
    tokens = []
    for token in doc.tokens:
        tokens.append(morph.parse(token.text)[0].normal_form)
    lemmatized_text = ' '.join([token for token in tokens if not token in st]).lower()
    return (lemmatized_text)


def make_corpus(path = '\\friends-data\\'):
    """Creates corpus .txt file from files in stated directory."""

    corpus = []
    curr_dir = os.getcwd()
    i = 0
    for root, dirs, files in os.walk(curr_dir + path):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='UTF-8') as f:
                corpus.append(preproc(f.read()))
                print(i)
                i += 1
    with open('corpus.txt', 'w', encoding='UTF-8') as f:
        for i in corpus:
            f.write(i + '\n')


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