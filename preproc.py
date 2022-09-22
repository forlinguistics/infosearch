import os
import string
from natasha import Doc, Segmenter
from nltk.corpus import stopwords
import pymorphy2


def preproc(text,stop_w = True):
    """"Preprocesses given file and returns its words as a string."""

    morph = pymorphy2.MorphAnalyzer()
    segmenter = Segmenter()
    st = stopwords.words("russian")
    text_without_punct = text.translate(str.maketrans('', '', string.punctuation))
    doc = Doc(text_without_punct)
    doc.segment(segmenter)
    tokens = []
    for token in doc.tokens:
        tokens.append(morph.parse(token.text)[0].normal_form)
    if stop_w:
        lemmatized_text = ' '.join([token for token in tokens if not token in st]).lower()
    else:
        lemmatized_text = ' '.join([token for token in tokens]).lower()
    return (lemmatized_text)


def make_corpus(path='\\friends-data\\'):
    """Creates corpus .txt file from files in stated directory."""

    corpus = []
    curr_dir = os.getcwd()
    if path == '\\friends-data\\':
        curr_dir = os.getcwd() + path
    else:
        curr_dir = path
    i = 0
    for root, dirs, files in os.walk(curr_dir):
        for name in files:
            with open(os.path.join(root, name), 'r', encoding='UTF-8') as f:
                corpus.append(preproc(f.read()))
                print(i)
                i += 1
    with open('corpus.txt', 'w', encoding='UTF-8') as f:
        for i in corpus:
            f.write(i + '\n')
