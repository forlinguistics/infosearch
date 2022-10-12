import json
from preproc import preproc

with open('data.jsonl', 'r', encoding='UTF-8') as f:
    corpus = list(f)[:50000]
with open('questions.txt', 'w', encoding='UTF-8') as f:
    i = 0
    for doc in corpus:
        answers = json.loads(doc)['answers']
        if answers:
            max_ans = max(answers, key=lambda x: int(x['author_rating']['value'] if not x['author_rating']['value'] =='' else 0))['text']
        f.write(max_ans+'\n')
        if i % 10 == 0:
            print(i)
        i+=1