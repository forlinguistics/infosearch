"""hw4 part 1"""
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from numpy.linalg import norm


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def bert_preproc(tokenizer, model, inp_name, emb_size, matr_name):
    """processing answer corpus and saving corpus vector"""
    # Sentences we want sentence embeddings for
    sentences = []
    with open(inp_name, encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            sentences.append(line[:-1])
    # Tokenize sentences
    encoded_input = tokenizer(sentences[0:5000], padding=True, truncation=True, max_length=24, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print(5000)
    np.save(matr_name, sentence_embeddings)
    for i in range(5000, emb_size, 5000):
        encoded_input = tokenizer(sentences[i:i + 5000], padding=True, truncation=True, max_length=24,
                                  return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = np.append(sentence_embeddings,
                                        mean_pooling(model_output, encoded_input['attention_mask']), 0)
    print(i + 5000)
    np.save(matr_name + '.npy', sentence_embeddings)


def bert_rank(input, tokenizer, model):
    """ranking top 10 closest to input results using bert"""
    matr = np.load('matr_bert.npy')
    sentences = [input]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_embedding = mean_pooling(model_output, encoded_input['attention_mask']).numpy()
    sim_vec = matr @ query_embedding.T / (norm(matr) * norm(query_embedding))
    with open('answers.txt', encoding='UTF-8') as f:
        answers = f.readlines()
    answers = np.array(answers)
    sorted_scores_indx = np.argsort(sim_vec, axis=0)[::-1]
    ranked = sorted_scores_indx.ravel()[0:10]
    for i in range(10):
        print(f'{i + 1}: {answers[ranked[i]]}')


if __name__ == '__main__':
    # Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    #    bert_preproc(tokenizer, model,'answers.txt', 50000 ,'matr_bert')
    bert_rank(input(), tokenizer, model)
