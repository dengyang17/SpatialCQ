import re
import os
import argparse
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from rank_bm25 import BM25Okapi



def bm25_rank(set_name):
    bm25_rank_lists = []
    with open('../dataset/question_bank.json','r') as infile:
        qbank = json.load(infile)
    with open('../dataset/{}.json'.format(set_name),'r') as infile:
        data = json.load(infile)
    ref = [x for x in data if x['clarifying_need'] == 'No']
    for sample in ref:
        sample['qbank'] = eval('[' + sample['qbank'] + ']')
        corpus = [qbank[x].lower() for x in sample['qbank'] if x in qbank]
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        query = sample['instruction'].lower()
        tokenized_query = query.split(" ")

        doc_scores = bm25.get_scores(tokenized_query)

        ranked_list = sorted(zip(list(range(len(corpus))), doc_scores, corpus), key=lambda x:x[1], reverse=True)
        
        bm25_rank_lists.append(ranked_list)

    return bm25_rank_lists


def calculate(targets, scores, set_name):

    precision = precision_score(gold_goals, pred_goals, average='macro', zero_division=0)
    recall = recall_score(gold_goals, pred_goals, average='macro', zero_division=0)
    f1 = f1_score(gold_goals, pred_goals, average='macro', zero_division=0)
    return [precision,recall,f1]


if __name__ == '__main__':
    sample_ids, targets, scores = [], [], []
    with open('output/iglu/roberta/iglu_test_roberta-large.score','r') as infile:
        for line in infile:
            items = line.strip('\n').split('\t')
            sample_ids.append(int(items[0]))
            targets.append(int(items[1]))
            scores.append(float(items[2]))
    
    
    print(calculate(sample_ids, targets, scores, 'test'))