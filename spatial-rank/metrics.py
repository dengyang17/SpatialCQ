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


def calculate(sample_ids, targets, scores, set_name, bm25=True, topk=200):
    def mrr(ranked_list):
        score = [0.0, 0.0, 0.0, 0.0]
        for rank, item in enumerate(ranked_list):
            if item[0] == 1:
                if rank < 5:
                    score = [1.0/(rank+1.0) for _ in range(4)]
                    break
                if rank < 10:
                    score = [0] + [1.0/(rank+1.0) for _ in range(3)]
                    break
                if rank < 20:
                    score = [0, 0] + [1.0/(rank+1.0) for _ in range(2)]
                    break
                score = [0, 0, 0] + [1.0/(rank+1.0)]
                break
        return score

    
    mrr_scores = []
    score_lists = []
    idx = -1
    for sid, tgt, score in zip(sample_ids, targets, scores):
        if sid != idx:
            if idx >= 0:
                score_lists.append(score_list)
                #mrr_scores.append(mrr(ranked_list))
            idx += 1
            score_list = []
        score_list.append((tgt, score))
    
    score_lists.append(score_list)
    
    if bm25:
        bm25_rank_lists = bm25_rank(set_name)
        assert len(score_lists) == len(bm25_rank_lists)
        
        topk_score_lists = []
        for score_list, bm25_rank_list in zip(score_lists, bm25_rank_lists):
            #print(len(score_list), len(bm25_rank_list))
            assert len(score_list) == len(bm25_rank_list)
            bm25_list = set()
            for t in bm25_rank_list[:topk]: 
                bm25_list.add(t[0])
            topk_score_list = []
            for i, score in enumerate(score_list):
                if i in bm25_list:
                    topk_score_list.append(score)
            topk_score_lists.append(topk_score_list)

        score_lists = topk_score_lists.copy()

    for score_list in score_lists:
        rank_list = sorted(score_list, key=lambda x:x[1])
        mrr_scores.append(mrr(rank_list))

    mrr_5 = sum([x[0] for x in mrr_scores])/len(mrr_scores)
    mrr_10 = sum([x[1] for x in mrr_scores])/len(mrr_scores)
    mrr_20 = sum([x[2] for x in mrr_scores])/len(mrr_scores)
    mrr = sum([x[3] for x in mrr_scores])/len(mrr_scores)
    return [mrr_5,mrr_10,mrr_20,mrr]


if __name__ == '__main__':
    sample_ids, targets, scores = [], [], []
    with open('output/iglu/roberta/iglu_test_roberta-large.score','r') as infile:
        for line in infile:
            items = line.strip('\n').split('\t')
            sample_ids.append(int(items[0]))
            targets.append(int(items[1]))
            scores.append(float(items[2]))
    
    
    print(calculate(sample_ids, targets, scores, 'test'))