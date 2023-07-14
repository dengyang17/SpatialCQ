import os
import logging
import torch
import pickle
import json
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    mode = args.set_name if evaluate else 'train'
    print(mode)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'spatial_cnp_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features['source_ids']))
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer, mode)
        print("Loaded number of instance:", len(features['source_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, tokenizer, mode):

    clarifying_need_map = {'Yes': 'False', 'No': 'True'}
    idx2color = {1: tokenizer.encode('blue')[1], 2: tokenizer.encode('green')[1], 3: tokenizer.encode('red')[1], 4: tokenizer.encode('orange')[1], 5: tokenizer.encode('purple')[1], 6: tokenizer.encode('yellow')[1], 7: tokenizer.encode('builder')[1]}
    path = '{}/{}.json'.format(args.data_dir, mode)
    print('tokenizing {}'.format(path))
    source_len = []
    mrr_scores = []
    with open(path, 'r', encoding='utf-8') as infile:
        source_ids = []
        target_ids = []
        sample_ids = []
        state_nodes = []
        state_adjs = []
        idx = 0
        #world_states = []
        data = json.load(infile)
        for sample in data:

            world_state = sample['world_state'].copy()
            avatar_state = [min(int(sample['avatar_info'][0]),10), min(int(sample['avatar_info'][1]),8), min(int(sample['avatar_info'][2]),10), 7]
            world_state.append(avatar_state)

            state_node = [idx2color[x[3]] for x in world_state]
            updown_adj = np.zeros((len(state_node),len(state_node)))
            eastwest_adj = np.zeros((len(state_node),len(state_node)))
            northsouth_adj = np.zeros((len(state_node),len(state_node)))

            for i in range(len(state_node)-1):
                for j in range(i+1,len(state_node)):
                    updown_adj[i][j] = world_state[i][2] - world_state[j][2]
                    updown_adj[j][i] = world_state[j][2] - world_state[i][2]
                    eastwest_adj[i][j] = world_state[i][1] - world_state[j][1]
                    eastwest_adj[j][i] = world_state[j][1] - world_state[i][1]
                    northsouth_adj[i][j] = world_state[i][0] - world_state[j][0]
                    northsouth_adj[j][i] = world_state[j][0] - world_state[i][0]

            adj = [updown_adj.tolist(), eastwest_adj.tolist(), northsouth_adj.tolist()]

            source_id = tokenizer.encode('[instruction]' + sample['instruction'])
            source_len.append(len(source_id))

            state_nodes.append(state_node)
            state_adjs.append(adj)

            if sample['clarifying_need'] == 'Yes':
                target_ids.append(0)
            else:
                target_ids.append(1)
                
            source_ids.append(source_id[-args.max_seq_length:])
            state_nodes.append(state_node)
            state_adjs.append(adj)
            
            idx += 1
    
    print("max/avg source len: %f, %f" % (max(source_len), sum(source_len)/float(len(source_len))))
    
    return {'source_ids':source_ids, 'target_ids':target_ids, 'state_nodes':state_nodes, 'state_adjs':state_adjs}




def mrr(ranked_list, gt):
    score = 0.0
    for rank, item in enumerate(ranked_list):
        if item[0] == gt:
            score = 1.0/(rank+1.0)
            break
    return score