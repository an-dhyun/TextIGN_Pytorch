import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
from gensim.models import Word2Vec 

dataset = 'nsmc'
window_size = 3
weighted_graph = False
truncate = False  
MAX_TRUNC_LEN = 350
word_embeddings_dim = 200
word_embeddings = Word2Vec.load("./kor_embedding/ko.bin") # 사전훈련된 word2vec 모델 호출

print('1. loading raw data')
doc_name_list = []
doc_train_list = []
doc_test_list = []

with open('data/' + dataset + '.txt', 'r') as f:
    for line in tqdm(f.readlines()):
        if line=='\n': pass
        else: 
            doc_name_list.append(line.strip())
            temp = line.split("\t")
            temp[2] = temp[2].replace('\n', '')

            if temp[1].find('test') != -1:
                doc_test_list.append(line.strip())
            elif temp[1].find('train') != -1:
                doc_train_list.append(line.strip())

print("2. load raw text")
doc_content_list = []

with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        doc_content_list.append(line.strip())

print("3. map and shuffle")
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

ids = train_ids + test_ids

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for i in ids:
    shuffle_doc_name_list.append(doc_name_list[int(i)])
    shuffle_doc_words_list.append(doc_content_list[int(i)])

print("4. build corpus vocabulary")
word_set = set()

for doc_words in shuffle_doc_words_list:
    doc_words = doc_words.split(' ')
    word_set.update(doc_words)

vocab = list(word_set)
vocab_size = len(vocab)
with open('./data/' + dataset + '_vocab.txt', 'w') as f:
    for i in vocab: f.write(i+'\n')
    
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)

print("5. build label list")
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(int(temp[2]))
label_list = list(label_set)

print("6. select 90% training set")
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_ids)

print("7. build graph function")
def build_graph(start, end):
    x_adj = []
    x_feature = []
    y = []
    doc_len_list = []
    vocab_set = set()

    for i in tqdm(range(start, end)):
        doc_words = shuffle_doc_words_list[i]
        doc_words = doc_words.split(' ')
        
        # 경우에 따라 trunc 진행
        if truncate:
            doc_words = doc_words[:MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)
        vocab_set.update(doc_vocab)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # window를 슬라이드하며 window단위 텍스트 생성
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)
    
        # 단어쌍 연산
        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []
        
        # adj(인접행렬), feature(임베딩값) 생성
        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[vocab[p]])
            col.append(doc_word_id_map[vocab[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            try:
                features_tmp = word_embeddings.wv[k]
                if len(features_tmp)==0: 
                    print(k)
                    break
            except:
                features_tmp = oov[k]
                if len(features_tmp)==0: 
                    print(k)
                    break
            features.append(features_tmp)

        x_adj.append(adj)
        x_feature.append(features)

    # 정답라벨값 원핫인코딩
    for i in range(start, end):
        doc_meta = shuffle_doc_name_list[i]
        temp = doc_meta.split('\t')
        label = int(temp[2])
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)

    return x_adj, x_feature, y, doc_len_list, vocab_set

print('8. building graphs for training')
x_adj, x_feature, y, _, _ = build_graph(start=0, end=real_train_size)
print('9. building graphs for training + validation')
allx_adj, allx_feature, ally, doc_len_list_train, vocab_train = build_graph(start=0, end=train_size)
print('10. building graphs for test')
tx_adj, tx_feature, ty, doc_len_list_test, vocab_test = build_graph(start=train_size, end=train_size + test_size)
doc_len_list = doc_len_list_train + doc_len_list_test

print('11. show statistics')
print('max_doc_length', max(doc_len_list), 'min_doc_length', min(doc_len_list),
      'average {:.2f}'.format(np.mean(doc_len_list)))
print('training_vocab', len(vocab_train), 'test_vocab', len(vocab_test),
      'intersection', len(vocab_train & vocab_test))

print('12. dump objects')
with open("data/ind.{}.x_adj".format(dataset), 'wb') as f:
    pkl.dump(x_adj, f)

with open("data/ind.{}.x_embed".format(dataset), 'wb') as f:
    pkl.dump(x_feature, f)

with open("data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("data/ind.{}.tx_adj".format(dataset), 'wb') as f:
    pkl.dump(tx_adj, f)

with open("data/ind.{}.tx_embed".format(dataset), 'wb') as f:
    pkl.dump(tx_feature, f)

with open("data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("data/ind.{}.allx_adj".format(dataset), 'wb') as f:
    pkl.dump(allx_adj, f)

with open("data/ind.{}.allx_embed".format(dataset), 'wb') as f:
    pkl.dump(allx_feature, f)

with open("data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)
