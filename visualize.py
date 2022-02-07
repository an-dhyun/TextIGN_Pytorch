from konlpy.tag import Okt
from utils import *
import torch
from model import GNN
import argparse

parser = argparse.ArgumentParser(description='Pytorch TextIGN Visualizing')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to test.')
args = parser.parse_args()

stop_words = open('./data/stopword_kor.txt').read().splitlines()
test_data = stop_words = open('./visualize.txt').read()

# 1. 토큰화
okt = Okt()
words = okt.morphs(test_data, stem=True) # 토큰화

# 2. 불용어 제거
doc_words = []
for word in words:
    if word not in stop_words: doc_words.append(word)

doc_vocab = list(set(doc_words))
doc_nodes = len(doc_vocab)
doc_word_id_map = {}
for j in range(doc_nodes):
    doc_word_id_map[doc_vocab[j]] = j

# 3. 단어사전 불러와서 현재 단어 업데이트
vocab = open('./data/nsmc_vocab.txt').read().splitlines()
vocab = list(set(vocab + doc_vocab))

word_id_map = {}
for i in range(len(vocab)):
    word_id_map[vocab[i]] = i

# 4. window단위 텍스트 생성
window_size = 3
windows = []
if len(doc_words) <= window_size:
    windows.append(doc_words)
else:
    for j in range(len(doc_words) - window_size + 1):
        window = doc_words[j: j + window_size]
        windows.append(window)

# 5. 단어쌍 갯수 세기
word_pair_count = {}
for window in windows:
    for p in range(1, len(window)):
        for q in range(0, p):
            word_p = window[p]
            word_p_id = word_id_map[word_p]
            word_q = window[q]
            word_q_id = word_id_map[word_q]
            if word_p_id == word_q_id: continue
            
            word_pair_key = (word_p_id, word_q_id)
            if word_pair_key in word_pair_count:
                word_pair_count[word_pair_key] += 1.
            else:
                word_pair_count[word_pair_key] = 1.
            word_pair_key = (word_q_id, word_p_id)
            if word_pair_key in word_pair_count:
                word_pair_count[word_pair_key] += 1.
            else:
                word_pair_count[word_pair_key] = 1.

# 6. 인접행렬, 임베딩 계산                       
weighted_graph = False

row = []
col = []
weight = []
features = []

for key in word_pair_count:
    p = key[0]
    q = key[1]
    row.append(doc_word_id_map[vocab[p]])
    col.append(doc_word_id_map[vocab[q]])
    weight.append(word_pair_count[key] if weighted_graph else 1.)
adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
adj = np.array(adj.toarray())

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, 200)

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
emb = np.array(features)

max_length = 58
pad = max_length - emb.shape[0] # padding for each epoch
emb = np.pad(emb, ((0,pad),(0,0)), mode='constant')

max_length = 58
mask = np.zeros((max_length, 1))
adj_normalized = normalize_adj(adj) # no self-loop
pad = 58 - adj.shape[0]
adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
mask[:adj.shape[0],:]=1
adj = adj_normalized


adj = torch.tensor([adj]).float().to('cuda')
mask = torch.tensor([mask]).float().to('cuda')
emb = torch.tensor([emb]).float().to('cuda')

# 7. 모델 호출
epochs = args.epochs
net = GNN(input_dim=200, hidden_dim=96, output_dim=2).to('cuda')
net = net.to('cuda')
net.load_state_dict(torch.load('./checkpoint/net_epoch_{}.pth'.format(epochs)))

output = net(emb, adj, mask)
output = output.cpu()
predicted = torch.argmax(output, dim=1)

if predicted == 1: print("Positive")
else: print("Negative")
