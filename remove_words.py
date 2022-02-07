import sys
from utils import clean_str, clean_str_sst, loadWord2Vec
from konlpy.tag import Okt
from tqdm import tqdm

dataset = 'nsmc'
func = clean_str
least_freq = 2

# 1. 불용어 제거
stop_words = open('./data/stopword_kor.txt').read().splitlines()
doc_content_list = []
doc_content_list = open('data/corpus/' + dataset + '.txt', 'r').read().splitlines()
doc_name_list = open('data/' + dataset + '.txt', 'r').read().splitlines()

word_freq = {}  # to remove rare words

for i in tqdm(range(len(doc_content_list))):
    okt = Okt()
    words = okt.morphs(doc_content_list[i], stem=True) # 토큰화
    doc_content_list[i] = ' '.join(words)
    
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
            
names_docs = []
original_docs = [] # 인덱스를 맞추기 위해, clean에서 삭제하는건 원본에서도 삭제해줘야 함
clean_docs = []

for i in range(len(doc_content_list)):
    doc_content = doc_content_list[i]
    words = doc_content.split()
    doc_words = []
    for word in words:
        if word not in stop_words and word_freq[word] >= least_freq:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    
    if len(doc_str) != 0: 
        names_docs.append(doc_name_list[i])
        original_docs.append(doc_content)
        clean_docs.append(doc_str)

for i in range(len(names_docs)):
    tmp_str = names_docs[i].split('\t')
    names_docs[i] = str(i+1)+"\t"+tmp_str[1]+"\t"+tmp_str[2]+"\n"

names_docs_str = '\n'.join(names_docs)
original_docs_str = '\n'.join(original_docs)
clean_corpus_str = '\n'.join(clean_docs)
    
with open('data/' + dataset + '.txt', 'w') as f:
    f.write(names_docs_str)
with open('data/corpus/' + dataset + '.txt', 'w') as f:
    f.write(original_docs_str)
with open('data/corpus/' + dataset + '.clean.txt', 'w') as f:
    f.write(clean_corpus_str)

len_list = []
with open('data/corpus/' + dataset + '.clean.txt', 'r') as f:
    for line in f.readlines():
        if line == '\n':
            continue
        temp = line.strip().split()
        len_list.append(len(temp))

print('min_len : ' + str(min(len_list)))
print('max_len : ' + str(max(len_list)))
print('average_len : ' + str(sum(len_list)/len(len_list)))
