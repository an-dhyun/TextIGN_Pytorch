train = open('./nsmc/ratings_train.txt').read().splitlines()
for i in range(len(train)):
    train[i] = train[i].split('\t')
train = train[1:]
f = open('./data/nsmc_train.txt', 'w')
f_corpus = open('./data/corpus/nsmc_train.txt', 'w')

for i in range(len(train)):
    f.write(str(i)+"\t"+"train"+"\t"+train[i][2]+"\n")
    f_corpus.write(train[i][1]+"\n")

train_len = len(train)

test = open('./nsmc/ratings_test.txt').read().splitlines()
for i in range(len(test)):
    test[i] = test[i].split('\t')
test = test[1:]
f_test = open('./data/nsmc_test.txt', 'w')
f_test_corpus = open('./data/corpus/nsmc_test.txt', 'w')

for i in range(len(test)):
    f_test.write(str(train_len+i)+"\t"+"test"+"\t"+test[i][2]+"\n")
    f_test_corpus.write(test[i][1]+"\n")