import random

d = dict()

regions = ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']

with open('full_timit_spk.list', 'r') as f:
    spk = f.readlines()
    for l in spk:
        l = l.strip()
        d[l] = None

with open('full.scp', 'r') as f:
    lines = f.readlines()
    for l in lines:
        spk = l[:5]
        if d[spk] != None:
            continue
        
        for dr in regions:
            if dr in l:
                d[spk] = dr
                break

d2 = dict()
for dr in regions:
    d2[dr] = []

for k in d:
    d2[d[k]].append(k)
# print(d2)

# train
# dev
# test
n1 = 4
n2 = 4
w1 = open('dev_timit_spk.list', 'w+')
w2 = open('test_timit_spk.list', 'w+')
w3 = open('train_timit_spk.list', 'w+')

for k in d2:
    l = d2[k]
    indexs = list(range(len(l)))
    random.shuffle(indexs)
    print(indexs)
    for e in indexs[:n1]:
        w1.write(l[e]+'\n')
    for e in indexs[n1:n1+n2]:
        w2.write(l[e]+'\n')

    for e in indexs[n1+n2:]:
        w3.write(l[e]+'\n')

w1.close()
w2.close()
w3.close()