import json

GT_FILE = '/workspace/data/train_gt.txt'
DICT_FILE = '/workspace/data/dict.json'

word_set = set()
with open(GT_FILE, 'r', encoding='utf-8') as rf:
    for sample in rf.readlines():
        img_name, label = sample.split('\t')
        label = label.strip()
        word_set.update([label])

with open(DICT_FILE, 'w', encoding='utf-8') as wf:
    json.dump(list(word_set), wf, indent=4, ensure_ascii=False)