import os
DATA = '/workspace/data/images'
mapping = {}
for fname in os.listdir(DATA):
    s_fname = fname.split('.')[0]
    mapping[s_fname] = fname

INPUT_FILE = '/workspace/WordArt/results/corner_transformer/corner.txt'
OUTPUT_FILE = '/workspace/results/corner.txt'

with open(INPUT_FILE, 'r', encoding='utf-8') as rf:
    in_samples = rf.readlines()

with open(OUTPUT_FILE, 'w', encoding='utf-8') as wf:
    for in_sample in in_samples:
        in_sample = in_sample.split()
        fname = mapping[in_sample[0]]
        label = in_sample[1].strip()
        score = 1 - float(in_sample[2])
        wf.write(f'{fname} {label} {score}\n')
