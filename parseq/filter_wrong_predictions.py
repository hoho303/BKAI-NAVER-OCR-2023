GT_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/synthtiger/results/gt.txt'
PRED_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/inference/synth_tiger.txt'
WRONG_PRED_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/data/true_predictions.txt'
TRUE_PRED_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/data/wrong_predictions.txt'

# Read gt.txt -> dict
gt_dict = {}
with open(GT_PATH, encoding='utf-8') as rf:
    samples = rf.readlines()
    for sample in samples:
        fpath, label = sample.split('\t')
        label = label.strip()
        gt_dict[fpath] = label

# Open wrong_predictions.txt, tru_predictions.txt -> write
wwf = open(WRONG_PRED_PATH, 'w', encoding='utf-8') 
twf = open(TRUE_PRED_PATH, 'w', encoding='utf-8')

# Read prediction.txt -> compare with read.txt
with open(PRED_PATH, encoding='utf-8') as rf:
    samples = rf.readlines()

    for sample in samples:
        fpath, pred = sample.split(' ')
        fname = fpath.split('/')[-1]
        pred = pred.strip()

        if gt_dict[fpath] != pred:
            wwf.write(f'{fpath}\t{gt_dict[fpath]}\t{pred}\n')
        else:
            twf.write(f'{fpath}\t{pred}\n')
wwf.close()
twf.close()