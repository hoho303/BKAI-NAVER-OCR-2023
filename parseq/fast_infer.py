PRED_PATH_1 = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/inference/epoch=92-step=25017-val_accuracy=99.9748-val_NED=99.9923.ckpt.txt'
PRED_PATH_2 = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/inference/last.ckpt.txt'
pred_base = {}
pred_dif = {}
with open(PRED_PATH_1, 'r', encoding='utf-8') as rf:
    for sample in rf.readlines():
        fname, pred = sample.split()
        pred = pred.strip()
        pred_base[fname] = pred

with open(PRED_PATH_2, 'r', encoding='utf-8') as rf:
    for sample in rf.readlines():
        fname, pred = sample.split()
        pred = pred.strip()
        if pred_base[fname] != pred:
            pred_dif[fname] = {}
            pred_dif[fname]['pred1'] = pred_base[fname]
            pred_dif[fname]['pred2'] = pred

print(len(pred_dif))