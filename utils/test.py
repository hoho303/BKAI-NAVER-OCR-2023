import os

list_img = os.listdir(r'/workspace/data/NAVER_OCR_private_test_update')
preds = open("/workspace/pred-visualization/front-end/results/test_pred.txt", 'r').readlines()
predicts = []
for pred in preds: 
    pred = pred.strip().split(' ')
    pred = pred[0]
    predicts.append(pred)

for x in list_img:
    if x not in predicts:
        print(x)
    
