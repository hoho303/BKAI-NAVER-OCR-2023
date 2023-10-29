import json

with open('/mlcv/WorkingSpace/Personals/ngocnd/BKAI2023/BKAI-NAVER-OCR-2023/WordArt/results/corner_transformer/out_public_test_img_1.json', 'r', encoding='utf-8') as rf:
    print(json.load(rf))