with open('/mlcv/WorkingSpace/Personals/ngocnd/BKAI2023/BKAI-NAVER-OCR-2023/mmocr/dicts/187_vietnamese.txt', 'r', encoding='utf-8') as rf:
    chars = rf.readlines()

with open('187viet.txt', 'w', encoding='utf-8') as wf:
    for char in chars:
        wf.write(char[0])