docker run --name CRUSADER_BKAI_NGOCNGUYEN_PIPELINE -v /mlcv/WorkingSpace/Personals/ngocnd/BKAI-NAVER-OCR-2023/data:/workspace/data -v /mlcv/WorkingSpace/Personals/ngocnd/BKAI-NAVER-OCR-2023/scripts:/workspace/scripts -v /mlcv:/mlcv -w /mlcv --gpus all -t -d --shm-size=256m bkai2023:latest #mmocr 