# python tools/infer.py \
#         /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
#         --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
#         --batch-size 8 \
#         --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/workdir/abinet_1/epoch_50.pth \
#         --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/abinet_1 \
#         --save_pred \

python run.py \
        --pred_path /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/abinet_1/preds/ \
        --img_path /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test/\
        --output_path /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/abinet_1/ \