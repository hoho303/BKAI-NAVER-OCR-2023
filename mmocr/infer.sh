# python tools/infer.py \
#         /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
#         --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/configs/textrecog/abinet/abinet_20e-custom.py \
#         --batch-size 8 \
#         --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/workdir/abinet/epoch_50.pth \
#         --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/abinet \
#         --save_pred \

# python tools/infer.py \
#         /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
#         --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/configs/textrecog/satrn/satrn_shallow_5e-custom.py \
#         --batch-size 8 \
#         --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/workdir/satrn/epoch_50.pth \
#         --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/satrn \
#         --save_pred \

# python tools/infer.py \
#         /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
#         --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/crnn2/crnn_mini-vgg_5e_mj-custom.py \
#         --batch-size 8 \
#         --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/crnn2/epoch_50.pth \
#         --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/crnn \
#         --save_pred \

# python tools/infer.py \
#         /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
#         --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/configs/textrecog/master/master_resnet31_12e_st_mj_sa-custom.py \
#         --batch-size 4 \
#         --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/workdir/master/epoch_40.pth \
#         --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/master \
#         --save_pred \

python tools/infer.py \
        /mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test \
        --rec /mlcv/WorkingSpace/Personals/ngocnd/mmocr/configs/textrecog/sar/sar_resnet31_parallel-decoder_5e-custom.py \
        --batch-size 8 \
        --rec-weights /mlcv/WorkingSpace/Personals/ngocnd/mmocr/workdir/sar/epoch_38.pth \
        --out-dir /mlcv/WorkingSpace/Personals/ngocnd/mmocr/hello-test \
        --save_pred \

# python run.py