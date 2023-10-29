CROPPED_FOLDER=/workspace/DBNetpp/mmocr/inference-result/crop-image-1
JSON_FOLDER=/workspace/DBNetpp/mmocr/inference-result/json
VIS_FOLDER=/workspace/DBNetpp/mmocr/inference-result/vis
IMG_FOLDER=/workspace/data/NAVER_OCR_private_test_update

BATCH_SIZE=3

rm -rf $CROPPED_FOLDER
mkdir $CROPPED_FOLDER

rm -rf $JSON_FOLDER
mkdir $JSON_FOLDER

rm -rf $VIS_FOLDER
mkdir $VIS_FOLDER

cd /workspace/DBNetpp/mmocr
CUDA_VISIBLE_DEVICES=0 python3 mmocr/utils/ocr.py $IMG_FOLDER \
    --det DBPP_r50 \
    --det-ckpt /workspace/DBNetpp/mmocr/pretrained/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth \
    --recog None \
    --export $JSON_FOLDER \
    --output $VIS_FOLDER \
    --batch-mode \
    --single-batch-size $BATCH_SIZE

# crop images
cd ..
python3 crop_image.py \
    $IMG_FOLDER \
    $JSON_FOLDER \
    $CROPPED_FOLDER