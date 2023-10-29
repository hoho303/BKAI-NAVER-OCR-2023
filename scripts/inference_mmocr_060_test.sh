# =================================================================================================
#                               CORNER Model Inference

mkdir -p /workspace/WordArt/results/corner_transformer
IMG_PATH=/workspace/data/NAVER_OCR_private_test_update
BATCH_SIZE=128

cd /workspace/WordArt
python3 mmocr/utils/ocr.py $IMG_PATH --det None \
            --recog-config /workspace/WordArt/configs/textrecog/corner_transformer/corner_transformer_academic.py \
            --recog-ckpt /workspace/checkpoints/corner_100_0.ckpt --export results/corner_transformer \
            --batch-mode \
            --single-batch-size $BATCH_SIZE

python3 tools/convert_pred.py

