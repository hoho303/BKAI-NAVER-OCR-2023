cd /workspace/WordArt
IMG_PATH=/workspace/Models/OCR/mmocr/inference-result/crop-image
BATCH_SIZE=128

python3 mmocr/utils/ocr.py $IMG_PATH --det None \
            --recog-config /workspace/WordArt/configs/textrecog/corner_transformer/corner_transformer_academic.py \
            --recog-ckpt /workspace/checkpoints/corner_100_0.ckpt --export results/corner_transformer \
            --batch-mode \
            --single-batch-size $BATCH_SIZE

python3 tools/convert_pred.py