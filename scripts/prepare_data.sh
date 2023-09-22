#                                       Download Dataset


# =================================================================================================
#                                       Prepare For MMOCR

python prepare_utils/prepare_for_mmocr.py \
    --gt-path /workspace/data/train_gt.txt \
    --train-path /workspace/data/train_splitted.json \
    --val-path /workspace/data/val_splitted.json \
    --train-ratio 0.8

# =================================================================================================
#                                       Prepare For Parseq



# =================================================================================================
#                                       Prepare For VietOCR
