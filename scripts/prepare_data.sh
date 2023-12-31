#                                      Download Dataset

pip install fire
apt-get install unzip

mkdir -p /workspace/data
cd /workspace/data

curl -L -o public_test_data.zip 'https://drive.google.com/uc?export=download&id=1wfrz4hACcT0FeNu9XPNgtbeKUMLbujdb&export=download&confirm=t'
curl -L -o training_data.zip 'https://drive.google.com/uc?export=download&id=1lR0b1QBIsXk9JqL9__HgJQi0PdWeNxKi&export=download&confirm=t'
curl -L -o train_gt.txt 'https://drive.google.com/uc?export=download&id=1b4fCTrnfKnR0GHm1XCve9nhy7JyrqahR&export=download&confirm=t'

unzip training_data.zip
unzip public_test_data.zip

# =================================================================================================
#                                      Prepare For MMOCR

cd /workspace
python utils/prepare_for_mmocr.py \
    --gt-path /workspace/data/train_gt.txt \
    --train-path /workspace/data/abi_train_gt_1.json \
    --val-path /workspace/data/abi_val_gt_1.json \
    --train-ratio 0.8

python utils/prepare_for_mmocr.py \
    --gt-path /workspace/data/train_gt.txt \
    --train-path /workspace/data/abi_train_gt_2.json \
    --val-path /workspace/data/abi_val_gt_2.json \
    --train-ratio 0.8

python utils/prepare_for_mmocr.py \
    --gt-path /workspace/data/train_gt.txt \
    --train-path /workspace/data/abi_train_gt_3.json \
    --val-path /workspace/data/abi_val_gt_3.json \
    --train-ratio 0.8

python utils/prepare_for_mmocr.py \
    --gt-path /workspace/data/train_gt.txt \
    --train-path /workspace/data/full_train_gt.json \
    --val-path /workspace/data/full_val_gt.json \
    --train-ratio 1

# =================================================================================================
#                                       Prepare For Parseq

rm -rf /workspace/parseq/data
mkdir /workspace/parseq/data
python utils/prepare_for_parseq.py \
    /workspace/data/new_train \
    /workspace/data/train_gt.txt \
    /workspace/parseq/data/train/soict2023 \
    /workspace/parseq/data/val/soict2023 \
    1 \
    True