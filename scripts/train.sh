# =================================================================================================
#                               MMOCR Model Training

# Abinet 1
cd /workspace/mmocr
python tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet1

# Abinet 2
python tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_2.py \
        --work-dir /workspace/mmocr/workdir/abinet2

# Abinet 3
python tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_3.py \
        --work-dir /workspace/mmocr/workdir/abinet3

# Abinet 4
python tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom.py \
        --work-dir /workspace/mmocr/workdir/abinet4

# Master
python tools/train.py \
        /workspace/mmocr/configs/textrecog/master/master_resnet31_12e_st_mj_sa-custom.py \
        --work-dir /workspace/mmocr/workdir/master 

# Satrn
python tools/train.py \
        /workspace/mmocr/configs/textrecog/satrn/satrn_shallow_5e-custom.py \
        --work-dir /workspace/mmocr/workdir/satrn

# =================================================================================================
#                               PARSEQ Model Training

cd /workspace/parseq
python3.8 train.py
cd /workspace

# =================================================================================================
#                               VIETOCR Model Training

cd /workspace/vietocr
python3.8 train_vietocr.py
cd /workspace