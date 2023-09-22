#                               Train Abinet_v1
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --work-dir /workspace/mmocr/workdir/abinet_v1 \

# =================================================================================================
#                               Train Abinet_v2
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_2.py \
        --work-dir /workspace/mmocr/workdir/abinet_v2 \

# =================================================================================================
#                               Train Abinet_v3
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_3.py \
        --work-dir /workspace/mmocr/workdir/abinet_v3 \

# =================================================================================================
#                               Train Abinet_final
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom.py \
        --work-dir /workspace/mmocr/workdir/abinet_final \
        
# =================================================================================================
#                               Train SATRN
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/satrn/satrn_shallow_5e-custom.py \
        --work-dir /workspace/mmocr/workdir/satrn \

# =================================================================================================
#                               Train Master
python /workspace/mmocr/tools/train.py \
        /workspace/mmocr/configs/textrecog/master/master_resnet31_12e_st_mj_sa-custom.py \
        --work-dir /workspace/mmocr/workdir/master \
