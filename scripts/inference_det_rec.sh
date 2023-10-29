# # =================================================================================================
# #                               Prepare Folder, Variable
# # Định dạng folder
# # workspace
# # ---result
# # ------model.txt
# # ------abinet
# # -----------abinetv1.txt
# # -----------abinetv2.txt

IMG_PATH=/workspace/Models/OCR/mmocr/inference-result/crop-image-pretrain-0.75
BATCH_SIZE=128
mkdir -p /workspace/preprocess_results/
mkdir -p /workspace/preprocess_results/abinet/

# # =================================================================================================
# #                               MMOCR Model Inference

# Abinet 1
cd /workspace/mmocr
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/abinet1_80_20.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/abinet1 \
        --save_pred

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet1/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/abinet/abinet1.txt 

# Abinet 2
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_2.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/abinet2_80_20.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/abinet2 \
        --save_pred 

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet2/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/abinet/abinet2.txt 

# Abinet 3
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_3.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/abinet3_80_20.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/abinet3 \
        --save_pred 

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet3/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/abinet/abinet3.txt 

# Abinet 4
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/abinet4_100_0.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/abinet4 \
        --save_pred

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet4/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/abinet/abinet4.txt
        
# Abinet 5
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_5.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/mmocr/workdir/abinet4/epoch_50.pth \
        --out-dir /workspace/mmocr/preprocess_results/abinet5 \
        --save_pred

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet5/preds/ \
        --img_path $IMG_PATH/\
        --output_path /workspace/preprocess_results/abinet5.txt

# Abinet 6
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_6.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/mmocr/workdir/abinet5/epoch_50.pth \
        --out-dir /workspace/mmocr/preprocess_results/abinet6 \
        --save_pred

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/abinet6/preds/ \
        --img_path $IMG_PATH/\
        --output_path /workspace/preprocess_results/abinet6.txt 

# master
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/master/master_resnet31_12e_st_mj_sa-custom.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/master_100_0.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/master \
        --save_pred 

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/master/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/master.txt

# satrn
python tools/infer.py \
        $IMG_PATH \
        --rec /workspace/mmocr/configs/textrecog/satrn/satrn_shallow_5e-custom.py \
        --batch-size $BATCH_SIZE \
        --rec-weights /workspace/checkpoints/satrn_100_0.ckpt \
        --out-dir /workspace/mmocr/preprocess_results/satrn \
        --save_pred 

python tools/convert_pred.py \
        --pred_path /workspace/mmocr/preprocess_results/satrn/preds/ \
        --img_path $IMG_PATH/ \
        --output_path /workspace/preprocess_results/satrn.txt 
# # =================================================================================================
# #                               PARSEQ Model Inference

cd /workspace/parseq
python3.8 read_for_ensemble_batch.py /workspace/checkpoints/parseq_100_0.ckpt \
    --images $IMG_PATH \
    --out_file_path /workspace/preprocess_results/parseq.txt \
    --batch_size $BATCH_SIZE

# # =================================================================================================
# #                               VIETOCR Model Inference

# cd /workspace/vietocr
# python3.8 infer.py \
#     --images $IMG_PATH \
#     --config-file-path /workspace/vietocr/config.yml \
#     --ckpt-path /workspace/checkpoints/vietocr_100_0.pth \
#     --out-file-path /workspace/preprocess_results/vietocr.txt
    
# =================================================================================================
#                               Ensemble Model

# Create dict.json
# python /workspace/utils/create_dict.py
# Ensemble
# python /workspace/utils/ensemble.py 

# result.zip will be saved at /workspace/result.zip
