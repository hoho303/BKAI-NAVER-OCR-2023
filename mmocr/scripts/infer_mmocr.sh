# =================================================================================================
#                                       Inference Abinet_v1

python /workspace/mmocr/tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_1.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/abinet_v1/epoch_50.pth \
        --out-dir /workspace/mmocr/results/abinet_v1 \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/abinet_v1/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/abinet/ --model_name abinet_v1 \

# =================================================================================================
#                                       Infernce Abinet_v2

python tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_2.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/abinet_v2/epoch_50.pth \
        --out-dir /workspace/mmocr/results/abinet_v2 \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/abinet_v2/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/abinet/ --model_name abinet_v2 \

# =================================================================================================
#                                       Infernce Abinet_v3

python tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom_3.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/abinet_3/epoch_50.pth \
        --out-dir /workspace/mmocr/results/abinet_v3 \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/abinet_v3/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/abinet/ --model_name abinet_v3 \

# =================================================================================================
#                                       Infernce Abinet_final

python tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/abinet/abinet_20e-custom.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/abinet_final/epoch_50.pth \
        --out-dir /workspace/mmocr/results/abinet_final \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/abinet_final/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/ --model_name abinet_final \

# =================================================================================================
#                                       Inference SATRN

python /workspace/mmocr/tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/satrn/satrn_shallow_5e-custom.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/satrn/epoch_50.pth \
        --out-dir /workspace/mmocr/results/satrn \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/satrn/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/ --model_name satrn \

# =================================================================================================
#                                       Inference Master

python /workspace/mmocr/tools/infer.py \
        /workspace/data/new_public_test \
        --rec /workspace/mmocr/configs/textrecog/master/master_resnet31_12e_st_mj_sa-custom.py \
        --batch-size 8 \
        --rec-weights /workspace/mmocr/workdir/master/epoch_40.pth \
        --out-dir /workspace/mmocr/results/master \
        --save_pred \

python /workspace/mmocr/tools/convert_pred.py \
        --pred_path /workspace/mmocr/results/master/preds/ \
        --img_path /workspace/data/new_public_test/ \
        --output_dir /workspace/results/ --model_name master \