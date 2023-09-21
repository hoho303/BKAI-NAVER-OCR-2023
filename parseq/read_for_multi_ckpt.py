import os

# CKPT_DIR = '/workspace/model/outputs/parseq/2023-08-19_04-08-27/checkpoints'
OUT_DIR = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/inference'
IMAGE_DIR = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test'

checkpoints = [
    '/mlcv/WorkingSpace/Personals/ngocnd/parseq/outputs/parseq/2023-09-19_14-51-39/checkpoints/epoch=84-step=22865-val_accuracy=99.9728-val_NED=99.9917.ckpt'
]

n_ckpts = len(checkpoints) # len(os.listdir(CKPT_DIR))

for idx, ckpt_path in enumerate(checkpoints): # enumerate(sorted(os.listdir(CKPT_DIR))):
    print(f'={idx}/{n_ckpts} @ {ckpt_path}')

    # if ckpt_name != 'last.ckpt':
    #     out_fname = ckpt_name.split('-', maxsplit=1)[0] + '.txt'
    # else:
    #     out_fname = 'last.txt'
    
    # ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    out_fname = ckpt_path.split('/')[-1] + '.txt'
    os.system(f'python read_with_conf.py {ckpt_path} --images {IMAGE_DIR} --out_file_name {out_fname}')