import json
import random
import os 
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--gt-path', type=str, help='Groundtruth file')
    parser.add_argument(
        '--train-path',
        type=str,
        help='Train path')
    parser.add_argument(
        "--val-path",
        type=str,
        help='Val path'
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Train ratio"
    )

    return vars(parser.parse_args())

args = parse_args()
print(args)
# TXT_FILE = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/train_gt.txt'
# TRAIN_JSON_FILE = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/random_forest_gt/abi_train_gt_3.json'
# VAL_JSON_FILE = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/random_forest_gt/abi_val_gt_3.json'

# # Open raw gt file
# with open(TXT_FILE, 'r', encoding='utf-8') as rf:
#     samples = rf.readlines()

# # Shuffle the samples randomly
# random.shuffle(samples)

# # Create gt dict
# train_gt = {
#     "metainfo": {
#         "dataset_type": "TextRecogDataset",
#         "task_name": "textrecog"
#     }, 
#     "data_list": []
# }

# val_gt = {
#     "metainfo": {
#         "dataset_type": "TextRecogDataset",
#         "task_name": "textrecog"
#     }, 
#     "data_list": []
# }

# n_samples = len(samples)
# train_ratio = 0.8
# partition = int(n_samples * train_ratio)

# # Split the shuffled samples into training and validation sets
# train_samples = samples[:partition]
# val_samples = samples[partition:]

# for sample in train_samples:
#     img_name, label = sample.split()
#     label = label.strip()
#     obj = {
#         'instances': [{'text': label}],
#         'img_path': os.path.join(img_name)
#     }
#     train_gt['data_list'].append(obj)

# for sample in val_samples:
#     img_name, label = sample.split()
#     label = label.strip()
#     obj = {
#         'instances': [{'text': label}],
#         'img_path': os.path.join(img_name)
#     }
#     val_gt['data_list'].append(obj)

# # Write to json file
# with open(TRAIN_JSON_FILE, 'w', encoding='utf-8') as wf:
#     json.dump(train_gt, wf, ensure_ascii=False)

# with open(VAL_JSON_FILE, 'w', encoding='utf-8') as wf:
#     json.dump(val_gt, wf, ensure_ascii=False, indent=4)
