import json
import random
import os 
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--gt-path', type=str, help='Groundtruth file (txt)')
    parser.add_argument(
        '--train-path',
        type=str,
        help='Train path (json)')
    parser.add_argument(
        "--val-path",
        type=str,
        help='Val path (json)'
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Train ratio"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Open raw gt file
    with open(args.gt_path, 'r', encoding='utf-8') as rf:
        samples = rf.readlines()

    # Shuffle the samples randomly
    random.shuffle(samples)

    # Create gt dict
    train_gt = {
        "metainfo": {
            "dataset_type": "TextRecogDataset",
            "task_name": "textrecog"
        }, 
        "data_list": []
    }

    val_gt = {
        "metainfo": {
            "dataset_type": "TextRecogDataset",
            "task_name": "textrecog"
        }, 
        "data_list": []
    }

    n_samples = len(samples)
    partition = int(n_samples * args.train_ratio)
    print(f'train set size: {partition}\t val set size: {n_samples - partition}')
    # Split the shuffled samples into training and validation sets
    train_samples = samples[:partition]
    val_samples = samples[partition:]

    for sample in train_samples:
        img_name, label = sample.split()
        label = label.strip()
        obj = {
            'instances': [{'text': label}],
            'img_path': os.path.join(img_name)
        }
        train_gt['data_list'].append(obj)

    for sample in val_samples:
        img_name, label = sample.split()
        label = label.strip()
        obj = {
            'instances': [{'text': label}],
            'img_path': os.path.join(img_name)
        }
        val_gt['data_list'].append(obj)

    # Write to json file
    with open(args.train_path, 'w', encoding='utf-8') as wf:
        json.dump(train_gt, wf, ensure_ascii=False)

    if partition < n_samples:
        with open(args.val_path, 'w', encoding='utf-8') as wf:
            json.dump(val_gt, wf, ensure_ascii=False, indent=4)
    else:
        print(f'val is empty because train_ratio is {args.train_ratio}')

if __name__ == '__main__':
    main()