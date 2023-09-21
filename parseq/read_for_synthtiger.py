#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import os

from PIL import Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

LV1_IMG_DIR = '/mlcv/WorkingSpace/Personals/ngocnd/synthtiger/results/images'
OUTPUT_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/inference/synth_tiger.txt'
CHECKPOINT_PATH = '/mlcv/WorkingSpace/Personals/ngocnd/parseq/checkpoints/parseq/last.ckpt'
DEVICE = 'cuda'

@torch.inference_mode()
def main():
    model = load_from_checkpoint(CHECKPOINT_PATH).eval().to(DEVICE)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    wf = open(OUTPUT_PATH, 'w', encoding='utf-8')

    lv2_img_folders = sorted(os.listdir(LV1_IMG_DIR))
    for lv2_img_folder in lv2_img_folders:
        lv2_img_dir = os.path.join(LV1_IMG_DIR, lv2_img_folder)
        img_files = sorted(os.listdir(lv2_img_dir))
        
        for img_file in img_files:
            img_path = os.path.join(lv2_img_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            image = img_transform(image).unsqueeze(0).to(DEVICE)

            p = model(image).softmax(-1)
            pred, p = model.tokenizer.decode(p)

            wf.write(f'images/{lv2_img_folder}/{img_file} {pred[0]}\n')
            wf.flush()

    print('DONE')
    wf.close()

if __name__ == '__main__':
    main()
