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


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', type=str, help='Images to read', default='/workspace/data/new_public_test')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_file_path', default='prediction.txt', help='file path')
    parser.add_argument('--batch_size', type=int, default=32)
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    wf_conf = open(args.out_file_path, 'w', encoding='utf-8')
    cnt = 0
    n_samples = len(os.listdir(args.images))
    fnames = sorted(os.listdir(args.images))

    fname = '/workspace/data/new_public_test/public_test_img_0.jpg'

    # image = Image.open(fname).convert('RGB')
    # image = img_transform(image).unsqueeze(0).to(args.device)
    # features = model.encode(image)
    # print(features.shape)
    # Gather images into batches
    batches = []
    for i in range(0, len(fnames), args.batch_size):
        batches.append(fnames[i: i + args.batch_size])
    
    # Infer 
    for bid, batch in enumerate(batches):
        if (bid + 1) % 1 == 0:
            print(f'{bid + 1}/{len(batches)}')

        img_batch = []
        for fname in batch:
            image = Image.open(os.path.join(args.images, fname)).convert('RGB')
            image = img_transform(image)
            img_batch.append(image)

        img_batch = torch.stack(img_batch, dim=0)
        img_batch = img_batch.to(args.device)
        features = model.encode(img_batch).reshape
        print(features.shape)
        # p = model(img_batch).softmax(-1)
        # preds, confs = model.tokenizer.decode(p)

        # for fname, pred, conf in zip(batch, preds, confs):
        #     conf = ' '.join(list(map(str, conf.tolist())))
        #     wf_conf.write(f'{fname} {pred} {conf}\n')

    # wf_conf.close()

if __name__ == '__main__':
    main()
