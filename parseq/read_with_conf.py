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
    parser.add_argument('--images', nargs='+', help='Images to read')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out_file_name', default='prediction.txt', help='Just file name, not path, result will be saved in inference folder')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    if not os.path.isdir('./inference'):
      os.mkdir('./inference')
    wf_no_conf = open(f'inference/{args.out_file_name}', 'w', encoding='utf-8')
    out_file_name_conf = args.out_file_name.replace('.txt', '_with_conf.txt')
    wf_with_conf = open(f'inference/{out_file_name_conf}', 'w', encoding='utf-8')
    # wf = open('prediction.txt', 'a', encoding='utf-8')
    cnt = 0
    n_samples = len(os.listdir(args.images[0]))

    for fname in os.listdir(args.images[0]):
        # Load image and prepare for input
        image = Image.open(f'{args.images[0]}/{fname}').convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, conf = model.tokenizer.decode(p)
        conf = ' '.join(list(map(str, conf[0].tolist())))

        wf_no_conf.write(f'{fname} {pred[0]}\n')
        wf_no_conf.flush()

        wf_with_conf.write(f'{fname} {pred[0]} {conf}\n')
        wf_with_conf.flush()
        
        if cnt % 1000 == 0:
          print(f'{cnt}/{n_samples}')
        cnt += 1

    wf_no_conf.close()
    wf_with_conf.close()
if __name__ == '__main__':
    main()
