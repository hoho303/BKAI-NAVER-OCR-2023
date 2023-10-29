from PIL import Image
import os
from natsort import natsorted
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import argparse

# config_file_path = '/workspace/vietocr/config.yml'
# config = Cfg.load_config_from_file(config_file_path)
# image_dir = '/workspace/data/new_public_test'
# images = natsorted(os.listdir(image_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, help='Images to read')
    parser.add_argument('--config-file-path', type=str, help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--ckpt-path', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--out-file-path', type=str, default="vietocr.txt", help="Path to output file")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Cfg.load_config_from_file(args.config_file_path)
    images = natsorted(os.listdir(args.images))

    if args.ckpt_path is not None:
        config['weights'] = args.ckpt_path

    predictions = []
    detector = Predictor(config)

    with open(args.out_file_path, 'w') as f:
        for idx, img_file in enumerate(images):
            img_path = os.path.join(args.images, img_file)
            img = Image.open(img_path)
            s, prob = detector.predict(img, return_prob=True)
            f.write(img_file + ' ' + s + ' ' + str(prob) + '\n')
            f.flush()  # Đảm bảo rằng dữ liệu đã được ghi vào tệp
            
            if (idx + 1) % 100 == 0:
                print(f'{idx + 1} / {len(images)}')

if __name__ == '__main__':
    main()
