import os
import json
import argparse

def convert_results(pred_path, img_path, output_path):
    pred_files = os.listdir(pred_path)
    img_files = os.listdir(img_path)

    preds = []
    preds_conf = []
    
    ext_dict = {}
    for img_file in img_files:
        name, ext = img_file.split('.')
        ext_dict[name] = ext

    print('Prepare')
    for idx, f in enumerate(pred_files):
        if (idx + 1) % 1000 == 0:
            print(f'{idx + 1}/{len(pred_files)}')
        result = open(pred_path + f, 'r', encoding='utf-8')
        result = result.read()
        result = json.loads(result)
        pred = result['rec_texts'][0]
        conf = result['rec_scores'][0]
        img_name = f.split('.')[0]
        img_name = img_name + '.' + ext_dict[img_name]

        preds_conf.append(img_name + " " + pred + " " + str(conf))
        preds.append(img_name + " " + pred)

    # print('Make prediction.txt')
    # with open(output_path + "prediction.txt", 'w', encoding='utf-8') as f:
    #     for pred in preds:
    #         f.write(pred + '\n')

    print('Make' + output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred_conf in preds_conf:
            f.write(pred_conf + '\n')

# convert_results("/mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/sar/preds/"
#                 ,"/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test/"
#                 ,"/mlcv/WorkingSpace/Personals/ngocnd/mmocr/results/sar/")

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')

    parser.add_argument('--pred_path', type=str, default='predictions', help='path to predictions folder')
    parser.add_argument('--img_path', type=str, default='/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/new_public_test/', help='path to test images folder')
    parser.add_argument('--output_path', type=str, default='predictions.txt', help='path to output file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    convert_results(args.pred_path, args.img_path, args.output_path)