import os
import json
import time
import zipfile

def ensemble_abinet(PRED_FOLDER, OUTPUT_FILE, DICT_FILE, threshold=0.88):
    sum_predictions = {}

    start = time.time()

    with open(DICT_FILE, 'r', encoding='utf-8') as rf:
        my_dict = json.load(rf)

    for pred_file in os.listdir(PRED_FOLDER):
        model_name = pred_file.split('.')[0]

        rf = open(os.path.join(PRED_FOLDER, pred_file), 'r', encoding='utf-8')
        for sample in rf.readlines():
            sample = sample.split()
            img_name, pred, conf_scores = sample[0], sample[1], sample[2:]
            min_conf_score = min(conf_scores)
            if img_name not in sum_predictions:
                sum_predictions[img_name] = {}
            sum_predictions[img_name][model_name] = {
                'pred': pred,
                'conf': min_conf_score
            }
        rf.close()

    wf = open(OUTPUT_FILE, 'w', encoding='utf-8')

    for img_name, content in sum_predictions.items():
        valid_predictions = {}

        for model_name, pred_conf in content.items():
            if float(pred_conf['conf']) >= threshold:
                valid_predictions[model_name] = pred_conf['pred']

        if valid_predictions:
            counter = {}
            re_conf = {}
            for model_name, pred in valid_predictions.items():
                if pred not in counter:
                    counter[pred] = 1
                    re_conf[pred] = float(content[model_name]['conf'])
                else:
                    counter[pred] += 1
                    re_conf[pred] = max(re_conf[pred], float(content[model_name]['conf']))

            for pred in counter:
                if pred in my_dict:
                    counter[pred] += 1

            max_pred = max(counter, key=counter.get)
            max_count = counter[max_pred]
            max_re_conf = re_conf[max_pred]

            is_unique = 0
            for pred, count in counter.items():
                if count == max_count:
                    is_unique += 1
            if is_unique == 1:
                wf.write(f'{img_name} {max_pred} {max_re_conf}\n')
            else:
                pred_conf_pairs = {}
                for model_name, pred in valid_predictions.items():
                    pred_conf_pairs[pred] = float(content[model_name]['conf'])
                max_pred = max(pred_conf_pairs, key=pred_conf_pairs.get)
                wf.write(f'{img_name} {max_pred} {max_re_conf}\n')
        else:
            max_conf = max(content.values(), key=lambda x: float(x['conf']))
            wf.write(f'{img_name} {max_conf["pred"]} {max_conf["conf"]}\n')

    wf.close()
    print(time.time() - start)

def make_final_result(PRED_FOLDER, OUTPUT_FILE, ZIP_FILE, DICT_FILE, threshold=0.88):
    sum_predictions = {}

    start = time.time()

    with open(DICT_FILE, 'r', encoding='utf-8') as rf:
        my_dict = json.load(rf)

    for pred_file in os.listdir(PRED_FOLDER):
        if ('txt' not in pred_file) or pred_file == 'prediction.txt':
            continue
        print(pred_file)
        model_name = pred_file.split('.')[0]

        rf = open(os.path.join(PRED_FOLDER, pred_file), 'r', encoding='utf-8')
        for sample in rf.readlines():
            sample = sample.split()
            img_name, pred, conf_scores = sample[0], sample[1], sample[2:]
            min_conf_score = min(conf_scores)
            if img_name not in sum_predictions:
                sum_predictions[img_name] = {}
            sum_predictions[img_name][model_name] = {
                'pred': pred,
                'conf': min_conf_score
            }
        rf.close()

    wf = open(OUTPUT_FILE, 'w', encoding='utf-8')

    for img_name, content in sum_predictions.items():
        valid_predictions = {}

        for model_name, pred_conf in content.items():
            if float(pred_conf['conf']) >= threshold:
                valid_predictions[model_name] = pred_conf['pred']

        if valid_predictions:
            counter = {}
            re_conf = {}
            for model_name, pred in valid_predictions.items():
                if pred not in counter:
                    counter[pred] = 1
                    re_conf[pred] = float(content[model_name]['conf'])
                else:
                    counter[pred] += 1
                    re_conf[pred] = max(re_conf[pred], float(content[model_name]['conf']))

            for pred in counter:
                if pred in my_dict:
                    counter[pred] += 1

            max_pred = max(counter, key=counter.get)
            max_count = counter[max_pred]
            max_re_conf = re_conf[max_pred]

            is_unique = 0
            for pred, count in counter.items():
                if count == max_count:
                    is_unique += 1
            if is_unique == 1:
                wf.write(f'{img_name} {max_pred} {max_re_conf}\n')
            else:
                max_pred = content['abirest']['pred']
                md_conf = content['abirest']['conf']
                wf.write(f'{img_name} {max_pred} {md_conf}\n')
        else:
            max_conf = max(content.values(), key=lambda x: float(x['conf']))
            wf.write(f'{img_name} {max_conf["pred"]} {max_conf["conf"]}\n')

    wf.close()
    print(time.time() - start)

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    result_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            result_lines.append(' '.join(parts[:2]))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        file.writelines('\n'.join(result_lines))

    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(OUTPUT_FILE, arcname=os.path.basename(OUTPUT_FILE))

    print(f'Tệp zip đã được tạo: {ZIP_FILE}')
    
# Example usage:
PRED_FOLDER = '/workspace/results/abinet'
OUTPUT_FILE = '/workspace/results/abirest.txt'
DICT_FILE = '/workspace/data/dict.json'
ensemble_abinet(PRED_FOLDER, OUTPUT_FILE, DICT_FILE)

PRED_FOLDER = '/workspace/results'
OUTPUT_FILE = '/workspace/results/prediction.txt'
ZIP_FILE = '/workspace/result.zip'
make_final_result(PRED_FOLDER, OUTPUT_FILE, ZIP_FILE, DICT_FILE)