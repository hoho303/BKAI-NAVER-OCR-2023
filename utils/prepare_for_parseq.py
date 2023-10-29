#!/usr/bin/env python3
""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import io
import os
import random

import fire
import lmdb
import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
    return np.prod(img.size) > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """

    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as f:
        data = f.readlines()

    nSamples = len(data)
    for i, line in enumerate(data):
        imagePath, label = line.strip().split(maxsplit=1)
        imagePath = os.path.join(inputPath, imagePath)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                img = Image.open(io.BytesIO(imageBin)).convert('RGB')
            except IOError as e:
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{}-th image data occured error: {}, {}\n'.format(i, imagePath, e))
                continue
            if np.prod(img.size) == 0:
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

def parseqUtil(inputPath, gtList, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1

    # with open(gtFile, 'r', encoding='utf-8') as f:
    #     data = f.readlines()

    nSamples = len(gtList)
    for i, line in enumerate(gtList):
        imagePath, label = line.strip().split(maxsplit=1)
        imagePath = os.path.join(inputPath, imagePath)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                img = Image.open(io.BytesIO(imageBin)).convert('RGB')
            except IOError as e:
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{}-th image data occured error: {}, {}\n'.format(i, imagePath, e))
                continue
            if np.prod(img.size) == 0:
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

def createDatasetParseq(
    inputPath, gtFile, trainOutputPath, valOutputPath, train_ratio, train_val_same=True
):
    with open(gtFile, 'r', encoding='utf-8') as rf:
        samples = rf.readlines()

    n_samples = len(samples)
    partition = int(n_samples * train_ratio)
    random.shuffle(samples)
    trainGtList = samples[:partition]

    if train_val_same:
        valGtList = trainGtList
    else:
        valGtList = samples[partition:]

    print(f'train set size: {len(trainGtList)}\t val set size: {len(valGtList)}')

    print(f'Create lmdb for train')
    parseqUtil(inputPath, trainGtList, trainOutputPath)

    print(f'Create lmdb for train')
    if len(valGtList) != 0:
        parseqUtil(inputPath, valGtList, valOutputPath)

if __name__ == '__main__':
    fire.Fire(createDatasetParseq)
