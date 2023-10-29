dataset_type = 'IcdarDataset'
data_root = '/mlcv/WorkingSpace/Personals/ngocnd/BKAI2023/BKAI-NAVER-OCR-2023/data/'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/naver2023_coco.json',
    img_prefix=f'{data_root}/Overlay',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/naver2023_coco.json',
    img_prefix=f'{data_root}/Overlay',
    pipeline=None)

train_list = [train]

test_list = [test]
