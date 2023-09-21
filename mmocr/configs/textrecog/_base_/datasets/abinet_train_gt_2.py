naver_data_root = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/'

naver_rec_train = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='random_forest_gt/abi_train_gt_2.json',
    pipeline=None,
    test_mode=False)

naver_rec_test = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='random_forest_gt/abi_val_gt_2.json',
    pipeline=None,
    test_mode=True)
