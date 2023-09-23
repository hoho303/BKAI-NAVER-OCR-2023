naver_data_root = '/workspace/data'

naver_rec_train = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train'),
    ann_file='abi_train_gt_1.json',
    pipeline=None,
    test_mode=False)

naver_rec_test = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train'),
    ann_file='abi_val_gt_1.json',
    pipeline=None,
    test_mode=True)
