naver_data_root = '/workspace/data/'

naver_rec_train = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='mmocr_fulltrain_labels.json',
    pipeline=None,
    test_mode=False)

naver_rec_test = dict(
    type='OCRDataset',
    data_root=naver_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='mmocr_test_labels.json',
    pipeline=None,
    test_mode=True)
