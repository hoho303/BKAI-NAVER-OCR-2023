soict2023_textrecog_data_root = '/mlcv/WorkingSpace/Personals/ngocnd/SoICT2023-OCR-Dataset/'

soict2023_textrecog_train = dict(
    type='OCRDataset',
    data_root=soict2023_textrecog_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='crnn_train_gt.json',
    pipeline=None)

soict2023_textrecog_test = dict(
    type='OCRDataset',
    data_root=soict2023_textrecog_data_root,
    data_prefix=dict(img_path='new_train/'),
    ann_file='crnn_val_gt.json',
    test_mode=True,
    pipeline=None)
