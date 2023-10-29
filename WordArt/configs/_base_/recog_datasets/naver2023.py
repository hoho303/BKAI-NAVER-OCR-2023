# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k

train_root = '/workspace/data'

train_img_prefix1 = f'{train_root}/new_train'
train_ann_file1 = f'{train_root}/train_gt.txt'

train1 = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=False)

train_list = [train1]

test_root = '/workspace/data'

test_img_prefix1 = f'{test_root}/new_train'
test_ann_file1 = f'{test_root}/train_gt.txt'

test1 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        file_storage_backend='disk',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=True)

test_list = [test1]
