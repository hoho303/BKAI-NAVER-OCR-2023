auto_scale_lr = dict(base_batch_size=128)
default_hooks = dict(
    checkpoint=dict(interval=50, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
dictionary = dict(
    dict_file=
    '/workspace/mmocr/configs/textrecog/abinet/../../../dicts/vietnam_digits_space.txt',
    same_start_end=True,
    type='Dictionary',
    with_end=True,
    with_padding=False,
    with_start=True,
    with_unknown=False)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_20e_st-an_mj/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(type='ResNetABI'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        d_model=512,
        dictionary=dict(
            dict_file=
            '/workspace/mmocr/configs/textrecog/abinet/../../../dicts/vietnam_digits_space.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=False,
            with_start=True,
            with_unknown=False),
        language_decoder=dict(
            d_inner=2048,
            d_model=512,
            detach_tokens=True,
            dropout=0.1,
            n_head=8,
            n_layers=4,
            type='ABILanguageDecoder',
            use_self_attn=False),
        max_seq_len=26,
        module_loss=dict(letter_case='unchanged', type='ABIModuleLoss'),
        num_iters=3,
        postprocessor=dict(type='AttentionPostprocessor'),
        type='ABIFuser',
        vision_decoder=dict(
            attn_height=8,
            attn_mode='nearest',
            attn_width=32,
            in_channels=512,
            init_cfg=dict(layer='Conv2d', type='Xavier'),
            num_channels=64,
            type='ABIVisionDecoder')),
    encoder=dict(
        d_inner=2048,
        d_model=512,
        dropout=0.1,
        max_len=256,
        n_head=8,
        n_layers=3,
        type='ABIEncoder'),
    type='ABINet')
naver_data_root = '/workspace/data/'
naver_rec_test = dict(
    ann_file='abi_val_gt_2.json',
    data_prefix=dict(img_path='new_train/'),
    data_root='/workspace/data/',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
naver_rec_train = dict(
    ann_file='abi_train_gt_2.json',
    data_prefix=dict(img_path='new_train/'),
    data_root='/workspace/data/',
    pipeline=None,
    test_mode=False,
    type='OCRDataset')
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='Adam'), type='OptimWrapper')
param_scheduler = [
    dict(
        convert_to_iter_based=True, end=2, start_factor=0.001,
        type='LinearLR'),
    dict(end=20, milestones=[
        16,
        18,
    ], type='MultiStepLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='abi_val_gt_2.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                128,
                32,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='abi_val_gt_2.json',
            data_prefix=dict(img_path='new_train/'),
            data_root='/workspace/data/',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(scale=(
            128,
            32,
        ), type='Resize'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
test_evaluator = dict(
    dataset_prefixes=[
        'NAVER2023',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
test_list = [
    dict(
        ann_file='abi_val_gt_2.json',
        data_prefix=dict(img_path='new_train/'),
        data_root='/workspace/data/',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        128,
        32,
    ), type='Resize'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        datasets=[
            dict(
                ann_file='abi_train_gt_2.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=False,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(scale=(
                128,
                32,
            ), type='Resize'),
            dict(
                prob=0.5,
                transforms=[
                    dict(
                        transforms=[
                            dict(max_angle=15, type='RandomRotate'),
                            dict(
                                degrees=15,
                                op='RandomAffine',
                                scale=(
                                    0.5,
                                    2.0,
                                ),
                                shear=(
                                    -45,
                                    45,
                                ),
                                translate=(
                                    0.3,
                                    0.3,
                                ),
                                type='TorchVisionWrapper'),
                            dict(
                                distortion_scale=0.5,
                                op='RandomPerspective',
                                p=1,
                                type='TorchVisionWrapper'),
                        ],
                        type='RandomChoice'),
                ],
                type='RandomApply'),
            dict(
                prob=0.25,
                transforms=[
                    dict(type='PyramidRescale'),
                    dict(
                        transforms=[
                            dict(
                                p=0.5, type='GaussNoise', var_limit=(
                                    20,
                                    20,
                                )),
                            dict(blur_limit=7, p=0.5, type='MotionBlur'),
                        ],
                        type='mmdet.Albu'),
                ],
                type='RandomApply'),
            dict(
                prob=0.25,
                transforms=[
                    dict(
                        brightness=0.5,
                        contrast=0.5,
                        hue=0.1,
                        op='ColorJitter',
                        saturation=0.5,
                        type='TorchVisionWrapper'),
                ],
                type='RandomApply'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='abi_train_gt_2.json',
            data_prefix=dict(img_path='new_train/'),
            data_root='/workspace/data/',
            pipeline=None,
            test_mode=False,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(scale=(
            128,
            32,
        ), type='Resize'),
        dict(
            prob=0.5,
            transforms=[
                dict(
                    transforms=[
                        dict(max_angle=15, type='RandomRotate'),
                        dict(
                            degrees=15,
                            op='RandomAffine',
                            scale=(
                                0.5,
                                2.0,
                            ),
                            shear=(
                                -45,
                                45,
                            ),
                            translate=(
                                0.3,
                                0.3,
                            ),
                            type='TorchVisionWrapper'),
                        dict(
                            distortion_scale=0.5,
                            op='RandomPerspective',
                            p=1,
                            type='TorchVisionWrapper'),
                    ],
                    type='RandomChoice'),
            ],
            type='RandomApply'),
        dict(
            prob=0.25,
            transforms=[
                dict(type='PyramidRescale'),
                dict(
                    transforms=[
                        dict(p=0.5, type='GaussNoise', var_limit=(
                            20,
                            20,
                        )),
                        dict(blur_limit=7, p=0.5, type='MotionBlur'),
                    ],
                    type='mmdet.Albu'),
            ],
            type='RandomApply'),
        dict(
            prob=0.25,
            transforms=[
                dict(
                    brightness=0.5,
                    contrast=0.5,
                    hue=0.1,
                    op='ColorJitter',
                    saturation=0.5,
                    type='TorchVisionWrapper'),
            ],
            type='RandomApply'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'valid_ratio',
            ),
            type='PackTextRecogInputs'),
    ],
    type='ConcatDataset')
train_list = [
    dict(
        ann_file='abi_train_gt_2.json',
        data_prefix=dict(img_path='new_train/'),
        data_root='/workspace/data/',
        pipeline=None,
        test_mode=False,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(scale=(
        128,
        32,
    ), type='Resize'),
    dict(
        prob=0.5,
        transforms=[
            dict(
                transforms=[
                    dict(max_angle=15, type='RandomRotate'),
                    dict(
                        degrees=15,
                        op='RandomAffine',
                        scale=(
                            0.5,
                            2.0,
                        ),
                        shear=(
                            -45,
                            45,
                        ),
                        translate=(
                            0.3,
                            0.3,
                        ),
                        type='TorchVisionWrapper'),
                    dict(
                        distortion_scale=0.5,
                        op='RandomPerspective',
                        p=1,
                        type='TorchVisionWrapper'),
                ],
                type='RandomChoice'),
        ],
        type='RandomApply'),
    dict(
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                transforms=[
                    dict(p=0.5, type='GaussNoise', var_limit=(
                        20,
                        20,
                    )),
                    dict(blur_limit=7, p=0.5, type='MotionBlur'),
                ],
                type='mmdet.Albu'),
        ],
        type='RandomApply'),
    dict(
        prob=0.25,
        transforms=[
            dict(
                brightness=0.5,
                contrast=0.5,
                hue=0.1,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'valid_ratio',
        ),
        type='PackTextRecogInputs'),
]
tta_model = dict(type='EncoderDecoderRecognizerTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=0, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=1, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
                dict(
                    condition="results['img_shape'][1]<results['img_shape'][0]",
                    true_transforms=[
                        dict(
                            args=[
                                dict(cls='Rot90', k=3, keep_size=False),
                            ],
                            type='ImgAugWrapper'),
                    ],
                    type='ConditionApply'),
            ],
            [
                dict(scale=(
                    128,
                    32,
                ), type='Resize'),
            ],
            [
                dict(type='LoadOCRAnnotations', with_text=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'valid_ratio',
                    ),
                    type='PackTextRecogInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                ann_file='abi_val_gt_2.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                128,
                32,
            ), type='Resize'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'valid_ratio',
                ),
                type='PackTextRecogInputs'),
        ],
        type='ConcatDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'NAVER2023',
    ],
    metrics=[
        dict(
            mode=[
                'exact',
                'ignore_case',
                'ignore_case_symbol',
            ],
            type='WordMetric'),
        dict(type='CharMetric'),
    ],
    type='MultiDatasetsEvaluator')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextRecogLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/workspace/mmocr/workdir/abinet2'
