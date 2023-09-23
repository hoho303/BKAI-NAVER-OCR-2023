auto_scale_lr = dict(base_batch_size=512)
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
    '/workspace/mmocr/configs/textrecog/master/../../../dicts/187_vietnamese.txt',
    same_start_end=True,
    type='Dictionary',
    with_end=True,
    with_padding=True,
    with_start=True,
    with_unknown=True)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmocr/textrecog/master/master_resnet31_12e_st_mj_sa/master_resnet31_12e_st_mj_sa_20220915_152443-f4a5cabc.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        arch_channels=[
            256,
            256,
            512,
            512,
        ],
        arch_layers=[
            1,
            2,
            5,
            3,
        ],
        block_cfgs=dict(
            plugins=dict(
                cfg=dict(
                    fusion_type='channel_add',
                    is_att_scale=False,
                    n_head=1,
                    pooling_type='att',
                    ratio=0.0625,
                    type='GCAModule'),
                position='after_conv2'),
            type='BasicBlock'),
        in_channels=3,
        init_cfg=[
            dict(layer='Conv2d', type='Kaiming'),
            dict(layer='BatchNorm2d', type='Constant', val=1),
        ],
        plugins=[
            dict(
                cfg=dict(kernel_size=2, stride=(
                    2,
                    2,
                ), type='Maxpool2d'),
                position='before_stage',
                stages=(
                    True,
                    True,
                    False,
                    False,
                )),
            dict(
                cfg=dict(
                    kernel_size=(
                        2,
                        1,
                    ), stride=(
                        2,
                        1,
                    ), type='Maxpool2d'),
                position='before_stage',
                stages=(
                    False,
                    False,
                    True,
                    False,
                )),
            dict(
                cfg=dict(
                    act_cfg=dict(type='ReLU'),
                    kernel_size=3,
                    norm_cfg=dict(type='BN'),
                    padding=1,
                    stride=1,
                    type='ConvModule'),
                position='after_stage',
                stages=(
                    True,
                    True,
                    True,
                    True,
                )),
        ],
        stem_channels=[
            64,
            128,
        ],
        strides=[
            1,
            1,
            1,
            1,
        ],
        type='ResNet'),
    data_preprocessor=dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='TextRecogDataPreprocessor'),
    decoder=dict(
        attn_drop=0.0,
        d_inner=2048,
        d_model=512,
        dictionary=dict(
            dict_file=
            '/workspace/mmocr/configs/textrecog/master/../../../dicts/187_vietnamese.txt',
            same_start_end=True,
            type='Dictionary',
            with_end=True,
            with_padding=True,
            with_start=True,
            with_unknown=True),
        feat_pe_drop=0.2,
        feat_size=240,
        ffn_drop=0.0,
        max_seq_len=30,
        module_loss=dict(
            ignore_first_char=True, reduction='mean', type='CEModuleLoss'),
        n_head=8,
        n_layers=3,
        postprocessor=dict(type='AttentionPostprocessor'),
        type='MasterDecoder'),
    encoder=None,
    type='MASTER')
naver_data_root = '/workspace/data/'
naver_rec_test = dict(
    ann_file='full_train_gt.json',
    data_prefix=dict(img_path='new_train/'),
    data_root='/workspace/data/',
    pipeline=None,
    test_mode=True,
    type='OCRDataset')
naver_rec_train = dict(
    ann_file='full_train_gt.json',
    data_prefix=dict(img_path='new_train/'),
    data_root='/workspace/data/',
    pipeline=None,
    test_mode=False,
    type='OCRDataset')
optim_wrapper = dict(
    optimizer=dict(lr=0.0004, type='Adam'), type='OptimWrapper')
param_scheduler = [
    dict(by_epoch=False, end=100, type='LinearLR'),
    dict(end=12, milestones=[
        11,
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
                ann_file='full_train_gt.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                height=48,
                max_width=160,
                min_width=48,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=160),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_dataset = dict(
    datasets=[
        dict(
            ann_file='full_train_gt.json',
            data_prefix=dict(img_path='new_train/'),
            data_root='/workspace/data/',
            pipeline=None,
            test_mode=True,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            height=48,
            max_width=160,
            min_width=48,
            type='RescaleToHeight',
            width_divisor=16),
        dict(type='PadToWidth', width=160),
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
        'NAVER',
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
        ann_file='full_train_gt.json',
        data_prefix=dict(img_path='new_train/'),
        data_root='/workspace/data/',
        pipeline=None,
        test_mode=True,
        type='OCRDataset'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        height=48,
        max_width=160,
        min_width=48,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='PadToWidth', width=160),
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
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=100)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        datasets=[
            dict(
                ann_file='full_train_gt.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=False,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
            dict(type='LoadOCRAnnotations', with_text=True),
            dict(
                height=48,
                max_width=160,
                min_width=48,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=160),
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
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_dataset = dict(
    datasets=[
        dict(
            ann_file='full_train_gt.json',
            data_prefix=dict(img_path='new_train/'),
            data_root='/workspace/data/',
            pipeline=None,
            test_mode=False,
            type='OCRDataset'),
    ],
    pipeline=[
        dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            height=48,
            max_width=160,
            min_width=48,
            type='RescaleToHeight',
            width_divisor=16),
        dict(type='PadToWidth', width=160),
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
        ann_file='full_train_gt.json',
        data_prefix=dict(img_path='new_train/'),
        data_root='/workspace/data/',
        pipeline=None,
        test_mode=False,
        type='OCRDataset'),
]
train_pipeline = [
    dict(ignore_empty=True, min_size=2, type='LoadImageFromFile'),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        height=48,
        max_width=160,
        min_width=48,
        type='RescaleToHeight',
        width_divisor=16),
    dict(type='PadToWidth', width=160),
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
                dict(
                    height=48,
                    max_width=160,
                    min_width=48,
                    type='RescaleToHeight',
                    width_divisor=16),
            ],
            [
                dict(type='PadToWidth', width=160),
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
                ann_file='full_train_gt.json',
                data_prefix=dict(img_path='new_train/'),
                data_root='/workspace/data/',
                pipeline=None,
                test_mode=True,
                type='OCRDataset'),
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                height=48,
                max_width=160,
                min_width=48,
                type='RescaleToHeight',
                width_divisor=16),
            dict(type='PadToWidth', width=160),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    dataset_prefixes=[
        'NAVER',
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
work_dir = '/workspace/mmocr/workdir/master'
