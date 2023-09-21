_base_ = [
    '../_base_/datasets/naver2023.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_sar_resnet31_parallel-decoder.py',
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/sar/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real/sar_resnet31_sequential-decoder_5e_st-sub_mj-sub_sa_real_20220915_185451-1fd6b1fc.pth"


train_cfg = dict(max_epochs=50, val_interval = 100)

# dataset settings
train_list = [
    _base_.naver_rec_train
]
test_list = [
    _base_.naver_rec_test
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['NAVER2023'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 8)