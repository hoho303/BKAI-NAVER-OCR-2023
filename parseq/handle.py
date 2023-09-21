import torch

# epoch
# global_step
# pytorch-lightning_version
# state_dict
# loops
# callbacks
# optimizer_states
# lr_schedulers
# MixedPrecisionPlugin
# hparams_name
# hyper_parameters 0.0001187661717580843

checkpoint = torch.load('/mlcv/WorkingSpace/Personals/ngocnd/parseq/outputs/parseq/2023-08-25_07-52-20/checkpoints/epoch=67-step=18238-val_accuracy=99.9748-val_NED=99.9923.ckpt')
print(checkpoint['lr_schedulers'])
checkpoint = torch.load('/mlcv/WorkingSpace/Personals/ngocnd/parseq/outputs/parseq/2023-08-25_07-52-20/checkpoints/last.ckpt')
print(checkpoint['lr_schedulers'])