
_base_ = [
    '../_base_/models/', '../_base_/datasets/drive.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
act_cfg = dict(type="ReLU")
log_level = 0
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='AFHead',
        input_transform='multiple_select',
        in_channels=[256, 512, 1024],
        in_index=[1, 2, 3],
        channels=512,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        align_corners=False,
        num_classes=10, # Fake One
        loss_decode=dict(
            type='AFLoss', loss_type="wbce")),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
