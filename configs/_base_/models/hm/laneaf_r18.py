# model settings
norm_cfg = dict(type='BN', requires_grad=True)

act_cfg = dict(type="ReLU")
model = dict(
    type='LaneSeg',
    # pretrained='open-mmlab://resnet50_v1c',
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
        in_channels=[64, 128, 256, 512], # s4, s8, s16, s32
        in_index=[0, 1, 2, 3],
        channels=512,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        align_corners=False,
        num_classes=10,  # Fake One
        loss_decode=dict(
            type='AFLoss', loss_type="wbce", ignore_label=255)),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
