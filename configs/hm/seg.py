_base_ = [
    '../_base_/models/hm/deeplabv3_unet_s5-d16-c14.py', '../_base_/datasets/hm/seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
find_unused_parameters=True
evaluation = dict(metric='mDice')
