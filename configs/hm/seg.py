_base_ = [
    '../_base_/models/hm/deeplabv3_unet_s5-d16-c14.py', '../_base_/datasets/hm/seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
find_unused_parameters=True
evaluation = dict(metric='mIoU')
