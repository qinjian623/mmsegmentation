_base_ = [
    '../_base_/models/hm/laneaf_r18.py', '../_base_/datasets/hm/culane_af.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
find_unused_parameters = True
model = dict(test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))
# evaluation = dict(metric='mDice')
evaluation = {}
# evaluation = dict(interval=1, metric='mDice')

