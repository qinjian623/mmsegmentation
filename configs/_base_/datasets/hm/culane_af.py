# dataset settings
dataset_type = 'CULane'
data_root = '/home/data/culane'

# Private Dataset, no pipeline
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        path=data_root,
        image_set='train',
        random_transforms=True
    ),
    val=dict(
        type=dataset_type,
        path=data_root,
        image_set='val',
        random_transforms=True
    ),
    test=dict(
        type=dataset_type,
        path=data_root,
        image_set='test',
        random_transforms=True
    )
)
