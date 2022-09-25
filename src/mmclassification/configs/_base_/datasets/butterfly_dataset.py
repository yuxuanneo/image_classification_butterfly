# dataset settings
dataset_type = 'CustomDataset'
classes = ['monarch', 'tiger', 'black', 'pipevine', 'viceroy', 'spicebush']  # The category names in dataset

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224), # augmentation process
    dict(type='Resize', size=224), # resize to fit input size of resnet pre-trained on ImageNet
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='CenterCrop', crop_size=224),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128, # batchsize
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/processed_data/train',
        # ann_file='data/processed_data/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/processed_data/val',
        # ann_file='data/processed_data/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/test',
        # ann_file='data/my_dataset/meta/test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=1, metric='accuracy')
