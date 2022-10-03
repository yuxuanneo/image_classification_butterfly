# dataset settings
dataset_type = 'CustomDataset'
classes = ['monarch', 'tiger', 'black', 'pipevine', 'viceroy', 'spicebush']  # The category names in dataset

# img_norm_cfg = dict(
#     mean=[115.512, 118.991, 77.582],
#     std=[25.019, 24.291, 26.483],
#     to_rgb=False)

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224), # augmentation process
    dict(type='Resize', size=224), # resize to fit input size of resnet pre-trained on ImageNet
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'), 
    dict(type='Normalize', **img_norm_cfg),   
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),   
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128, # batchsize
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/processed_data/train',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/processed_data/val',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/processed_data/val',
        classes=classes,
        pipeline=test_pipeline
    )
)

evaluation = dict(interval=10, metric='accuracy')
