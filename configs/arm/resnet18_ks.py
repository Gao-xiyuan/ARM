_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/av_ks.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

find_unused_parameters=True

# model settings
pretrained = None

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
    ),
    head=dict(num_classes=31, ))

train_dataloader = dict(batch_size=128)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
)
