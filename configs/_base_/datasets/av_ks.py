# dataset settings
dataset_type = 'AV_KS_Dataset'
data_preprocessor = dict(
    num_classes=31,
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)


train_pipeline = [
    dict(type='MyLoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='MyLoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='/root/dataset/ks/image/train',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file="/root/dataset/ks/train_list_noempty.txt",  # 相对于 `data_root` 的标注文件路径
        split = 'train',
        classes=['blowing_nose','blowing_out_candles', 'bowling', 'chopping_wood', 'dribbling_basketball', 'finger_snapping','laughing', 'mowing_lawn', 'playing_accordion',
                 'playing_bagpipes', 'playing_bass_guitar', 'playing_clarinet', 'playing_drums', 'playing_guitar', 'playing_harmonica', 'playing_keyboard', 'playing_organ',
                 'playing_piano', 'playing_saxophone', 'playing_trombone', 'playing_trumpet', 'playing_violin', 'playing_xylophone', 'ripping_paper', 'shoveling_snow',
                 'shuffling_cards', 'singing', 'tap_dancing', 'tapping_guitar', 'tapping_pen', 'tickling'],  # 每个类别的名称
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='/root/dataset/ks/image/val',  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
        ann_file="/root/dataset/ks/val_list_noempty.txt",  # 相对于 `data_root` 的标注文件路径
        split = 'test',
        classes=['blowing_nose', 'blowing_out_candles', 'bowling', 'chopping_wood', 'dribbling_basketball',
                 'finger_snapping', 'laughing', 'mowing_lawn', 'playing_accordion',
                 'playing_bagpipes', 'playing_bass_guitar', 'playing_clarinet', 'playing_drums', 'playing_guitar',
                 'playing_harmonica', 'playing_keyboard', 'playing_organ',
                 'playing_piano', 'playing_saxophone', 'playing_trombone', 'playing_trumpet', 'playing_violin',
                 'playing_xylophone', 'ripping_paper', 'shoveling_snow',
                 'shuffling_cards', 'singing', 'tap_dancing', 'tapping_guitar', 'tapping_pen', 'tickling'],  # 每个类别的名称
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
