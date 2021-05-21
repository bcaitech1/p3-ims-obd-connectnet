dataset_type = 'CocoDataset'
data_root = '../../input/data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type="RandomRotate90", p=1.0),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomBrightnessContrast", brightness_limit=0.25, contrast_limit=0.25),
            dict(type="RGBShift", r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.4,
    ),
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.3,
        rotate_limit=5,
        scale_limit=(-0.3, 0.75),
        border_mode=0,
        value=img_norm_cfg["mean"][::-1],
    ),
    #dict(type="RandomBBoxesSafeCrop", num_rate=(0.5, 1.0), erosion_rate=0.2)
    dict(
        type='RandomSizedBBoxSafeCrop',
        height = 512,
        width = 512,
        erosion_rate = 0.2,
        p=0.3
    )
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.3,
            filter_lost_elements=True)
    ),
    dict(type="Mixup", p=0.25, min_buffer_size=2, pad_val=tuple(img_norm_cfg["mean"][::-1])),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize', 
        img_scale=[(512 + 64 * i, 512 + 64 * i) for i in range(5)],
        multiscale_mode="value",
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        #type="ModifiedMultiScaleFlipAug",
        type ="MultiScaleFlipAug",
        img_scale=[(512, 512), (768, 768)],
        flip=True,
        flip_direction=["horizontal", "vertical"],
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))