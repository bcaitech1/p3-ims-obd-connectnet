_base_ = [
    '../dataset/dataset_k2.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

norm_cfg=dict(type='SyncBN', momentum=0.01,
                eps=1e-3, requires_grad=True)

model = dict(
    type='RetinaNet',
    #pretrained='efficientnet-b2-8bb594d6.pth',
    pretrained = True,
    backbone=dict(
        type='EfficientNet',
        model_type='efficientnet-b0',
        out_indices=(0, 1, 2, 3)
        #out_levels=[3, 4, 5],
        #norm_cfg=norm_cfg,
        #norm_eval=False,
    
    ),
    neck=dict(
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048, 4096],
        out_channels=256,
        num_outs=5),

    bbox_head=dict(
        type='RetinaHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0))
)

# model = dict(
#     type='RetinaNet',
#     pretrained=True,
#     backbone=dict(
#         type='EfficientNet',
#         model_type='efficientnet-b0',  # Possible types: ['efficientnet-b0' ... 'efficientnet-b7']
#         out_indices=(0, 1, 3, 6)),  # Possible indices: [0 1 2 3 4 5 6],