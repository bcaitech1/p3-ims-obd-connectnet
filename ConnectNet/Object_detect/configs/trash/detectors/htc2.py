_base_ = [
    './htc_without_semantic_r50_fpn_1x_coco.py',
    '../dataset_mask.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            pretrained='torchvision://resnet50',
            style='pytorch')))



checkpoint_config = dict(interval=1)
optimizer = dict(lr=0.01)
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = "/opt/ml/code/mmdetection_trash/work_dirs/htc_dectors_r50_x1_trash/epoch_48.pth"