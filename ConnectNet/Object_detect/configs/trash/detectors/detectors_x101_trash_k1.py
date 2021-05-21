_base_ = [
    './cascade_rcnn_x101_32x4d_fpn_1x_coco.py',
    '../dataset/dataset_k2.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNeXt',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))




seed=2020
optimizer_config = dict( _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

optimizer = dict(lr=0.01)

lr_config = dict(step=[43, 47])
runner = dict(type='EpochBasedRunner', max_epochs=50)

checkpoint_config = dict(max_keep_ckpts=12, interval=1)

#gpu_ids = [0]
#work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

#data = dict(samples_per_gpu=2)

#optimizer = dict(lr=0.01)

# lr_config = dict(step=[43, 47])
# runner = dict(type='EpochBasedRunner', max_epochs=48)


# checkpoint_config = dict(interval=1)
# optimizer = dict(lr=0.01)
# lr_config = dict(step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12)

#load_from = "/opt/ml/code/mmdetection_trash/work_dirs/detectors_r50_x1_trash_basic/epoch_48.pth"



