# 需要修改的配置文件
_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    type='XNet',

    # 点云特征提取 Point Cloud Feature Extraction
    pts_voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFEDev',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),

    # 图像特征提取 Image Feature Extraction
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        type='XNetFusion',
        img_channels=256,
        pts_channels=256,
        out_channels=256),

    # 解码器

    # 点云目标检测检测头 Point Cloud Object Detection Head
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),

    # 图像目标检测检测头 Image Object Detection Head
    img_rpn_head=None,
    img_roi_head=None,

    # 训练配置 model training setting
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # for Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),

    # 测试配置 model testing settings
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)))

# 数据集配置 dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(use_lidar=True, use_camera=True)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='Resize',
        img_scale=[(640, 192), (2560, 768)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

# Training settings
optimizer = dict(weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

evaluation = dict(interval=1, pipeline=eval_pipeline)

# You may need to download the model first is the network is unstable
load_from = None
