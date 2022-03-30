# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_car-multimodal.py', 
    './model/xnet_base-SECOND_ResNet_NoFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py', 
    '../_base_/default_runtime.py'
    ]


point_cloud_range = [0, -40, -3, 70.4, 40, 1]
model = dict(
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True)),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            _delete_=True,
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            allowed_border=0,
            pos_weight=-1,
            debug=False)))