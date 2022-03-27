_base_ = ['./xnet_model_PointPillar_SECOND_ResNet_Fusion_kitti-3d-car.py']

model = dict(
    fusion_layer=dict(
        type='XNetFusion',
        img_channels=256,
        pts_channels=256,
        out_channels=256)
)