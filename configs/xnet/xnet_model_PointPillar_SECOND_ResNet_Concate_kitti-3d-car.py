# 需要修改的配置文件
_base_ = ['./xnet_model_Voxel_SECOND_ResNet_Fusion_kitti-3d-car.py']

# model settings
voxel_size = [0.05, 0.05, 0.1]

model = dict(
    pts_voxel_layer=dict(
        max_num_points=32,  # max_points_per_voxel
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432])
)
