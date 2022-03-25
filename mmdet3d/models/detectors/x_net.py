# 检测器，主要工程入口，根据配置文件，构建模型。

import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from .. import builder


@DETECTORS.register_module()
class DynamicMVXFasterRCNNDev(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self,
                 fusion_layer=None,
                 **kwargs):
        super(DynamicMVXFasterRCNNDev, self).__init__(**kwargs)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points): # 注：这里用不到img，重写父类方法
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_feats, feature_coors = self.pts_voxel_encoder(voxels, coors, points)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_feats, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points)
        pts_feats = self.fusion_layer(pts_feats, img_feats)
        return (img_feats, pts_feats)