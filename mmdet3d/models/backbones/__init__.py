# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import ResNet
from .second import SECOND
from .second_info_ver import SECOND_INFO

__all__ = [
    'ResNet', 'SECOND', 'SECOND_INFO'
]
