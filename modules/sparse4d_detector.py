# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch

from inspect import signature
from modules.cnn.base_detector import BaseDetector
from modules.backbone import *
from modules.neck import *
from modules.head import *
from modules.head.sparse4d_blocks.core_blocks import *

from tool.runner.fp16_utils import force_fp32, auto_fp16

try:
    from .ops import feature_maps_format

    DFA_VALID = True
except:
    DFA_VALID = False

### 对外只暴露Sparse4D这个类
__all__ = ["Sparse4D"]


class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        img_neck,
        head,
        depth_branch=None,
        use_grid_mask=True,
        use_deformable_func=False,
        init_cfg=None,
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)

        # =========== build modules ===========
        
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        ### resnet 部分
        self.img_backbone = build_module(img_backbone)

        ### FPN部分
        if img_neck is not None:
            self.img_neck = build_module(img_neck)
            
        self.head = build_module(head)
        
        if use_deformable_func:
            assert DFA_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        
        if depth_branch is not None:
            self.depth_branch = build_module(depth_branch)
        else:
            self.depth_branch = None

        ### GridMask 数据增强
        self.use_grid_mask = use_grid_mask
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    ### 装饰器的作用是：当它装饰的函数被调用时，它会将img参数自动转换为16位浮点数，然后调用原函数；
    ### 函数执行完毕后，再将结果转换回32位浮点数，最后返回这个结果
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        ### 获得输入图像的batch
        bs = img.shape[0]  # (1, 6, 3, 256, 704)

        ### 获得相机数、扁平batch cam_numn维
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            ### 0到1维度进行扁平化
            img = img.flatten(end_dim=1)  # [1*6, 3, 256, 704]
        else:
            num_cams = 1
            
        ### 训练过程中数据增强    
        if self.use_grid_mask:
            img = self.grid_mask(img)

        ### img_backbone.forward的参数列表中含有metas，则传入metas和num_cams参数;否则，只传图
        ### 看img_backbone的设计
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)

        ### feature_maps经过backbone输出如下：
        ## (1*6, 256, 64, 176)
        ## (1*6, 512, 32, 88)
        ## (1*6, 1024, 16, 44)
        ## (1*6, 2048, 8, 22)

        ### 这里转换成一系列的特征图，即多尺度 多相机
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
            
        ### enumerate函数会返回每个特征图的索引i和特征图本身feat
        for i, feat in enumerate(feature_maps):
        ### reshape为（batch,cam_nums，channel,w,h)
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])

        ### feature_maps经过neck和reshape输出如下：
        ## (1, 6, 256, 64, 176) float16
        ## (1, 6, 256, 32, 88) float16
        ## (1, 6, 256, 16, 44) float16
        ## (1, 6, 256, 8, 22) float16
            
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
        ### 测试分支
            depths = None
            
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        ### 输出
        ## (1, 89760, 256） torch.float16
        ## (6, 4, 2) torch.int64
        ## (6, 4) torch.int64

        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
