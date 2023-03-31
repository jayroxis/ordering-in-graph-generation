
# This is the Semantic Scene Graph dataset

# Credits to:
#   Panoptic Scene Graph on COCO-2017 Dataset
#   PSG Challenge: https://github.com/Jingkang50/OpenPSG


import torch
import torch.nn.functional as F
from torchvision.transforms import Resize

from data.openpsg import PanopticSceneGraphDataset

from .misc import *
from timm.models.registry import register_model



class BasePSGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        ann_file: str, 
        data_root: str,
        test_mode: bool = False,
        split: str = 'train',
        sort_func: str = "no_sort",
        oversample_thr: float = 0.05,
        **kwargs,
    ) -> None:
        """
        Base Panoptic Scene Graph Dataset. This dataset reads and output
        the full data as dictionary.

        For use of only a small part of the data. Please inherit from this
        class.
        """
        super().__init__()

        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True
        )

        # sequence sorting
        self.sort_func = eval(sort_func)

        # Train Pipeline From PSGFormer-ResNet50
        # https://github.com/Jingkang50/OpenPSG/blob/main/configs/psgformer/psgformer_r50_psg.py    
        self.train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadPanopticSceneGraphAnnotations',
                with_bbox=True,
                with_rel=True,
                with_mask=False,
                with_seg=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[
                    [
                        dict(type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                        (576, 1333), (608, 1333), (640, 1333),
                                        (672, 1333), (704, 1333), (736, 1333),
                                        (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True)
                    ],
                    [
                        dict(type='Resize',
                            img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True),
                        dict(type='RelRandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=False),  # no empty relations
                        dict(type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                        (576, 1333), (608, 1333), (640, 1333),
                                        (672, 1333), (704, 1333), (736, 1333),
                                        (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True)
                    ]
                ]),
            dict(type='Normalize', **self.img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='RelsFormatBundle'),
            dict(type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels']
            )
        ]

        self.test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadSceneGraphAnnotations', 
                with_bbox=True,
                with_rel=True,
                with_mask=False,
                with_seg=False),
            dict(type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **self.img_norm_cfg),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels', 'gt_rels']),
                    dict(type='ToDataContainer',
                        fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
                    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels']),
                ]
            )
        ]

        # set pipeline
        self.test_mode = test_mode
        if test_mode:
            pipeline = self.test_pipeline
        else:
            pipeline = self.train_pipeline

        self.dataset = PanopticSceneGraphDataset(
            ann_file=ann_file, 
            pipeline=pipeline, 
            data_root=data_root,
            test_mode=test_mode,
            split=split,
            all_bboxes=True
        )

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)
        return data
    


@register_model
class PSGRelationDataset(BasePSGDataset):
    """
    This Child dataset only outputs images and scene graph
    relations.

    Returns:
        - 4D `img` of shape (B, C, W, H).
        - One-hot encoded relational triple-lets of shape (B, L, 3).
            The relational triple-lets are (Object, Predicate, Object)
    """
    def __init__(self, img_size: int = 384, one_hot: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.resize = Resize(size=(img_size, img_size))
        self.one_hot = one_hot
        self.obj_cls = len(self.dataset.CLASSES)
        self.pd_cls = len(self.dataset.PREDICATES)

    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)

        # load image
        img = data["img"]
        if type(img) == tuple or type(img) == list:
            img = torch.cat(img, dim=0)
        img = img.data.float()
        img = self.resize(img)

        # load relational graph
        rels = data["gt_rels"]
        if type(rels) == tuple or type(rels) == list:
            rels = torch.cat(rels, dim=0)
        rels = rels.data.long()
        rels = self.sort_func(rels)

        # map the segment id to semantic id
        labels = data["gt_labels"]
        if type(labels) == tuple or type(labels) == list:
            labels = torch.cat(labels, dim=0)
        seg_id_to_obj_id = {
            i: l.item() for i, l in enumerate(labels.data)
        }
        for r in rels:
            r[0] = seg_id_to_obj_id[r[0].item()]
            r[1] = seg_id_to_obj_id[r[1].item()]

        # remove duplicates for semantic graphs
        rels = torch.unique(rels, dim=0)
    
        # one-hot encoding
        if self.one_hot:
            one_hot_rels = torch.cat([
                F.one_hot(rels[..., 0], num_classes=self.obj_cls + 1).float(),
                F.one_hot(rels[..., 1], num_classes=self.obj_cls + 1).float(),
                F.one_hot(rels[..., 2] - 1, num_classes=self.pd_cls + 1).float(), # need to - 1
            ], dim=-1)
            return img, one_hot_rels
        else:
            return img, rels