

# Panoptic Scene Graph on COCO-2017 Dataset

import torch

from data.openpsg import PanopticSceneGraphDataset
from mmdet.datasets.dataset_wrappers import ClassBalancedDataset

from timm.models.registry import register_model


class BasePSGDataset(torch.utils.data.Dataset):
    def __init__(self, 
        ann_file: str, 
        data_root: str,
        test_mode: bool = False,
        split: str = 'train',
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

        # Train Pipeline From PSGFormer-ResNet50
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

        dataset = PanopticSceneGraphDataset(
            ann_file=ann_file, 
            pipeline=self.train_pipeline, 
            data_root=data_root,
            test_mode=test_mode,
            split=split,
        )
        self.dataset = ClassBalancedDataset(
            dataset=dataset,
            oversample_thr=0.01, 
        )

    def __len__(self, idx):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)
        return data
    


@register_model
class PSGRelationDataset(BasePSGDataset):
    """
    This Child dataset only outputs images and scene graph
    relations.
    """
    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)
        img = data["img"].data.float()
        rels = data["gt_rels"].data.long()
        return img, rels