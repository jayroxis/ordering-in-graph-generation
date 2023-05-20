# GraphGPT
A minimal implementation of GraphGPT for Vision-Graph generation.

## Note On PyTorch Lightning Environment
- **Do NOT use FP16 on PyTorch Lightning**, it may cause buggy checkpoint behavior. As we notice, it can cause a known bug that could lead to very poor model performance.
- Though this paper's results were produced using the pipeline with PyTorch Lightning 1.8.0, Ubuntu LTS 20.0.4 and Python 3.10, we observed that some systems and environments may have some major problems that the validation accuracy are significantly lower. Yet it is still unpredictable what are the environments that may cause this issue from PyTorch Lightning. If you want to use the model, we highly recommend you to implement a version without using PyTorch Lightning.

**References To The Issues:** 
- https://github.com/Lightning-AI/lightning/issues/6159
- https://github.com/Lightning-AI/lightning/issues/924
- https://github.com/Lightning-AI/lightning/issues/4045

**Information About The Apex FP16 Bug:**
- https://github.com/Lightning-AI/lightning/issues/525#issuecomment-596963253


## Datasets


### Topological Graph Generation From Visual Inputs


**Toulouse Road Network Dataset**: [Road_Network](https://github.com/davide-belli/generative-graph-transformer).

***Description***: The Toulouse Road Network dataset is a collection of road maps from the city of Toulouse, represented both as graphs and grayscale segmentation images, and is used to benchmark the Generative Graph Transformer, with its creation involving multiple steps of preprocessing, filtering, data augmentation, and graph representation.

```bibtex
@article{belli2019image,
  title={Image-conditioned graph generation for road network extraction},
  author={Belli, Davide and Kipf, Thomas},
  journal={arXiv preprint arXiv:1910.14388},
  year={2019}
}
```

**Image to Topological Graph** (Proposed in This Work): [Planar_Graph](data/planar_graph.py).

***Description***: This is a synthetic dataset used in this paper for learning a 2D topoligical planar graph from its rendered image. Models are expected to predict not only the graph connectivity but also the node coordinates.

### Scene Graph Dataset For Visual Understanding
**Panoptic Scene Graph Dataset** (Based On COCO-2017): [PSG](https://github.com/Jingkang50/OpenPSG).

***Description***: The Panoptic Scene Graph Generation (PSG) Task aims to interpret a complex scene image with a scene graph representation, with each node in the scene graph grounded by its pixel-accurate segmentation mask in the image. To promote comprehensive scene understanding, the authors take into account all the content in the image, including "things" and "stuff", to generate the scene graph. However, this work does not involve object detection or segmentation.

***Acknowledgement:***
```bibtex
@inproceedings{yang2022psg,
    author = {Yang, Jingkang and Ang, Yi Zhe and Guo, Zujin and Zhou, Kaiyang and Zhang, Wayne and Liu, Ziwei},
    title = {Panoptic Scene Graph Generation},
    booktitle = {ECCV}
    year = {2022}
}
```

### Circuit Graph For Automatic EDA


**Electronic Circuit Graph Dataset**: [ECGD](https://github.com/hehaodele/circuit-gnn).

***Description***: Description: ECGD is a specialized resource designed for graph-based learning and optimization in the field of electronic circuit design. The dataset contains structured electronic circuit data represented as graphs, capturing the relationships between resonators in a circuit. ECGD is organized by the number of circuit blocks and circuit types, sourced from diverse real-world electronic circuit configurations.

```bibtex
@inproceedings{he2019circuit,
  title={Circuit-GNN: Graph neural networks for distributed circuit design},
  author={Zhang, Guo and He, Hao and Katabi, Dina},
  booktitle={International Conference on Machine Learning},
  pages={7364--7373},
  year={2019}
}
```
## Notes

### Why Do We Adopt **Binary Cross-Entropy (BCE)** Loss Instead of **Cross-Entropy (CE)** Loss on Scene Graph Tasks.

A good reading material: [Mosaic ResNet Deep Dive](https://www.mosaicml.com/blog/mosaic-resnet-deep-dive).

The Binary-CE loss, as shown in [ResNet Strikes Back [1]](https://arxiv.org/abs/2110.00476) and [Are we done with ImageNet? [2]](https://arxiv.org/abs/2006.07159), has demonstrate it as a better to-go option than the Multiclass-CE loss.

```bibtex
# ResNet Strikes Back
@article{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2110.00476},
  year={2021},
  archivePrefix = {arXiv},
  eprint={2110.00476}
}
# Are we done with ImageNet?
@article{beyer2020imagenet,
  title={Are we done with ImageNet?},
  author={Beyer, Lucas and H{\'e}naff, Olivier J and Kolesnikov, Alexander and Zhai, Xiaohua and van den Oord, Aaron},
  journal={arXiv preprint arXiv:2006.07159},
  year={2020},
  archivePrefix = {arXiv},
  eprint={2006.07159}
}
```
