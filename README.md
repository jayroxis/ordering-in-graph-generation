# GraphGPT
A minimal implementation of GraphGPT for Vision-Graph generation.

# Commands
To monitor jobs on ARC: 
```cmd
watch -n1 "squeue -u jayroxis --format=\"%.10i %.10P %.30j %.8u %.8T %.8M %.10l %12R\""
```

# Datasets

## Topological Graph Generation From Visual Inputs
**Image to Topological Graph** (Proposed in This Work): [Planar_Graph](data/planar_graph.py).

***Description***: This is a synthetic dataset used in this paper for learning a 2D topoligical planar graph from its rendered image. Models are expected to predict not only the graph connectivity but also the node coordinates.

## Scene Graph Dataset For Visual Understanding
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

## Molecule Dataset For Molecule Structure Prediction
**Molecular Sets** (Based on ZINC): [MOSES](https://graphgt.github.io/molecule.html).

***Description***: Molecular Sets (MOSES) is a benchmark platform for distribution learning based molecule generation. Within this benchmark, MOSES provides a cleaned dataset of molecules that are ideal of optimization. It is processed from the ZINC Clean Leads dataset.

***Acknowledgement:***
```bibtex
@article{polykovskiy2020molecular,
    title={Molecular sets (MOSES): a benchmarking platform for molecular generation models},
    author={Polykovskiy, Daniil and Zhebrak, Alexander and Sanchez-Lengeling, Benjamin and Golovanov, Sergey and Tatanov, Oktai and Belyaev, Sergey and Kurbanov, Ruslan and Artamonov, Andrew and Aladinskiy, Vladimir and Veselov, Mark and others},
    journal={Frontiers in pharmacology},
    volume={11},
    year={2020},
    publisher={Frontiers Media SA}
}

@inproceedings{du2021graphgt,
    title={GraphGT: Machine Learning Datasets for Graph Generation and Transformation},
    author={Du, Yuanqi and Wang, Shiyu and Guo, Xiaojie and Cao, Hengning and Hu, Shujie and Jiang, Junji and Varala, Aishwarya and Angirekula, Abhinav and Zhao, Liang},
    booktitle={NeurIPS 2021},
    year={2021}
} 
```

## Circuit Graph For Automatic EDA


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
# Notes

### Why Do We Adopt **Binary Cross-Entropy (BCE)** Loss Instead of **Cross-Entropy (CE)** Loss on Scene Graph Tasks.

A good reading material: [Mosaic ResNet Deep Dive](https://www.mosaicml.com/blog/mosaic-resnet-deep-dive).

The Binary-CE loss, as shown in [ResNet Strikes Bac [1]](https://arxiv.org/abs/2110.00476) and [Beyer et al., 2020 [2]](https://arxiv.org/abs/2006.07159), has demonstrate it as a better to-go option than the Multiclass-CE loss.

```bibtex
@article{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm},
  author={Wightman, Ross and Touvron, Hugo and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2110.00476},
  year={2021},
  archivePrefix = {arXiv},
  eprint={2110.00476}
}
```
```bibtex
@article{beyer2020imagenet,
  title={Are we done with ImageNet?},
  author={Beyer, Lucas and H{\'e}naff, Olivier J and Kolesnikov, Alexander and Zhai, Xiaohua and van den Oord, Aaron},
  journal={arXiv preprint arXiv:2006.07159},
  year={2020},
  archivePrefix = {arXiv},
  eprint={2006.07159}
}
```
### Other Possible Improvements:

BlurPool, EMA, FixRes, Label Smoothing, and Progressive Image Resizing; it also uses the FFCV dataloader and Channels Last memory format.
