# GraphGPT
A minimal implementation of GraphGPT for Vision-Graph generation.

# Datasets

## Scene Graph Dataset For Visual Understanding
PSG (Based On COCO-2017)

## Molecule Dataset For Molecule Structure Prediction
MOSES (Based on ZINC)

## Circuit Graph For Automatic EDA
https://github.com/hehaodele/circuit-gnn


# Notes

### Why Do We Adopt **Binary Cross-Entropy (BCE)** Loss Instead of **Cross-Entropy (CE)** Loss on Scene Graph Tasks.

A good reading material: [Mosaic ResNet Deep Dive](https://www.mosaicml.com/blog/mosaic-resnet-deep-dive).

The Binary-CE loss, as shown in [ResNet Strikes Bac [1]](https://arxiv.org/abs/2110.00476) and [Beyer et al., 2020 [2]](https://arxiv.org/abs/2006.07159), has demonstrate it as a better to-go option than the Multiclass-CE loss.

- [1] Wightman, R., Touvron, H., & J'egou, H. (2021). ResNet strikes back: An improved training procedure in timm. ArXiv, abs/2110.00476.

- [2] Beyer, L., H'enaff, O.J., Kolesnikov, A., Zhai, X., & Oord, A.V. (2020). Are we done with ImageNet? ArXiv, abs/2006.07159.

### Other Possible Improvements:

BlurPool, EMA, FixRes, Label Smoothing, and Progressive Image Resizing; it also uses the FFCV dataloader and Channels Last memory format.
