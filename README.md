# Zero-X21
Code of paper: Zero-X21: Scale-agnostic image feature conditioned INR for multi-modal andmulti-planar anisotropic MRI inter-slice interpolation



## GPU
Model was trained on a 3090 GPU


## Citation

```
@article{MA2025236,
title = {Zero-X21: Scale-agnostic image feature conditioned INR for multi-modal and multi-planar anisotropic MRI inter-slice interpolation},
journal = {Pattern Recognition Letters},
volume = {196},
pages = {236-242},
year = {2025},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2025.06.001},
url = {https://www.sciencedirect.com/science/article/pii/S0167865525002296},
author = {Zibo Ma and Jianfei Huo and Guanchun Yin and Bo Zhang and Xiuzhuang Zhou and Zhen Cui and Wendong Wang},
keywords = {MRI inter-slice interpolation, Implicit Neural Representation, Multi-modal image feature interaction},
abstract = {Combining different planes of multi-contrast anisotropic Magnetic Resonance Imaging (MRI) into high-resolution isotropic and multi-contrast MRI can provide richer diagnostic information. However, this is challenging due to the inherent inconsistencies in image features and other aspects across different views. This challenge can be referred to as the inter-slice interpolation problem of multi-modal and multi-planar anisotropic MRI. Currently, Implicit Neural Representations (INR) offer the advantage of handling arbitrary up-sampling scales (integer or fractional). However, existing INR-based super-resolution methods often suffer from limitations, including poor generalization ability, inadequate multi-modal information interaction, and limited capacity for image feature modulation. To address these limitations, we propose Zero-X21, a novel image feature-conditioned INR framework specifically designed for the inter-slice interpolation problem of anisotropic MRI. Leveraging the inherent continuity of INRs, the Zero-X21 framework excels in achieving high-quality results across arbitrary up-sampling scales, surpassing other volumetric super-resolution methods. Experimental results on a brain MRI dataset demonstrate that the Zero-X21 framework achieves state-of-the-art performance for inter-slice interpolation. Notably, a single trained Zero-X21 model can effectively handle arbitrary up-sampling scales, making it a versatile and efficient solution for this challenging task.}
}
```


