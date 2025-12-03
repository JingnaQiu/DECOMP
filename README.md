# DECOMP

This repository provides the source code for our WACV 2026 paper **"Decomposition Sampling for Efficient Region Annotations in Active Learning"**
[[`arXiv`
](https://arxiv.org/abs/2407.06363)] [[`blogpost`](https://deepmicroscopy.org/leveraging-image-captions-for-streamlining-histopathology-image-annotation-miccai-2024-paper/)].

Abstract: Active learning improves annotation efficiency by selecting the most informative samples for annotation and model training. While most prior work has focused on selecting informative images for classification tasks, we investigate the more challenging setting of dense prediction, where annotations are more costly and time-intensive, especially in medical imaging. Region-level annotation has been shown to be more efficient than image-level annotation for these tasks. However, existing methods for representative annotation region selection suffer from high computational and memory costs, irrelevant region choices, and heavy reliance on uncertainty sampling. We propose **decomposition sampling (DECOMP)**, a new active learning sampling strategy that addresses these limitations. It enhances annotation diversity by decomposing images into class-specific components using pseudo-labels and sampling regions from each class. Class-wise predictive confidence further guides the sampling process, ensuring that difficult classes receive additional annotations. Across ROI classification, 2-D segmentation, and 3-D segmentation, DECOMP consistently surpasses baseline methods by better sampling minority-class regions and boosting performance on these challenging classes.

-----------------------------------------------------------------------
This repository includes a full implementation of DECOMP, as well as code for applying the method to the BRACS evaluation dataset.

## Installation
Environment setup instructions, along with known issues and their solutions, are provided in [install.txt](install.txt).

## Usage
```python
python experiments.py --exp-id=0 --image-sampling-strategy="decomposition_threshold_0.7" --region-sampling-strategy="decomposition_threshold_0.7" --n-query=1 --max-query-per-WSI=15 
```
