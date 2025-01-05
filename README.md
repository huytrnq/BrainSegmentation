# Brain Segmentation
This repository is a collection of scripts and notebooks for brain segmentation using deep learning which is part of the project for the course "Medical Image Segmentation" at the University of Girona on my Master's degree.

## Table of Contents
1. [Introduction](#introduction)
2. [Data](#data)
3. [Preprocessing](#preprocessing)
4. [Model](#model)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [References](#references)

## Introduction
This project aims to segment the brain from MRI images using deep learning

## Data
The dataset used in this study is the IBSR18, containing 18 T1-weighted scans of normal subjects from the Internet Brain Segmentation Repository (IBSR). It includes preprocessed scans with a 1.5 mm slice thickness and ground truth segmentation for white matter (WM), gray matter (GM), and cerebrospinal fluid (CSF).

![IBSR18](./images/IBSR18.png)
<p align="center">Figure 1: IBSR18 dataset</p>

## Preprocessing

### Slice-Based Segmentation
- Description: Processes 2D slices extracted from 3D brain MRI volumes for segmentation tasks.
- Steps:
    - Normalize intensity values for consistent pixel distributions.
    - Apply data augmentation techniques (e.g., rotations, flips) to enhance training diversity.

### Full Volume Segmentation
- Description: Directly processes full 3D MRI volumes to retain spatial context and provide higher segmentation accuracy.
- Steps:
    - Normalize intensity values.
    - Apply 3D augmentations such as affine transformations and bias field adjustments.

### Patch-Based Segmentation
- Description: Focuses on smaller patches extracted from 3D volumes, balancing computational efficiency with model accuracy.
- Steps:
    - Extract fixed-size patches using uniform or foreground-focused sampling.
    - Normalize patches and apply augmentations.
    - Weighted sampling prioritizes regions with brain tissue to improve sensitivity.

## Model
The repository includes several deep learning architectures evaluated for brain MRI segmentation:
- 2D Models: U-Net and EfficientNet-based models tailored for slice-based segmentation.
- 3D Models: Includes 3D U-Net, UNETR, and SegResNet for volumetric segmentation, leveraging spatial context across slices.
- Patch-Based Models: Designed to process small 3D patches with emphasis on critical regions, such as CSF, GM, and WM, while mitigating class imbalance.

## Training
The training pipeline supports:
- Custom loss functions, including Dice Loss, Focal Loss, and a combination of both.
- Weighted sampling during patch-based training to focus on foreground regions.
- Dynamic learning rate adjustment using Cosine Annealing.
- Integration with MLflow and DagsHub for experiment tracking, logging, and reproducibility.

## Evaluation
- Metrics:
    - Dice Score: Measures overlap between predictions and ground truth.
    - Hausdorff Distance (HD): Evaluates boundary matching.
    - Patch-Based Inference: Employs grid sampling and aggregation to reconstruct full-volume predictions from patch outputs.

## Results

## References