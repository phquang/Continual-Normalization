# Continual Normalization: Rethinking Batch Normalization for Online Continual Learning

This project contains the implementation of the following ICLR 2022 paper: 

**Title:** Continual Normalization: Rethinking Batch Normalization for Online Continual Learning (ICLR 2022), [[openreview]](https://openreview.net/forum?id=vwLLQ-HwqhZ)

**Authors:** [Quang Pham](https://sites.google.com/view/quangpham93), [Chenghao Liu](https://sites.google.com/view/liuchenghao/home), and [Steven Hoi](https://sites.google.com/view/stevenhoi/home)


## Overview
Continual Normalization (CN) is a simple, yet effective normalization strategy specially deveoped for the **online continual learning** problem. CN is highly compatible with state-of-the-art experience replay based methods and offers improvements over the traditional Batch Normalization strategy.

## Usage

CN is simple and easy to implement. A standalone implementation of CN is provded in the ``cn.py`` file.

## Replacing BN with CN in existing models

In many experiments, it is more convenient to consider a pre-built or pre-trained models and replace its BN layer with our CN, while keeping the pre-trained affine transformation parameters. We provide an utility function to do so and a working example in the ``example.py`` file.

## Replicating the Online Class Incremental Learning Experiments

Lastly, to replicate the Online Class Incremental Learning Experiments, please follow the instructions in the ``mammoth/`` folder.

## Citing CN

If you found our work to be useful, please consider citing as
```
@inproceedings{pham2021continual,
  title={Continual Normalization: Rethinking Batch Normalization for Online Continual Learning},
  author={Pham, Quang and Liu, Chenghao and Steven, HOI},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
