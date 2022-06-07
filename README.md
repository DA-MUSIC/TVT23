# DA-MUSIC: DATA-DRIVEN DOA ESTIMATION VIA DEEP AUGMENTED MUSIC ALGORITHM

[DA-MUSIC: Data-Driven DoA Estimation via Deep Augmented MUSIC](https://ieeexplore.ieee.org/document/9746637)

## Abstract

Direction of arrival (DoA) estimation of multiple signals is pivotal in sensor array signal processing. A popular multi-signal DoA estimation method is the multiple signal classification (MUSIC) algorithm, which enables high-performance super-resolution DoA recovery while being highly applicable in practice. MUSIC is a model-based algorithm, relying on an accurate mathematical description of the relationship between the signals and the measurements and assumptions on the signals themselves (non-coherent, narrowband sources). As such, it is sensitive to model imperfections. In this work we propose to overcome these limitations of MUSIC by augmenting the algorithm with specifically designed neural architectures. Our proposed deep augmented MUSIC (DA-MUSIC) algorithm is thus a hybrid model-based/data-driven DoA estimator, which leverages data to improve performance and robustness while preserving the interpretable flow of the classic method. DA-MUSIC is shown to learn to overcome limitations of the purely model-based method, such as its inability to successfully localize coherent sources as well as estimate the number of coherent signal sources present. We further demonstrate the superior resolution of the DA-MUSIC algorithm in synthetic narrowband and broadband scenarios as well as with real-world data of DoA estimation from seismic signals.


## Overview

This repository consists of following Python scripts:
* The `augMUSIC.py` [ARTEFACT] was used to implement previous versions of DA-MUSIC.
* The `bbSynthEx.py` script implements synthetic examples for broadband DoA estimation.
* The `beamformer.py` implements the classic beamforming algorithm.
* The `broadbandMUSIC.py` implements a model-based incoherent broadband MUSIC algorithm.
* The `classicMUSIC.py` implements the purely model-based MUSIC algorithm.
* The `errorMeasures.py` defines error measures used to evaluate the DoA estimation algorithms.
* The `layers.py` script defines custom layers for the neural networks.
* The `losses.py` script defines custom losses used to train neural augmentations for the MUSIC algorithm.
* The `models.py` defines neural augmentation architectures for the MUSIC algorithm.
* The `plotFigures.py` provides visualization of the performances of different DoA algortihms.
* The `regularizers.py` [ARTEFACT] script defines custom regularizers for the neural augmentations.
* The `syntheticEx.py` script implements synthetic examples for DoA and combines them to a datase.
* The `trainModel.py` implements the training of the neural augmentation.
* The `utils.py` defines some helpful functions.


## Requirements

| Module | Version |
| :--- | :---: |
| scipy  | 1.6.2  |
| h5py  | 2.10.0 |
| pandas  | 0.25.1 |
| matplotlib  | 3.1.1 |
| numpy  | 1.19.3 |
| tensorflow  | 2.4.1 |
| tqdm  | 4.36.1 |
| scikit_learn | 1.1.1 |
| seaborn | 0.9.0 |
 
