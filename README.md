# DA-MUSIC: DATA-DRIVEN DOA ESTIMATION VIA DEEP AUGMENTED MUSIC ALGORITHM

More information will follow shortly.


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
 
