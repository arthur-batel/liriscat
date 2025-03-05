
<p align="center"><img src="figs/LIRISCAT_logo.png" alt="logo" height="300"/></p>

<h1 align="center"> An interpretable model for multi-target predictions with multi-class outputs </h1>

---
Welcome to the official repository for LIRISCAT

## Installing IMPACT
From source
```bash
git clone https://github.com/arthur-batel/liriscat.git
cd liriscat
make install
conda activate liriscat-env
# open one of the notebooks in the experiments/notebook_examples folder
```


## Requirements
- Linux OS
- conda package manager
- CUDA version >= 12.4
- **pytorch for CUDA** (to install with pip in accordance with your CUDA version : [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/))


## Repository map
- `experiments/` : Contains the jupyter notebooks and datasets to run the experiments of the scientific paper.
    - `experiments/ckpt/` : Folder for models parameter saving
    - `experiments/datasets/` : Contains the raw and pre-processed datasets, as well as there pre-processing jupyter notebook
    - `experiments/embs/` : Folder for user embeddings saving
    - `experiments/hyperparam_search/` : Contains the csv files of the optimal hyperparameter for each method (obtained with Tree-structured Parzen Estimator (TPE) sampler)
    - `experiments/logs/` : Folder for running logs saving
    - `experiments/notebook_example/` : Contains the jupyter notebooks to run the experiments of the scientific paper, including competitors. 
    - `experiments/preds/` : Folder for predictions saving
    - `experiments/tensorboard/` : Folder for tensorboard data saving
- `figs/` : Contains the figures of the paper
- `liriscat/` : Contains the source code of the IMPACT model
  - `liriscat/dataset/` : Contains the code of the dataset class
  - `liriscat/models/` : Contains the code of the **LIRISCAT model** and its abstract class, handling the learning process
  - `liriscat/utils/` : Contains utility functions for logging, complex metric computations, configuration handling, etc.
## Authors

Arthur Batel,
arthur.batel@insa-lyon.fr,
INSA Lyon, LIRIS UMR 5205 FR

Marc Plantevit,
marc.plantevit@epita.fr,
EPITA Lyon, EPITA Research Laboratory (LRE) FR

Idir Benouaret,
idir.benouaret@epita.fr,
EPITA Lyon, EPITA Research Laboratory (LRE) FR

Céline Robardet,
celine.robardet@insa-lyon.fr,
INSA Lyon, LIRIS UMR 5205 FR

