
<h1 align="center"> Bi-Objective Meta-Learning for Interpretable Computerized Adaptive Testing</h1>


Welcome to the official repository for MICAT containing all requirements for scientific reproducibility of the paper experiments.

## 1.Installing MICAT

**Datasets** for experiments can be downloaded at the following url. The two folders "1-raw_data" and "2-preprocessed_data" must be placed in the folder 'experiments/datasets/':

[Datasets_url](https://zenodo.org/records/16729674?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwMTY5ZmY4LTBiZWEtNGM4Zi05ZmQ3LTAzNGQ3ODczN2YyMCIsImRhdGEiOnt9LCJyYW5kb20iOiJkYjY3N2Q0MmRhNDcxZjQ2ZjM3ZmNjMDg2NDMzNDNjZSJ9.wgsEm5rb9dgxhdG2LVvA_zMk7aZqQQcXQaVjiIopc9ld19b79fgyPo_uQrJNvKrF2mxACVlDdX8QzAdFuOXe7Q)

**Pretrained CDMs parameters** for experiments can be downloaded at the following url. The folder 'ckpt/' must be placed in the folder 'experiments/':

[Parameters_url](https://zenodo.org/records/16733971?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImEwM2IwY2U0LWE0ZjQtNDI1NC04YjhiLTdmN2QyNWI3NThlYyIsImRhdGEiOnt9LCJyYW5kb20iOiI5MDJlYjhhOTA5YzMxM2UzNjk1Nzc3YzA4N2U3N2E4MyJ9.qFqSbcq9_XqkS8GlVkUuCH3b_vcXieRkO-d3QhIX9NJVoE1Xt-tTNxDojwy65SsCMdWEv8Bmry6oDhnZ66xInw)


The **conda environment** supporting the libraries can be created using the Makefile:

```bash
make install
conda activate micat-env
```

## 2.Reproducing paper experiments

Paper's experiments can be replicated using the jupyter notebook situated in: 'experiments/notebook_example/paper_experiments.ipynb'.

## 3.Requirements
- Linux OS
- conda package manager
- CUDA version >= 12.4
- **pytorch for CUDA** (to install with pip in accordance with your CUDA version : [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/))


## 4.Repository map
- `experiments/` : Contains the jupyter notebooks and datasets to run the experiments of the scientific paper.
    - `experiments/ckpt/` : Folder for CDM parameters saving
    - `experiments/data/` : Folder containing hyperparameters of CDM, MICAT and  its competitors, as well as saved embeddings for paper figures.
    - `experiments/datasets/` : Contains the raw and pre-processed datasets, as well as there pre-processing jupyter notebook
    - `experiments/embs/` : Folder for user embeddings saving
    - `experiments/hyperparam_search/` : Contains the csv files of the optimal hyperparameter for each method (obtained with Tree-structured Parzen Estimator (TPE) sampler)
    - `experiments/logs/` : Folder for running logs saving
    - `experiments/notebook_example/` : Contains the jupyter notebooks to run the experiments of the scientific paper, including competitors. 
    - `experiments/preds/` : Folder for predictions saving
- `figs/` : Contains the figures of the paper
- `micat/` : Contains the source code of the MICAT model
  - `micat/CDM/` : Contains the code of the CDM
  - `micat/dataset/` : Contains the code of the dataset class
  - `micat/meta_models/` : Contains extra functions for BETA-CD code
  - `micat/selectionStrategy/` : Contains the core component of CAT, **MICAT** and its competitors
  - `micat/utils/` : Contains utility functions for logging, complex metric computations, configuration handling, etc.

