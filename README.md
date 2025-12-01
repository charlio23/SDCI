# SDCI: Causal Discovery from Conditionally Stationary Time series

[![ArXiv](https://img.shields.io/static/v1?label=arXiv&message=2110.06257&color=B31B1B)](https://arxiv.org/abs/2110.06257) [![Venue:ICML 2025](https://img.shields.io/badge/Venue-ICML_2025-007CFF)](https://icml.cc)

This repository contains source code for SDCI, the method proposed for conditionally stationary time series

## Setup

The source code is build on Python 3.10 and the models are implemented using Pytorch 1.13.1 with cuda 11.7.

Run `conda create -n <env-name> python=3.10`

Run `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html` to install the dependencies.

Change torch and torch-scatter versions at convenience. Consider removing torch-scatter if this causes problems, and remove it from the code if not using `att` or `rnn` edge encoders.


## Testing the code

Below we provide simple indications to test the code. Please find examples in the `scripts` folder.

We also provide sample datasets and sample models [on this link](https://drive.google.com/drive/folders/1ZFDBmDFUuaAjb30W0iOR3QoobFVTi_1I?usp=sharing).

### Prepare downloaded datasets and models

Copy the zip file to the repository. 

Run ``unzip <zipfile>``.

``mv Data\ ICML\ 2025/models_saved/ ..``
``mv Data\ ICML\ 2025/datasets/* datasets/``


## Springs and Magnets Data

### Generate Data

Alternatively, you can generate data with using ``datasets/generate_data_spring.py``. An example is shown in `scripts/generate_data_script.sh`

``python datasets/generate_data_spring.py --state-type collision --num-train 1000 --num-valid 1 --num-test 100 --n_balls 5 --length 8100 --box-size 1 --n_states 2 --n_edge_types 3 --temperature 0.5 --datadir springs_3_var_collision_80_strong_2_states_fixed --fixed_connectivity --seed 24``

This generates the dataset with 5 balls, 3 types (no-edge, spring, magnet), fixed edges across samples, 2 states, and transition on wall collision. This dataset is used in our paper Figure 4, right.

For collision data (states recurrent) use flag ``--state-type collision``. 

For region data (states determined) use flag ``--state-type region``.

### Train

Code for training is found in ``train_springs_hidden.py``. For examples, see `scripts/run_springs_*`.

We describe the different options SDCI offers in case you are interested in using it as a baseline.

#### Edge encoders

We use different GNN architectures across our paper depending on the setup. This can be controlled by the ``--encoder`` flag. See below:
- ``--encoder fixed``: Fixed encoder $q(\mathcal{W}| \boldsymbol{x}_{1:T}):= q(\mathcal{W})$. The model outputs **a single graph** across the dataset.
- ``--encoder mlp``: Variable graphs across samples, it uses the original ACD and NRI MLP implementations adapted to state-dependent causal structures. We use this for region data.
- ``--encoder rnn``: Variable graphs across samples, and based on NRI-NPM RNN edge-edge aggregation. We do not use this, but we tested it when tuning architectures.
- ``--encoder att``: Variable graphs across samples, and based on NRI-NPM Attention edge-edge aggregation. We use this for collision data.

#### State encoders

Depending on whether we consider `states determined` or `states recurrent` setting, we use the following encoders and priors with the flag ``--state-decoder``.
- ``state-decoder region``: We consider a state decoder based on the last observation $q(s_t^{i}|\boldsymbol{x}_t^{(i)})$. Here we define $q(s_t^{i}|\boldsymbol{x}_t^{(i)})=p(s_t^{i}|\boldsymbol{x}_t^{(i)})$ in the variational approximation for convenience (state-KL term goes to 0).
- ``state-decoder recurrent``: We consider a GNN-RNN to encode states `StateGRNNEncoderSmall`, and define a separate prior `StateDecoder` to learn dynamics for forecasting.


**Note:** For baseline testing, we recommend using region-based encoders, as they are more efficient.

#### States and Edges sampling

Learning discrete latent spaces is challenging, as the dynamics result in high-variance during training. For this, we provide two sampling settings.

- ``sampler gumble`` Original categorical reparemetrisation from Jang et al. 2017, and used in ACD, NRI and NRI-NPM. 
- ``sampler simple`` Straigh-Through Gumbel-softmax alternative, where gradients are fixed to pass through unperturbed marginals. Proposed by Ahmed et al., 2023, this helps reduce variance, which is critical in variable graph settings, especially in collision type data. We always fix this setting for sampling states.


### Measure performance

Code for measuring performance can be found in ``measure_performance.py``. It accounts for all permutation alignments.

For an example with data and models saved, run ``sh scripts/measure_performance_springs.sh``

## GRN

We provide data for GRN. You can test SDCI by running the following command.

``sh scripts/run_GRN_region_embedding.sh``


For additional datasets, please contact the corresponding author.


## Misc

We provide graph visualisation notebooks in `notebooks`. Some of these visualisations are part of the main paper.

## Citation

If you use this code, or you find our paper useful for your research, consider citing our paper.
```@inproceedings{
balsells-rodas2025causal,
title={Causal Discovery from Conditionally Stationary Time Series},
author={Carles Balsells-Rodas and Xavier Sumba and Tanmayee Narendra and Ruibo Tu and Gabriele Schweikert and Hedvig Kjellstrom and Yingzhen Li},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=j88QAtutwW}
}
```