![LOGO](https://github.com/DeepWave-KAUST/FWI_Conditional-CNN_UQ/blob/main/asset/workflow-1.png)

Reproducible material for **Conditional Image Prior for Uncertainty Quantification in Full Waveform Inversion-Lingyun Yang, Omar M. Saad, Guochen wu and Tariq Alkhalifah**


# Project structure
This repository is organized as follows:

* :open_file_folder: **data**: folder containing data (or instructions on how to retrieve the data;
* :open_file_folder: **Result**: folder containing marmousi model FWI results;
* :open_file_folder: **FIELD_data**: folder contain field data application.

## Notebooks
The following notebooks are provided:

- :orange_book: ``Marmousi_ContestUnet.ipynb``: Pretraining stage for the conditional CNN using 50 particel;
- :orange_book: ``Marmousi_FWI-part.ipynb``:  Performing UQ FWI using the pre-trained conditional CNN.


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the following pip environment.

Simply run:
```
conda create -n deepwaveold_env python=3.7
conda activate deepwaveold_env
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install deepwave==0.0.8
pip install gstools
pip install torchsummary
pip install scikit-learn
pip install pylops
pip install hydra-core
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 


**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
Lingyun Yang, Omar M. Saad, Tariq Alkhalifah, et al. 2025. Conditional image prior for uncertainty quantification in full waveform inversion. International Meeting for Applied Geoscience and Energy.

