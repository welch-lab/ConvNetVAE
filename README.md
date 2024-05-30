<img src="docs/img/fig_graph_abs.png">

# ConvNet-VAE: Integrating single-cell multimodal epigenomic data using 1D-convolutional neural networks

## Introduction

The parallel profiling of histone modification and chromatin states at single-cell resolution provides a means to quantify the epigenetic landscape and delineate cell heterogeneity. A pivotal step in this process is the integration of these epigenomic modalities to learn a unified representation of each cell for accurate cell type inference. Here, we present `ConvNet-VAE`s, a novel framework based on 1D-convolutional variational autoencoders (VAEs), tailored for single-cell multimodal epigenomic data integration. We evaluated `ConvNet-VAE`s on data generated from juvenile mouse brain and human bone marrow. Through benchmark analyses, we demonstrated that `ConvNet-VAE`s, despite utilizing substantially fewer parameters, deliver performance that is superior or comparable to traditional VAEs equipped with fully connected layers (`FC-VAE`s) in terms of dimension reduction and batch-effect correction.

## Environment setup

```
conda env create -f env_config.yml
conda activate convnetvae
conda install ipykernel
python -m ipykernel install --user --name convnetvae --display-name "Python3 (cnn)"
```

## Run models

Jupyter notebooks are provided for demo the [mouse brain data](https://www.dropbox.com/scl/fo/0zuu6irftualnwzd3izqv/h?rlkey=0ifqexeqql21nxaamqqmth678&dl=0).
* ConvNet-VAE: `demo_ConvNetVAE_nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3.ipynb`
* FC-VAE: `demo_FCVAE_nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3.ipynb`

To run cross validation, follow the sample code below.\
ConvNet-VAE:
 ```
cd ConvNetVAE/convnetvae
python run_convNetVAE.py \
--dataSet nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_bin10kb \
--countDist Poisson \
--numFeature 25000 \
--modalityList ATAC H3K27ac H3K27me3 \
--includeWNN \
--numSample 2 \
--batchSize 128 \
--dimLatent 30 \
--numHiddenUnits 128 \
--numChannel 32 64 \
--sizeConvKernel 11 \
--sizeStrideEnc 3 \
--dropoutRate 0.2 \
--learningRate 1e-3 \
--numKFolds 5 \
--foldStratifyOption '' \
--numEpochs 300 \
--resList 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 \
--beta 1 \
--randomSeed 2023 \
 ```
FC-VAE:
 ```
cd ConvNetVAE/convnetvae
python run_convNetVAE.py \
--dataSet nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_bin10kb \
--countDist Poisson \
--numFeature 25000 \
--modalityList ATAC H3K27ac H3K27me3 \
--includeWNN \
--numSample 2 \
--batchSize 128 \
--dimLatent 30 \
--numHiddenUnits 128 \
--numHiddenLayers 1 \
--dropoutRate 0.2 \
--learningRate 1e-3 \
--numKFolds 5 \
--foldStratifyOption '' \
--numEpochs 300 \
--resList 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 \
--beta 1 \
--randomSeed 2023 \
 ```


