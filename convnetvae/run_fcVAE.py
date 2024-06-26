# line ending: unix

import os
import argparse
import pandas as pd
import numpy as np
import umap
import random

from sklearn import preprocessing
from sklearn.metrics import adjusted_rand_score
#from sknetwork.clustering import Louvain
from sklearn.model_selection import StratifiedKFold, KFold
#from sklearn import metrics
from sklearn.utils import shuffle

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import torch.optim as optim

import time
from tqdm import tqdm

# model
from models import * 
from utilities import *
# util functions
from plotting import *


parser = argparse.ArgumentParser(description='Model Hyperparameters')
parser.add_argument('--dataSet', type=str, metavar='DS',
                    help='dataset to analyze')
parser.add_argument('--countDist', type=str, metavar='CD',
                    help='modeling distribution for the data ')
parser.add_argument('--numFeature', type=int, default=5000, metavar='F',
                    help='number of features')
parser.add_argument('--modalityList', nargs='+', type=str, metavar='M',
                    help='number of modalities')
parser.add_argument('--includeWNN', action='store_true',
                    help='whether to include WNN labels for evaluation')
parser.add_argument('--numSample', type=int, default=5000, metavar='NS',
                    help='number of samples')
parser.add_argument('--batchSize', type=int, default=128, metavar='B',
                    help='batch size (default: 128)')
parser.add_argument('--dimLatent', type=int, default=20, metavar='D',
                    help='dimension of latent space (default: 20)')
parser.add_argument('--numHiddenLayers', type=int, default=1, metavar='L',
                    help='number of hidden layers (default: 1)')
parser.add_argument('--numHiddenUnits', type=int, default=128, metavar='H',
                    help='number of hidden units (default: 128)')
# parser.add_argument('--numChannel', type=int, default=1, metavar='Ch',
#                     help='number of hidden layers (default: 1)')
parser.add_argument('--dropoutRate', type=float, default=0.2, metavar='DR',
                    help='dropout rate (default: 0.2)')
parser.add_argument('--learningRate', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--numKFolds', type=int, default=10, metavar='K',
                    help='number of folds for cross-validation (default: 5)')
parser.add_argument('--foldStratifyOption', type=nullable_string, metavar='STR',
                    help='Strategy to stratify K folds')
parser.add_argument('--numEpochs', type=int, default=100, metavar='E',
                    help='number of epochs (default: 100)')
parser.add_argument('--resList', nargs='+', type=float, metavar='R',
                    help='list of resolution for clustering')
parser.add_argument('--beta', type=float, default=1, metavar='BT',
                    help='beta for VAE (default: 1)')
parser.add_argument('--randomSeed', type=int, default=123, metavar='RDS',
                    help='random seed for reproducible research (default: 123)')


args = parser.parse_args()
dataSet=args.dataSet
countDist = args.countDist
numFeature=args.numFeature
modalityList=args.modalityList
includeWNN=args.includeWNN
numSample=args.numSample
batchSize=args.batchSize
dimLatent=args.dimLatent
numHiddenLayers= args.numHiddenLayers
numHiddenUnits=args.numHiddenUnits
dropoutRate=args.dropoutRate
learningRate=args.learningRate
numKFolds=args.numKFolds
foldStratifyOption=args.foldStratifyOption
numEpochs=args.numEpochs
resList=args.resList
beta=args.beta
randomSeed=args.randomSeed
numModality=len(modalityList)

print(f'Dataset to look at: {dataSet}')
print(f'Count distribution: {countDist}')
print(f'Number of features: {numFeature}')
print(f'Modalities: {numModality}')
print(f'WNN included for evaluation: {includeWNN}')
print(f'Number of samples: {numSample}')
print(f'Dimension of latent space: {dimLatent}')
print(f'Number of hidden units in dense layer: {numHiddenUnits}')
print(f'Number of hidden layers in encoder/decoder: {numHiddenLayers}')
print(f'Dropout rate for training: {dropoutRate}')
print(f'Learning rate: {learningRate}')
print(f'Number of folds for cross-validation: {numKFolds}')
print(f'K folds splits based on annotation: {foldStratifyOption}')
print(f'Number of epochs for training: {numEpochs}')
print(f'Size of mini-batch: {batchSize}')
print(f'Resolution list for Louvain clustering: {resList}')
print(f'Random seed for reproducibility: {randomSeed}')

# Select proper model architecture
if numModality == 1:
    VAE = fcVAE
if numModality == 2:
    VAE = poeExp2VAE
if numModality == 3:
    VAE = poeExp3VAE

# dataset to load
if dataSet == 'pbmc_GEX':
    input_prep = input_prep_pbmc_GEX
    batch_correct = ''
if dataSet == 'pbmc_ATAC_fragment':
    input_prep = input_prep_pbmc_ATAC
    batch_correct = ''
if dataSet == 'mouse_CTX_Hip_GEX':
    input_prep = input_prep_mouse_CTX_Hip_GEX
    batch_correct = ''
if dataSet == 'mouse_organogenesis_ATAC_fragment':
    input_prep = input_prep_mouse_organogenesis_ATAC
    batch_correct = ''
if dataSet == 'nano_ct_mouse_brain_H3K27ac_H3K27me3_bin10kb':
    input_prep = input_prep_nano_ct_mouse_brain_H3K27ac_H3K27me3_bin10kb
    batch_correct = '_batchCorrect'
if dataSet == 'nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_bin10kb':
    input_prep = input_prep_nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_bin10kb
    batch_correct = '_batchCorrect'
if dataSet == 'ntt_bmmc_H3K27ac_H3K27me3_bin10kb_topBin25k':
    input_prep = input_prep_ntt_bmmc_H3K27ac_H3K27me3_bin10kb_topBin25k
    batch_correct = ''


# Load datasets
if numModality == 1:
    omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict = input_prep()
else:
    omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict = input_prep(includeWNN, modalityList,numSample)

model_input = TensorDataset(Tensor(omics_data), 
                            batch_code_one_hot, 
                            Tensor(cell_idx).long())
print(f'Input tensor ready.')

# Save results
path = 'demo_results/'
model = 'FC'
if foldStratifyOption is None:
    foldStratifyBy = 'fold_stratify_by_NONE'
else:
    foldStratifyBy = 'fold_stratify_by_' + foldStratifyOption
if countDist == 'NegativeBinomial':
    countDistSave = 'nb'
else:
    countDistSave = 'poisson'
experiment = 'D' + str(dimLatent) + '_' +  \
             'B' + str(batchSize) + '_' +  \
             'E' + str(numEpochs) + '_' + \
             'H' + str(numHiddenUnits) + '_' + \
             'L' + str(numHiddenLayers) + '_' + \
             'DR' + str(dropoutRate) + '_' + \
             'LR' + str(learningRate) + '_' + countDistSave + batch_correct
path_to_dir = os.path.join(path,dataSet,model,foldStratifyBy,experiment)

if os.path.exists(path_to_dir):
    print("Directory already exists")
else:
    print("Directory doesn't exists. Create one")
    os.mkdir(path_to_dir)

# run fcVAE
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# For storing fold results
results = {}

# Set random number seed
seed_everything(randomSeed)
print(f'Random seed: {randomSeed}')
    
# K-fold Cross Validator
if foldStratifyOption is None:
    kfold = KFold(n_splits=numKFolds, shuffle=True, random_state=123)
    kfold_split = kfold.split(omics_data)
else:
    kfold = StratifiedKFold(n_splits=numKFolds, shuffle=True, random_state=123)
    kfold_split = kfold.split(omics_data, celltype_code_dict[foldStratifyOption])

# dicts for saving results
modality_list = modalityList
print(f'Modalities for analysis: {" ".join(modality_list)}')

# List of resolution for Louvain clustering
res_list = resList # test model on series of clustering resolutions [0.6,0.7,0.8,0.9,1.0]

ARI_train_cv_multiRes_dict = {annot_file: [[None for _ in range(numKFolds)] for _ in range(len(res_list))] 
                              for annot_file in annot_file_dict.keys()}
ARI_valid_cv_multiRes_dict = {annot_file: [[None for _ in range(numKFolds)] for _ in range(len(res_list))] 
                              for annot_file in annot_file_dict.keys()}

# loss cv
loss_cv = {}
loss_cv['train'] = [None for _ in range(numKFolds)]
loss_cv['valid'] = [None for _ in range(numKFolds)]
loss_epoch_train_cv = [None for _ in range(numKFolds)]

# runtime train cv
runtime_train_cv = [None for _ in range(numKFolds)]

# marginal llk cv
marginal_llk_valid_cv = [None for _ in range(numKFolds)]

# number of cluster cv
num_cluster_res_cv = {}
num_cluster_res_cv['train'] = [[None for _ in range(numKFolds)] for _ in range(len(res_list))]
num_cluster_res_cv['valid'] = [[None for _ in range(numKFolds)] for _ in range(len(res_list))]

# Cross-validation
print(f'Start running the model with {numKFolds}-Fold cross-validation')
for fold, (train_idx, valid_idx) in enumerate(kfold_split):
    print(f'------------------ Fold: {fold+1} ------------------')
    print('---------------------------------------------')
    # Track runtime
    start_time_fold = time.time()
    
    G = torch.Generator()
    G.manual_seed(fold)
    
    # Define sampler
    train_subsampler = SubsetRandomSampler(train_idx, generator=G)
    valid_subsampler = SubsetRandomSampler(valid_idx, generator=G)
    
    # Data loaders for traning and validation sets
    if dataSet == 'mouse_CTX_Hip_GEX':
        batchSize_eval = 2500
    else:
        batchSize_eval = batchSize
        
    params = {'batch_size': batchSize,
              'shuffle': False,
              'num_workers': 0,
              'drop_last': True}
    params_eval = {'batch_size': batchSize,
              'shuffle': False,
              'num_workers': 0,
              'drop_last': False}
    train_loader = DataLoader(model_input, sampler=train_subsampler, **params)
    train_eval_loader = DataLoader(model_input, sampler=train_subsampler, **params_eval)
    valid_eval_loader = DataLoader(model_input, sampler=valid_subsampler, **params_eval)

    # Initialize model
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vae_model = VAE(num_feature=numFeature,
                      count_dist=countDist,
                      num_modality=numModality,
                      num_sample=numSample,
                      dim_latent=dimLatent,
                      num_hidden_layers=numHiddenLayers,
                      num_hidden_units=numHiddenUnits,
                      beta=beta,
                      dropout_rate=dropoutRate).to(device)
    
    # save number of trainable params
    if fold == 0:
        num_param = count_model_parameters(vae_model) 
        np.save(os.path.join(path_to_dir, f'num_param.npy'), num_param)
        print(f'Total number of trainable params: {num_param}')
    
    # Reset model
    vae_model.apply(reset_model_weights)
    print('Model reset completed.')
    
    # Initialize optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=learningRate)
    
    # CVAE training
    train_loss_tracker = []
    
    print('Training starts...')
    vae_model.train()

    # Model Traning
    start_time_train_fold = time.time()
    for epoch in tqdm(range(1, numEpochs+1)):
        # if epoch % 100 == 0:
        #     print(f'Epoch: {epoch}')
        train_overall_loss = 0.0
        for batch, (cell_data, batch_id, cell_idx) in enumerate(train_loader):
            #print(cell_idx)
            cell_data = cell_data.to(device)
            batch_id = batch_id.to(device)
            optimizer.zero_grad() # zero the parameter gradients›
            inference_outputs, generative_outputs, loss = vae_model(cell_data, batch_id) 
            train_overall_loss += loss.item()
        
            loss.backward()
            optimizer.step()
    
        train_avg_loss = train_overall_loss / ((batch+1)*batchSize)
        train_loss_tracker.append(train_avg_loss)

    runtime_train_fold = time.time() - start_time_train_fold
    runtime_train_cv[fold] = runtime_train_fold
    loss_epoch_train_cv[fold] = train_loss_tracker
    print('Training completed.')
    print('Evaluation starts...')
    
    # Evaluation on validation set
    # Obtain the latent representation and reconstructed data of the cells
    latent_z_train = []
    latent_z_valid = []
    qz_x_var_train = []
    qz_x_var_valid = []
    cell_idx_train_eval_fold = []
    cell_idx_valid_eval_fold = []

    vae_model.eval()
    sample_z = False
     
    with torch.no_grad():
        overall_loss = 0.0
        marginal_llk = 0.0
        for data in valid_eval_loader: # validation set (loading without shuffling)
            cell_data, batch_id, cell_idx = data
            cell_idx_valid_eval_fold += cell_idx.tolist()
            cell_data = cell_data.to(device)
            batch_id = batch_id.to(device)

            inference_outputs, generative_outputs, loss = vae_model(cell_data, batch_id)
            overall_loss += loss.item()

            qz_x_mean = inference_outputs['qz_x_mean']
            qz_x_var = inference_outputs['qz_x_var']
            reconst_mean = get_reconst_data(generative_outputs)
                    
            if sample_z:
                samples = Normal(qz_x_mean, qz_x_var.sqrt()).sample([1000])
                z = nn.Softmax(dim=-1)(samples)
                z = z.mean(dim=0)
            else:
                z = qz_x_mean
            
            latent_z_valid += [z.cpu()]
            qz_x_var_valid += [qz_x_var.cpu()]
            
            # Importance sampling to estimate the marginal log likelihood
            num_samples = 100 # 100 importance samples
            for cell_i in range(cell_data.shape[0]): # iterate over all cells in the mini-batch
                qz_x_mean_i = qz_x_mean[cell_i]
                qz_x_std_i = qz_x_var[cell_i].sqrt()
                z_sample = Normal(qz_x_mean_i, qz_x_std_i).sample([num_samples])#.squeeze(0)
                if numModality >= 2:
                    library_size_factor_cell_i = inference_outputs['library_size_factor'][cell_i][None,:,:].repeat(num_samples,1,1)
                else:
                    library_size_factor_cell_i = inference_outputs['library_size_factor'][cell_i].repeat(num_samples,1)
                px_z_sample_, _ = vae_model.generative(z=z_sample, 
                                                         batch_id=batch_id[cell_i].repeat(num_samples,1), 
                                                         library_size_factor=library_size_factor_cell_i)
                log_qz_x = Normal(qz_x_mean_i, qz_x_std_i).log_prob(z_sample).sum(dim=-1)
                log_pz = Normal(torch.zeros_like(z_sample), torch.ones_like(z_sample)).log_prob(z_sample).sum(dim=1)
                log_weight = log_pz - log_qz_x
                for modality_idx in range(len(modality_list)):
                    log_weight += Poisson(px_z_sample_[modality_idx]['mean']).log_prob(cell_data[cell_i,modality_idx,:].repeat(num_samples,1)).sum(dim=-1)

                marginal_llk += torch.logsumexp(log_weight, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
    
    marginal_llk /= len(cell_idx_valid_eval_fold)
    marginal_llk_valid_cv[fold] = marginal_llk.item()
    print('Marginal log likelihood (Validation set): ', marginal_llk.item())
        
    avg_loss = overall_loss / len(cell_idx_valid_eval_fold)
    loss_cv['valid'][fold] = avg_loss
    print('Average Loss (Validation set): ', avg_loss)
    
    # Evaluation on training set
    with torch.no_grad():
        overall_loss = 0.0
        for data in train_eval_loader: # training set
            cell_data, batch_id, cell_idx = data
            cell_idx_train_eval_fold += cell_idx.tolist()
            cell_data = cell_data.to(device)
            batch_id = batch_id.to(device)
        
            inference_outputs, generative_outputs, loss = vae_model(cell_data, batch_id)
            overall_loss += loss.item()
        
            qz_x_mean = inference_outputs['qz_x_mean']
            qz_x_var = inference_outputs['qz_x_var']
            reconst_mean = get_reconst_data(generative_outputs)
                    
            if sample_z:
                samples = Normal(qz_x_mean, qz_x_var.sqrt()).sample([1000]) # sample 1000 values
                z = nn.Softmax(dim=-1)(samples)
                z = z.mean(dim=0)
            else:
                z = qz_x_mean 

            latent_z_train += [z.cpu()]
            qz_x_var_train += [qz_x_var.cpu()]
        
    avg_loss = overall_loss / len(cell_idx_train_eval_fold)
    loss_cv['train'][fold] = avg_loss
    print('Average Loss (Training set): ', avg_loss)

    cell_latent_representation_valid = np.array(torch.cat(latent_z_valid))
    print('Cell latent representation (Validation set) obtained.')
    cell_latent_representation_train = np.array(torch.cat(latent_z_train))
    print('Cell latent representation (Training set) obtained.')
    posterior_variance_valid = np.array(torch.cat(qz_x_var_valid))
    posterior_variance_train = np.array(torch.cat(qz_x_var_train))
    
    if dataSet == 'mouse_CTX_Hip_GEX': # if sample size too large, use subset of data for evaluation
        valid_subset_idx = random.sample(range(len(cell_idx_valid_eval_fold)), 1000)
        train_subset_idx = random.sample(range(len(cell_idx_train_eval_fold)), 1000)
        cell_idx_valid_eval_fold = [cell_idx_valid_eval_fold[idx] for idx in valid_subset_idx]
        cell_idx_train_eval_fold = [cell_idx_train_eval_fold[idx] for idx in train_subset_idx]
        num_start = 1 # louvain param
    else:
        valid_subset_idx = list(range(len(cell_idx_valid_eval_fold)))
        train_subset_idx = list(range(len(cell_idx_train_eval_fold)))
        num_start = 5 # louvain param
    
    # save
    np.save(os.path.join(path_to_dir, f'z_valid_fold_{fold}.npy'), cell_latent_representation_valid)
    np.save(os.path.join(path_to_dir, f'z_train_fold_{fold}.npy'), cell_latent_representation_train)
    np.save(os.path.join(path_to_dir, f'qz_x_var_valid_fold_{fold}.npy'), posterior_variance_valid)
    np.save(os.path.join(path_to_dir, f'qz_x_var_train_fold_{fold}.npy'), posterior_variance_train)
    np.save(os.path.join(path_to_dir, f'cell_idx_valid_eval_fold_{fold}.npy'), cell_idx_valid_eval_fold)
    np.save(os.path.join(path_to_dir, f'cell_idx_train_eval_fold_{fold}.npy'), cell_idx_train_eval_fold)
    np.save(os.path.join(path_to_dir, f'valid_subset_idx_fold_{fold}.npy'), valid_subset_idx)
    np.save(os.path.join(path_to_dir, f'train_subset_idx_fold_{fold}.npy'), train_subset_idx)

    # Clustering evaluation on train/validation sets
    for res_idx in range(len(res_list)):
        res = res_list[res_idx]
        print(f'Clustering resolution: {res}')
        cluster_train = louvain_cluster(cell_latent_representation_train[train_subset_idx,], resolution=res, k=20, n_starts=num_start)
        cluster_valid = louvain_cluster(cell_latent_representation_valid[valid_subset_idx,], resolution=res, k=20, n_starts=num_start)
        num_cluster_res_cv['train'][res_idx][fold] = len(np.unique(cluster_train))
        num_cluster_res_cv['valid'][res_idx][fold] = len(np.unique(cluster_valid))
    
        for annot_file in ARI_train_cv_multiRes_dict.keys():
            print(f'Annotation generated on: {annot_file}')
            ARI_train_cv_multiRes_dict[annot_file][res_idx][fold] = adjusted_rand_score(celltype_code_dict[annot_file][cell_idx_train_eval_fold], 
                                                                                        cluster_train)
            ARI_valid_cv_multiRes_dict[annot_file][res_idx][fold] = adjusted_rand_score(celltype_code_dict[annot_file][cell_idx_valid_eval_fold], 
                                                                                        cluster_valid)
    
    end_time_fold = time.time() - start_time_fold
    print('Validation completed.')
    print(f'Runtime is {end_time_fold} for this fold.')
    
    #if fold==0: break
    
print(f'{numKFolds}-fold cross-validation completed.')

## save results
np.save(os.path.join(path_to_dir, 'runtime_train_cv.npy'), runtime_train_cv)
np.save(os.path.join(path_to_dir, 'loss_cv.npy'), loss_cv)
np.save(os.path.join(path_to_dir, 'loss_epoch_train_cv.npy'), loss_epoch_train_cv)
np.save(os.path.join(path_to_dir, 'marginal_llk_valid_cv.npy'), marginal_llk_valid_cv)
np.save(os.path.join(path_to_dir, 'num_cluster_res_cv.npy'), num_cluster_res_cv)

## ARI
ARI_cv_report = report_AdjustedRandIndex(ARI_train_cv_multiRes_dict, ARI_valid_cv_multiRes_dict, res_list, 
                                         save_to_disk=True, save_path=path_to_dir)
print('ARI_report on CV:')
for annot_file in ARI_cv_report.keys():
    print(annot_file)
    print(ARI_cv_report[annot_file])