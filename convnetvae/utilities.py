# line ending: unix

import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch import Tensor

import louvain
from .louvain_utils import run_knn, compute_snn, build_igraph


def nullable_string(val): 
    if not val:
        return None
    return val

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# report loss from cross-validation results
def report_loss(loss_result, num_kfolds, save_to_disk=False, save_path=None):
    # create loss report
    loss_df = pd.DataFrame(list(zip(loss_result['train'],
                                    loss_result['valid'])),
                              columns =['lossTrainingSet',
                                        'lossValidationSet'])
    loss_df_summary = loss_df.describe()
    if save_to_disk and save_path is not None:
        loss_df_summary.to_pickle(os.path.join(save_path, f'loss_report.pkl'))
        print('File saved to disk.')
    return loss_df_summary


# report ARI metrics from cross-validation results
def report_AdjustedRandIndex(ARI_train_dict, ARI_valid_dict, resolution_list, save_to_disk=False, save_path=None):
    print(f'Generate report on ARI...')
    ARI_df_dict = {annot_file: None for annot_file in ARI_train_dict.keys()}

    for annot_file in ARI_train_dict.keys():
        avg_cv_ARI_train_res = [np.array(ARIs).mean() for ARIs in ARI_train_dict[annot_file]]
        avg_cv_ARI_valid_res = [np.array(ARIs).mean() for ARIs in ARI_valid_dict[annot_file]]
        std_cv_ARI_train_res = [np.array(ARIs).std() for ARIs in ARI_train_dict[annot_file]]
        std_cv_ARI_valid_res = [np.array(ARIs).std() for ARIs in ARI_valid_dict[annot_file]]
    
    # create ARI report
        ARI_df_dict[annot_file] = pd.DataFrame(list(zip(resolution_list, 
                                                        avg_cv_ARI_train_res,
                                                        std_cv_ARI_train_res,
                                                        avg_cv_ARI_valid_res,
                                                        std_cv_ARI_valid_res)),
                                               columns =['clusterResolution', 
                                                         'ARITrainingSetMean',
                                                         'ARITrainingSetStd',
                                                         'ARIValidationSetMean',
                                                         'ARIValidationSetStd'])
    if save_to_disk and save_path is not None:
        for annot_file in ARI_df_dict.keys():
            ARI_df_dict[annot_file].to_pickle(os.path.join(save_path, f'ARI_report_{annot_file}.pkl'))
        print('Files saved to disk.')
    return ARI_df_dict


# report evaluation metrics from cross-validation results
def report_evalMetrics(eval_result, modality_list, metric_name=None, save_to_disk=False, save_path=None):
    if metric_name is not None:
        print(f'Generate report on {metric_name}...')
    avg_cv_train = []
    std_cv_train = []
    avg_cv_valid = []
    std_cv_valid = []
    
    for modality in modality_list:
        avg_cv_train.append(np.mean(eval_result[modality]['train']['mean']))
        std_cv_train.append(np.std(eval_result[modality]['train']['mean']))
        avg_cv_valid.append(np.mean(eval_result[modality]['valid']['mean']))
        std_cv_valid.append(np.std(eval_result[modality]['valid']['mean']))
    
    # create ARI report
    metrics_df = pd.DataFrame(list(zip(modality_list,
                                       avg_cv_train,
                                       std_cv_train,
                                       avg_cv_valid,
                                       std_cv_valid)),
                              columns =['modality',
                                        metric_name+'TrainingSetMean',
                                        metric_name+'TrainingSetStd',
                                        metric_name+'ValidationSetMean',
                                        metric_name+'ValidationSetStd'])
    if save_to_disk and save_path is not None:
        metrics_df.to_pickle(os.path.join(save_path, f'{metric_name}_report.pkl'))
        print('File saved to disk.')
    return metrics_df


# louvain clustering (derived from pyliger)
def louvain_cluster(x,
                    resolution=1.0,
                    k=20,
                    prune=1 / 15,
                    random_seed=1,
                    n_starts=1,
                    verbose=True):
    """Louvain algorithm for community detection

    After quantile normalization, users can additionally run the Louvain algorithm
    for community detection, which is widely used in single-cell analysis and excels at merging
    small clusters
    nto broad cell classes.
    Parameters
    ----------
    liger_object : liger object
        Should run quantile_norm before calling.
    resolution : float, optional
        Value of the resolution parameter, use a value above (below) 1.0 if you want
        to obtain a larger (smaller) number of communities (the default is 1.0).
    k : int, optional
        The maximum number of nearest neighbours to compute (the default is 20).
    prune : float, optional
        Sets the cutoff for acceptable Jaccard index when
        computing the neighborhood overlap for the SNN construction. Any edges with
        values less than or equal to this will be set to 0 and removed from the SNN
        graph. Essentially sets the strigency of pruning (0 --- no pruning, 1 ---
        prune everything) (the default is 1/15).
    random_seed : int, optional
        Seed of the random number generator (the default is 1).
    n_starts : int, optional
        The number of random starts to be used (the default is 10).
    Returns
    -------
    liger_object : liger object
        object with refined 'cluster'.

    Examples
    --------
    >>> ligerex = louvain_cluster(ligerex, resulotion = 0.3) # liger object, factorization complete
    """
    ### 1. Compute snn
    H_norm = x
    knn = run_knn(H_norm, k)
    snn = compute_snn(knn, prune=prune)

    ### 2. Get igraph from snn
    g = build_igraph(snn)

    ### 3. Run louvain
    np.random.seed(random_seed)
    max_quality = -1
    for i in range(n_starts):  # random starts to improve stability
        seed = np.random.randint(0, 1000)
        kwargs = {'weights': g.es['weight'], 'resolution_parameter': resolution, 'seed': seed}  # parameters setting
        part = louvain.find_partition(g, louvain.RBConfigurationVertexPartition, **kwargs)
        if verbose:
            print("Number of clusters (Louvain): ", len(part))

        if part.quality() > max_quality:
            cluster = part.membership
            max_quality = part.quality()

    ### 4. Assign cluster results
    #cluster_assignment = _assign_cluster(H_norm, cluster)

    return cluster


def input_prep_pbmc_GEX():  
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'pbmc'
    data_file = 'pbmc_10x_multiome_top_gex_data.npy'
    annot_file_dict = {'GEX':'pbmc_10x_multiome_celltype_seurat_GEX_5k_Peak_25k_louvain_res03.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                                for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path)
    omics_data = omics_data[:, None, :] # cell x feature --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_code_one_hot = Tensor(np.array([0]*omics_data.shape[0])).long().unsqueeze(-1)

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx

    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict


def input_prep_pbmc_ATAC():  
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'pbmc'
    data_file = 'pbmc_10x_multiome_fragment_top_peaks_data.npy'
    annot_file_dict = {'ATAC':'pbmc_10x_multiome_celltype_seurat_GEX_5k_Peak_25k_louvain_res03.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                                for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path) # cell x feature
    omics_data = omics_data[:, None, :] # cell x feature --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_code_one_hot = Tensor(np.array([0]*omics_data.shape[0])).long().unsqueeze(-1)

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx

    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict


def input_prep_mouse_CTX_Hip_GEX():  
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'mouse_cortex_hippocampus_RNA_zeng'
    data_file = 'CTX_Hip_RNA_zeng_vargenes_5k.npy'
    annot_file_dict = {'GEX':'CTX_Hip_anno_10x.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                                for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path) # cell x feature
    omics_data = omics_data[:, None, :] # cell x feature --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_code_one_hot = Tensor(np.array([0]*omics_data.shape[0])).long().unsqueeze(-1)

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx

    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict

def input_prep_mouse_organogenesis_ATAC():  
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'mouse_organogenesis_ATAC_reik'
    data_file = 'ATAC/mouse_organogenesis_ATAC_reik_fragment_toppeaks_25k.npy'
    annot_file_dict = {'ATAC':'ATAC/mouse_organogenesis_reik_ann.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                                for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path) # cell x feature
    omics_data = omics_data[:, None, :] # cell x feature --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_code_one_hot = Tensor(np.array([0]*omics_data.shape[0])).long().unsqueeze(-1)

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx

    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict

def input_prep_nano_ct_mouse_brain_H3K27ac_H3K27me3_bin10kb(includeWNN,modalityList,numSample):   
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'nano_ct_mouse_brain'
    data_file = 'nano_ct_mouse_brain_H3K27ac_H3K27me3_topBin25k_binSize10kb.npy'
    batch_info_file = 'nano_ct_mouse_brain_H3K27ac_H3K27me3_batch_info.csv'
    if includeWNN:
        print('Including WNN labels for evaluation.')
        annot_file_dict = {modality:'nano_ct_mouse_brain_2mod_' + modality + '_cell_type_l3.csv' 
                           for modality in (modalityList + ['WNN'])}
    else:
        print('Not including WNN labels for evaluation.')
        annot_file_dict = {modality:'nano_ct_mouse_brain_2mod_' + modality + '_cell_type_l3.csv' 
                           for modality in modalityList}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    batch_info_file_path = os.path.join(data_directory, data_collection, batch_info_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                            for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path)
    omics_data = np.swapaxes(omics_data, 1, 2) # cell x feature x modality --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')
    #print(multimodal_data.shape)

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_id_info = pd.read_csv(batch_info_file_path, sep='\t')
    batch_id_info.batch = pd.Categorical(batch_id_info.batch)
    batch_id_info['batch_code'] = batch_id_info.batch.cat.codes #batch_id_info  
    # Class values must be smaller than num_classes.
    batch_code = torch.tensor(batch_id_info.batch_code, dtype=torch.int64)  #,dtype=torch.float32 #batch_code
    batch_code_one_hot = F.one_hot(batch_code, num_classes=numSample) #batch_code_one_hot

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx
    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict

def input_prep_nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_bin10kb(includeWNN,modalityList,numSample):   
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'nano_ct_mouse_brain'
    data_file = 'nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_topBin25k_binSize10kb.npy'
    batch_info_file = 'nano_ct_mouse_brain_ATAC_H3K27ac_H3K27me3_batch_info.csv'

    print('Including WNN labels for evaluation.')
    annot_file_dict = {'WNN':'nano_ct_mouse_brain_WNN_cell_type_l3.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    batch_info_file_path = os.path.join(data_directory, data_collection, batch_info_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                            for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path)
    omics_data = np.swapaxes(omics_data, 1, 2) # cell x feature x modality --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')
    #print(multimodal_data.shape)

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_id_info = pd.read_csv(batch_info_file_path, sep='\t')
    batch_id_info.batch = pd.Categorical(batch_id_info.batch)
    batch_id_info['batch_code'] = batch_id_info.batch.cat.codes #batch_id_info  

    batch_code = torch.tensor(batch_id_info.batch_code, dtype=torch.int64)#,dtype=torch.float32 #batch_code
    batch_code_one_hot = F.one_hot(batch_code, num_classes=numSample) #batch_code_one_hot

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx
    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict

def input_prep_ntt_bmmc_H3K27ac_H3K27me3_bin10kb_topBin25k(includeWNN,modalityList,numSample):   
    # Load datasets
    # File path
    data_directory = 'datasets/'
    data_collection = 'ntt_seq_bmmc'
    data_file = 'ntt_bmmc_H3K27ac_H3K27me3_topBin25k_binSize_10kb.npy'
    annot_file_dict = {'WNN':'ntt_bmmc_WNN_cell_type.csv'}

    omics_data_file_path = os.path.join(data_directory, data_collection, data_file)
    annot_file_path_list = [os.path.join(data_directory, data_collection, annot_file) 
                            for annot_file in annot_file_dict.values()]

    ## Load input data (if 3D tensor)
    omics_data = np.load(omics_data_file_path)
    omics_data = np.swapaxes(omics_data, 1, 2) # cell x feature x modality --> cell x modality (channel) x feature
    print(f'Input data loading completed. Dimension: {omics_data.shape}')
    #print(multimodal_data.shape)

    celltype_code_dict = {modality: None for modality in list(annot_file_dict.keys())} #celltype_code
    for modality_idx, modality in enumerate(celltype_code_dict.keys()):
        celltype_pd = pd.read_csv(annot_file_path_list[modality_idx], sep='\t')
        celltype_pd.ann = pd.Categorical(celltype_pd.ann)
        celltype_pd['ann_code'] = celltype_pd.ann.cat.codes
        celltype_code_dict[modality] = celltype_pd.ann_code.to_numpy()

    batch_code_one_hot = Tensor(np.array([0]*omics_data.shape[0])).long().unsqueeze(-1)

    cell_idx = np.array([i for i in range(omics_data.shape[0])]) #cell_idx
    return omics_data, batch_code_one_hot, cell_idx, annot_file_dict, celltype_code_dict
