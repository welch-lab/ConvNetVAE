# line ending: unix

import numpy as np
from sklearn import metrics

# pytorch
import torch
import torch.nn.functional as F
from torch.distributions import Poisson


# Obtain shape of the output from a given layer (Toon Tran's post on stackoverflow)
def get_output_shape(model, input_dim, padding):
    return model(F.pad(torch.rand(*(input_dim)), (padding,0))).data.shape 


# Get the number of trainable parameters of a model
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Reset model weights during cross-validation
def reset_model_weights(model):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# Obtain the rate parameter for p(x|z) (Poisson distribution)
def get_reconst_data(generative_outputs):
    # frag_count_x = Poisson(generative_outputs['px_']['poisson_mean']).sample()
    # frag_count_y = Poisson(generative_outputs['py_']['poisson_mean']).sample()
    # frag_count_z = Poisson(generative_outputs['pz_']['poisson_mean']).sample()

    # frag_count = [frag_count_x.cpu(),frag_count_y.cpu(),frag_count_z.cpu()]
    px_z_mean = [generative_outputs[i]['mean'].cpu() for i in range(len(generative_outputs))]    

    # px_z_mean = [generative_outputs['px_']['poisson_mean'].cpu(),
    #              generative_outputs['py_']['poisson_mean'].cpu(),
    #              generative_outputs['pz_']['poisson_mean'].cpu()]

    return px_z_mean


def save_reconst_data(reconst_data_dict, data_split, modality_list, reconst_data):
    for mod in modality_list:
        mod_idx = modality_list.index(mod)
        # reconst_data_dict[mod][data_split]["count"] += [reconst_count[0]]
        reconst_data_dict[mod][data_split]['mean'] += [reconst_data[mod_idx]]


def aggregate_reconst_data(reconst_data_dict, data_split, modality_list):
    for mod in modality_list:
        # reconst_result[mod][data_split]["count"] = np.vstack(reconst_result[mod][data_split]["count"])
        reconst_data_dict[mod][data_split]['mean'] = np.vstack(reconst_data_dict[mod][data_split]['mean'])

          
def reconst_eval(dataset, reconst_data, rmse_result, corr_result, data_split, cell_idx_list, modality_list, subset_idx=None):
    for mod in modality_list: # loop over all modalities
        mod_idx = modality_list.index(mod)

        orig_mod_data = dataset[cell_idx_list, mod_idx, :]

        if len(subset_idx) < reconst_data[mod][data_split]['mean'].shape[0]: # sampled data for eval for efficiency
            print('Evaluation on subset of data.')
            reconst_data_mod = reconst_data[mod][data_split]['mean'][subset_idx,]
        else: 
            reconst_data_mod = reconst_data[mod][data_split]['mean']
        # RMSE
        # rmse_cv[mod][data_split]["count"].append(metrics.mean_squared_error(orig_mod_data, 
        #                                                                     reconst_result[mod][data_split]["count"], 
        #                                                                     squared=False))
        rmse_result[mod][data_split]['mean'].append(metrics.mean_squared_error(orig_mod_data, reconst_data_mod, squared=False))
        # correlation
        orig_mod_data = orig_mod_data.flatten()
        # count_rho = np.corrcoef(orig_mod_data, reconst_result[mod][data_split]["count"].flatten())
        mean_rho = np.corrcoef(orig_mod_data, reconst_data_mod.flatten())
        # rho_cv[mod][data_split]["count"].append(count_rho[0,1])
        corr_result[mod][data_split]['mean'].append(mean_rho[0,1])