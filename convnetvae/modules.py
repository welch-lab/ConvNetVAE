# line ending: unix

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import math

from convNetVAE_utils import *


class EncoderConvNetL1(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self,
        num_feature: int, 
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,      
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_padding: list = [0],
        size_stride: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        # calculate padding for conv1D layers
        #self.convEnc_1_padding = size_conv_kernel - num_feature % size_conv_kernel
        self.convEnc_1_padding = (int(math.ceil(num_feature/size_stride)-1))*size_stride-num_feature+size_conv_kernel
        #print(self.convEnc_1_padding)
        
        # define conv1D layers
        self.convEnc_1 = nn.Sequential(            
            nn.Conv1d(num_modality, num_channel[0], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[0]), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Dimension of input for the Linear layer
        convEnc_1_out = get_output_shape(self.convEnc_1, (1, num_modality, num_feature),
                                         padding=self.convEnc_1_padding)
        self.convEnc_1_out = convEnc_1_out[-1]
        self.fcEnc_in = np.prod(list(convEnc_1_out)) # Flatten
        #print(self.fcEnc_in)
        
        if num_sample > 1:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in + num_sample, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )

        self.z_mean_enc = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent),
        )

        self.z_logvar_enc = nn.Linear(num_hidden_units, dim_latent)

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        #library_size_factor = self.calculate_library_size_factor(data)
        library_size_factor = self.calculate_multimodal_size_factor(data,self.num_modality)

        data = torch.log(data + 1)
        q = self.convEnc_1(F.pad(data, (self.convEnc_1_padding,0)))
        q = q.view(q.shape[0],-1)
        if batch_id.size()[1] > 1:
            q_cat = torch.cat((q, batch_id), dim=-1)
        else:
            q_cat = q
        q = self.fcEnc(q_cat)
        
        qz_x_mean = self.z_mean_enc(q)
        qz_x_var = torch.exp(self.z_logvar_enc(q))
        z = Normal(qz_x_mean, qz_x_var.sqrt()).rsample()

        return qz_x_mean, qz_x_var, z, library_size_factor


class EncoderConvNetL1Alt(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters

    concate batch info with bin data before 1st layer
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self,
        num_feature: int, 
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,      
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_padding: list = [0],
        size_stride: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        # calculate padding for conv1D layers
        #self.convEnc_1_padding = size_conv_kernel - num_feature % size_conv_kernel
        self.convEnc_1_padding = (int(math.ceil((num_feature+num_sample)/size_stride)-1))*size_stride-num_feature+size_conv_kernel
        #print(self.convEnc_1_padding)
        
        # define conv1D layers
        self.convEnc_1 = nn.Sequential(            
            nn.Conv1d(num_modality, num_channel[0], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[0]), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Dimension of input for the Linear layer
        convEnc_1_out = get_output_shape(self.convEnc_1, (1, num_modality, num_feature),
                                         padding=self.convEnc_1_padding)
        self.convEnc_1_out = convEnc_1_out[-1]
        self.fcEnc_in = np.prod(list(convEnc_1_out)) # Flatten
        #print(self.fcEnc_in)

        self.fcEnc = nn.Sequential(
            nn.Linear(self.fcEnc_in, num_hidden_units),
            nn.BatchNorm1d(num_hidden_units), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        self.z_mean_enc = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent),
        )

        self.z_logvar_enc = nn.Linear(num_hidden_units, dim_latent)

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        #library_size_factor = self.calculate_library_size_factor(data)
        library_size_factor = self.calculate_multimodal_size_factor(data,self.num_modality)
        
        data = torch.log(data + 1)
        if batch_id.size()[1] > 1:
            q = self.convEnc_1(F.pad(torch.cat((data, batch_id.unsqueeze(axis=1).repeat(1,self.num_modality,1)), dim=-1), (self.convEnc_1_padding,0)))
        else:
            q = self.convEnc_1(F.pad(data, (self.convEnc_1_padding,0)))
        q = q.view(q.shape[0],-1)

        q = self.fcEnc(q)
        
        qz_x_mean = self.z_mean_enc(q)
        qz_x_var = torch.exp(self.z_logvar_enc(q))
        z = Normal(qz_x_mean, qz_x_var.sqrt()).rsample()

        return qz_x_mean, qz_x_var, z, library_size_factor


class EncoderConvNetL2(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self,
        num_feature: int, 
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,      
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_padding: list = [0],
        size_stride: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        # calculate padding for conv1D layers
        self.convEnc_1_padding = (int(math.ceil(num_feature/size_stride)-1))*size_stride-num_feature+size_conv_kernel
        
        # define conv1D layers
        self.convEnc_1 = nn.Sequential(            
            nn.Conv1d(num_modality, num_channel[0], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[0]), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        
        self.convEnc_2 = nn.Sequential(            
            nn.Conv1d(num_channel[0], num_channel[1], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[1]), #, eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #nn.MaxPool1d(2, stride=2, padding=0)
        )

        # Dimension of input for the Linear layer
        convEnc_1_out = get_output_shape(self.convEnc_1, (1, num_modality, num_feature),
                                         padding=self.convEnc_1_padding)
        self.convEnc_2_padding = (int(math.ceil(convEnc_1_out[-1]/size_stride)-1))*size_stride-convEnc_1_out[-1]+size_conv_kernel
        convEnc_2_out = get_output_shape(self.convEnc_2, convEnc_1_out, padding=self.convEnc_2_padding)
        self.fcEnc_in = np.prod(list(convEnc_2_out)) # Flatten
        self.convEnc_1_out = convEnc_1_out[-1] # feature dim
        self.convEnc_2_out = convEnc_2_out[-1]
        
        if num_sample > 1:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in + num_sample, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )

        self.z_mean_enc = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_enc = nn.Linear(num_hidden_units, dim_latent)

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        #library_size_factor = self.calculate_library_size_factor(data)
        library_size_factor = self.calculate_multimodal_size_factor(data,self.num_modality)
        
        data = torch.log(data + 1)
        q = self.convEnc_1(F.pad(data, (self.convEnc_1_padding,0)))
        q = self.convEnc_2(F.pad(q, (self.convEnc_2_padding,0)))
        q = q.view(q.shape[0],-1)
        if batch_id.size()[1] > 1:
            q_cat = torch.cat((q, batch_id), dim=-1)
        else:
            q_cat = q
        q = self.fcEnc(q_cat)
        
        qz_x_mean = self.z_mean_enc(q)
        qz_x_var = torch.exp(self.z_logvar_enc(q))
        z = Normal(qz_x_mean, qz_x_var.sqrt()).rsample()

        return qz_x_mean, qz_x_var, z, library_size_factor


class EncoderConvNetL3(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self,
        num_feature: int, 
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,      
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_padding: list = [0],
        size_stride: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        # calculate padding for conv1D layers
        self.convEnc_1_padding = (int(math.ceil(num_feature/size_stride)-1))*size_stride-num_feature+size_conv_kernel
        
        # define conv1D layers
        self.convEnc_1 = nn.Sequential(            
            nn.Conv1d(num_modality, num_channel[0], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[0]), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        
        self.convEnc_2 = nn.Sequential(            
            nn.Conv1d(num_channel[0], num_channel[1], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[1]), #, eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #nn.MaxPool1d(2, stride=2, padding=0)
        )
        
        self.convEnc_3 = nn.Sequential(            
            nn.Conv1d(num_channel[1], num_channel[2], kernel_size=(size_conv_kernel,), stride=size_stride, padding=0),
            nn.BatchNorm1d(num_channel[2]), #, eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            #nn.MaxPool1d(2, stride=2, padding=0)
        )

        # Dimension of input for the Linear layer
        convEnc_1_out = get_output_shape(self.convEnc_1, (1, num_modality, num_feature),
                                         padding=self.convEnc_1_padding)
        self.convEnc_2_padding = (int(math.ceil(convEnc_1_out[-1]/size_stride)-1))*size_stride-convEnc_1_out[-1]+size_conv_kernel
        convEnc_2_out = get_output_shape(self.convEnc_2, convEnc_1_out, padding=self.convEnc_2_padding)
        self.convEnc_3_padding = (int(math.ceil(convEnc_2_out[-1]/size_stride)-1))*size_stride-convEnc_2_out[-1]+size_conv_kernel
        convEnc_3_out = get_output_shape(self.convEnc_3, convEnc_2_out, padding=self.convEnc_3_padding)
        self.fcEnc_in = np.prod(list(convEnc_3_out)) # Flatten
        
        self.convEnc_1_out = convEnc_1_out[-1] # feature dim
        self.convEnc_2_out = convEnc_2_out[-1]
        self.convEnc_3_out = convEnc_3_out[-1]
        
        if num_sample > 1:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in + num_sample, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcEnc = nn.Sequential(
                nn.Linear(self.fcEnc_in, num_hidden_units),
                nn.BatchNorm1d(num_hidden_units), 
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            )

        self.z_mean_enc = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent),
        )

        self.z_logvar_enc = nn.Linear(num_hidden_units, dim_latent)

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        #library_size_factor = self.calculate_library_size_factor(data)
        library_size_factor = self.calculate_multimodal_size_factor(data,self.num_modality)
        
        data = torch.log(data + 1)
        q = self.convEnc_1(F.pad(data, (self.convEnc_1_padding,0)))
        q = self.convEnc_2(F.pad(q, (self.convEnc_2_padding,0)))
        q = self.convEnc_3(F.pad(q, (self.convEnc_3_padding,0)))
        q = q.view(q.shape[0],-1)
        if batch_id.size()[1] > 1:
            q_cat = torch.cat((q, batch_id), dim=-1)
        else:
            q_cat = q
        q = self.fcEnc(q_cat)
        
        qz_x_mean = self.z_mean_enc(q)
        qz_x_var = torch.exp(self.z_logvar_enc(q))
        z = Normal(qz_x_mean, qz_x_var.sqrt()).rsample()

        return qz_x_mean, qz_x_var, z, library_size_factor


class DecoderConvNetL1(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        convEnc_1_out: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3,       
        size_stride: int = 1,
        size_padding: list = [0],
        size_padding_extra: int = 0,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        self.num_channel = num_channel
        self.stride = size_stride
        self.fcDec_out = convEnc_1_out * num_channel[-1]
        self.convDec_1_in = convEnc_1_out * size_stride
        self.convDec_1_padding = int(num_feature - self.convDec_1_in + size_conv_kernel - 1)
        
        if num_sample > 1:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent + num_sample, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        
        self.convDec_1 = nn.Sequential(            
            nn.Conv1d(num_channel[0], num_modality, kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.Softmax(dim=-1)
        )
        
        self.dec_upsample_1 = nn.Upsample(scale_factor=size_stride)

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        if batch_id.size()[1] > 1:
            px_z = self.fcDec(torch.cat((z, batch_id), dim=-1))
        else:
            px_z = self.fcDec(z)
        px_z = px_z.view(px_z.shape[0], self.num_channel[-1], -1)
        px_z = self.dec_upsample_1(px_z)
        px_z = self.convDec_1(F.pad(px_z,(self.convDec_1_padding,0)))

        for modality_idx in range(self.num_modality):
            px_z_[modality_idx]['mean'] = px_z[:,modality_idx,:] * library_size_factor[:,modality_idx,:]

        return px_z_


class DecoderConvNetL2(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        convEnc_1_out: int,
        convEnc_2_out: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3,       
        size_stride: int = 1,
        size_padding: list = [0],
        size_padding_extra: int = 0,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        self.num_channel = num_channel
        self.stride = size_stride
        self.fcDec_out = convEnc_2_out * num_channel[-1]
        self.convDec_1_in = convEnc_2_out * size_stride
        self.convDec_1_padding = int(convEnc_1_out -1 - self.convDec_1_in + size_conv_kernel)
        self.convDec_2_in = convEnc_1_out * size_stride
        self.convDec_2_padding = int(num_feature - 1 - self.convDec_2_in + size_conv_kernel)
        
        if num_sample > 1:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent + num_sample, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        
        self.convDec_1 = nn.Sequential(            
            nn.Conv1d(num_channel[1], num_channel[0], kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.BatchNorm1d(num_channel[0]),# eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.convDec_2 = nn.Sequential(            
            nn.Conv1d(num_channel[0], num_modality, kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.Softmax(dim=-1)
        )
        
        self.dec_upsample_1 = nn.Upsample(scale_factor=size_stride)
        self.dec_upsample_2 = nn.Upsample(scale_factor=size_stride)

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        if batch_id.size()[1] > 1:
            px_z = self.fcDec(torch.cat((z, batch_id), dim=-1))
        else:
            px_z = self.fcDec(z)
        px_z = px_z.view(px_z.shape[0], self.num_channel[-1], -1)
        px_z = self.dec_upsample_1(px_z)
        px_z = self.convDec_1(F.pad(px_z,(self.convDec_1_padding,0)))
        px_z = self.dec_upsample_2(px_z)
        px_z = self.convDec_2(F.pad(px_z,(self.convDec_2_padding,0)))

        for modality_idx in range(self.num_modality):
            px_z_[modality_idx]['mean'] = px_z[:,modality_idx,:] * library_size_factor[:,modality_idx,:]

        return px_z_


class DecoderConvNetL3(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        convEnc_1_out: int,
        convEnc_2_out: int,
        convEnc_3_out: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3,       
        size_stride: int = 1,
        size_padding: list = [0],
        size_padding_extra: int = 0,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        self.num_channel = num_channel
        self.stride = size_stride
        self.fcDec_out = convEnc_3_out * num_channel[-1]
        self.convDec_1_in = convEnc_3_out * size_stride
        self.convDec_1_padding = int(convEnc_2_out -1 - self.convDec_1_in + size_conv_kernel)
        self.convDec_2_in = convEnc_2_out * size_stride
        self.convDec_2_padding = int(convEnc_1_out -1 - self.convDec_2_in + size_conv_kernel)
        self.convDec_3_in = convEnc_1_out * size_stride
        self.convDec_3_padding = int(num_feature - 1 - self.convDec_3_in + size_conv_kernel)
        
        if num_sample > 1:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent + num_sample, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        else:
            self.fcDec = nn.Sequential(
                nn.Linear(dim_latent, self.fcDec_out),
    #             nn.BatchNorm1d(fcDec_in),
    #             nn.ReLU(),
    #             nn.Dropout(p=dropout_rate)
            )
        
        self.convDec_1 = nn.Sequential(            
            nn.Conv1d(num_channel[2], num_channel[1], kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.BatchNorm1d(num_channel[1]),# eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.convDec_2 = nn.Sequential(            
            nn.Conv1d(num_channel[1], num_channel[0], kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.BatchNorm1d(num_channel[0]),# eps=1e-05, momentum=0.1, affine=True
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.convDec_3 = nn.Sequential(            
            nn.Conv1d(num_channel[0], num_modality, kernel_size=(size_conv_kernel,), stride=1, padding=0), #-1
            nn.Softmax(dim=-1)
        )
        
        self.dec_upsample_1 = nn.Upsample(scale_factor=size_stride)
        self.dec_upsample_2 = nn.Upsample(scale_factor=size_stride)
        self.dec_upsample_3 = nn.Upsample(scale_factor=size_stride)

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        if batch_id.size()[1] > 1:
            px_z = self.fcDec(torch.cat((z, batch_id), dim=-1))
        else:
            px_z = self.fcDec(z)
        px_z = px_z.view(px_z.shape[0], self.num_channel[-1], -1)
        px_z = self.dec_upsample_1(px_z)
        px_z = self.convDec_1(F.pad(px_z,(self.convDec_1_padding,0)))
        px_z = self.dec_upsample_2(px_z)
        px_z = self.convDec_2(F.pad(px_z,(self.convDec_2_padding,0)))
        px_z = self.dec_upsample_3(px_z)
        px_z = self.convDec_3(F.pad(px_z,(self.convDec_3_padding,0)))

        for modality_idx in range(self.num_modality):
            px_z_[modality_idx]['mean'] = px_z[:,modality_idx,:] * library_size_factor[:,modality_idx,:]

        return px_z_


class EncoderPOE(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self, 
        num_feature: int,      
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.num_feature = num_feature
        
        modules_x = []
        modules_x.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_x.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_x = nn.Sequential(*modules_x)
        
        modules_y = []
        modules_y.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_y.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_y = nn.Sequential(*modules_y)
        
        modules_z = []
        modules_z.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_z.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_z = nn.Sequential(*modules_z)

        self.z_mean_encoder_x = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )
        
        self.z_logvar_encoder_x = nn.Linear(num_hidden_units, dim_latent) 
        
        self.z_mean_encoder_y = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_encoder_y = nn.Linear(num_hidden_units, dim_latent)
            
        self.z_mean_encoder_z = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_encoder_z = nn.Linear(num_hidden_units, dim_latent)
        
    def product_of_experts(self, mus, logvars):
        dist_vars = torch.exp(logvars)
        mus_joint = torch.sum(mus / dist_vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)  # batch size
        logvars_joint += torch.sum(torch.ones_like(mus) / dist_vars, dim=1)
        logvars_joint = 1.0 / logvars_joint  # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    # calculate library size
    def cal_library_size_factor(self, data):
        total_count_1 = torch.sum(data[:, 0, :], axis=1).unsqueeze(1)
        total_count_1 = total_count_1.repeat(1, data.shape[2])
        
        total_count_2 = torch.sum(data[:, 1, :], axis=1).unsqueeze(1)
        total_count_2 = total_count_2.repeat(1, data.shape[2])
        
        total_count_3 = torch.sum(data[:, 2, :], axis=1).unsqueeze(1)
        total_count_3 = total_count_3.repeat(1, data.shape[2])
        return torch.cat((total_count_1[:,None,:], total_count_2[:,None,:], total_count_3[:,None,:]), dim=1)
        
    def forward(self, data, batch_id):
        library_size_factor = self.cal_library_size_factor(data)

        data = torch.log(data + 1)
        batch_id_tensor = batch_id

        q_x = self.encoder_x(torch.cat((data[:,0,:], batch_id_tensor), dim=-1))
        qz_mean_x = self.z_mean_encoder_x(q_x)
        qz_logvar_x = self.z_logvar_encoder_x(q_x)

        q_y = self.encoder_y(torch.cat((data[:,1,:], batch_id_tensor), dim=-1))
        qz_mean_y = self.z_mean_encoder_y(q_y)
        qz_logvar_y = self.z_logvar_encoder_y(q_y)
            
        q_z = self.encoder_y(torch.cat((data[:,2,:], batch_id_tensor), dim=-1))
        qz_mean_z = self.z_mean_encoder_z(q_z)
        qz_logvar_z = self.z_logvar_encoder_z(q_z)
        
        mu = torch.stack([qz_mean_x, qz_mean_y, qz_mean_z], dim=1)
        logvar = torch.stack([qz_logvar_x, qz_logvar_y, qz_logvar_z], dim=1)
        mu_joint, logvar_joint = self.product_of_experts(mu, logvar)
        z_poe = Normal(mu_joint, torch.exp(logvar_joint).sqrt()).rsample()

        return mu_joint, torch.exp(logvar_joint), z_poe, library_size_factor 

class EncoderExp2POE(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self, 
        num_feature: int,      
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.num_modality = num_modality
        
        modules_x = []
        modules_x.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_x.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_x = nn.Sequential(*modules_x)
        
        modules_y = []
        modules_y.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_y.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_y = nn.Sequential(*modules_y)
        

        self.z_mean_encoder_x = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )
        
        self.z_logvar_encoder_x = nn.Linear(num_hidden_units, dim_latent) 
        
        self.z_mean_encoder_y = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_encoder_y = nn.Linear(num_hidden_units, dim_latent)
        
    def product_of_experts(self, mus, logvars):
        dist_vars = torch.exp(logvars)
        mus_joint = torch.sum(mus / dist_vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)  # batch size
        logvars_joint += torch.sum(torch.ones_like(mus) / dist_vars, dim=1)
        logvars_joint = 1.0 / logvars_joint  # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        library_size_factor = self.calculate_multimodal_size_factor(data, self.num_modality)

        data = torch.log(data + 1)
        batch_id_tensor = batch_id

        q_x = self.encoder_x(torch.cat((data[:,0,:], batch_id_tensor), dim=-1))
        qz_mean_x = self.z_mean_encoder_x(q_x)
        qz_logvar_x = self.z_logvar_encoder_x(q_x)

        q_y = self.encoder_y(torch.cat((data[:,1,:], batch_id_tensor), dim=-1))
        qz_mean_y = self.z_mean_encoder_y(q_y)
        qz_logvar_y = self.z_logvar_encoder_y(q_y)
        
        mu = torch.stack([qz_mean_x, qz_mean_y], dim=1)
        logvar = torch.stack([qz_logvar_x, qz_logvar_y], dim=1)
        mu_joint, logvar_joint = self.product_of_experts(mu, logvar)
        z_poe = Normal(mu_joint, torch.exp(logvar_joint).sqrt()).rsample()

        return mu_joint, torch.exp(logvar_joint), z_poe, library_size_factor 

class EncoderExp3POE(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self, 
        num_feature: int,      
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.num_modality = num_modality
        
        modules_x = []
        modules_x.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_x.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_x = nn.Sequential(*modules_x)
        
        modules_y = []
        modules_y.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_y.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_y = nn.Sequential(*modules_y)
        
        modules_z = []
        modules_z.append(nn.Sequential(nn.Linear(num_feature + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_z.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.encoder_z = nn.Sequential(*modules_z)

        self.z_mean_encoder_x = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )
        
        self.z_logvar_encoder_x = nn.Linear(num_hidden_units, dim_latent) 
        
        self.z_mean_encoder_y = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_encoder_y = nn.Linear(num_hidden_units, dim_latent)
            
        self.z_mean_encoder_z = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )

        self.z_logvar_encoder_z = nn.Linear(num_hidden_units, dim_latent)
        
    def product_of_experts(self, mus, logvars):
        dist_vars = torch.exp(logvars)
        mus_joint = torch.sum(mus / dist_vars, dim=1)
        logvars_joint = torch.ones_like(mus_joint)  # batch size
        logvars_joint += torch.sum(torch.ones_like(mus) / dist_vars, dim=1)
        logvars_joint = 1.0 / logvars_joint  # inverse
        mus_joint *= logvars_joint
        logvars_joint = torch.log(logvars_joint)
        return mus_joint, logvars_joint

    # calculate library size
    def calculate_multimodal_size_factor(self, data, num_modality):
        size_factor_list = []
        for modality_idx in range(num_modality):
            size_factor = torch.sum(data[:, modality_idx, :], axis=1).unsqueeze(1)
            size_factor = size_factor.repeat(1, data.shape[2])
            size_factor_list.append(size_factor[:,None,:])

        return torch.cat(size_factor_list, dim=1)
        
    def forward(self, data, batch_id):
        library_size_factor = self.calculate_multimodal_size_factor(data, self.num_modality)

        data = torch.log(data + 1)
        batch_id_tensor = batch_id

        q_x = self.encoder_x(torch.cat((data[:,0,:], batch_id_tensor), dim=-1))
        qz_mean_x = self.z_mean_encoder_x(q_x)
        qz_logvar_x = self.z_logvar_encoder_x(q_x)

        q_y = self.encoder_y(torch.cat((data[:,1,:], batch_id_tensor), dim=-1))
        qz_mean_y = self.z_mean_encoder_y(q_y)
        qz_logvar_y = self.z_logvar_encoder_y(q_y)
            
        q_z = self.encoder_y(torch.cat((data[:,2,:], batch_id_tensor), dim=-1))
        qz_mean_z = self.z_mean_encoder_z(q_z)
        qz_logvar_z = self.z_logvar_encoder_z(q_z)
        
        mu = torch.stack([qz_mean_x, qz_mean_y, qz_mean_z], dim=1)
        logvar = torch.stack([qz_logvar_x, qz_logvar_y, qz_logvar_z], dim=1)
        mu_joint, logvar_joint = self.product_of_experts(mu, logvar)
        z_poe = Normal(mu_joint, torch.exp(logvar_joint).sqrt()).rsample()

        return mu_joint, torch.exp(logvar_joint), z_poe, library_size_factor 

class DecoderExp2POE(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        
        modules_x = []
        modules_x.append(nn.Sequential(nn.Linear(dim_latent + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_x.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.decoder_x = nn.Sequential(*modules_x)     

        self.scale_decoder_x = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )
        
        modules_y = []
        modules_y.append(nn.Sequential(nn.Linear(dim_latent + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_y.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.decoder_y = nn.Sequential(*modules_y)     

        self.scale_decoder_y = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        px = self.decoder_x(torch.cat((z, batch_id), dim=-1))
        px = self.scale_decoder_x(px)
        
        py = self.decoder_y(torch.cat((z, batch_id), dim=-1))
        py = self.scale_decoder_y(py)
        
        px_z_[0]['mean'] = px * library_size_factor[:,0,:] 
        px_z_[1]['mean'] = py * library_size_factor[:,1,:]

        return px_z_

class DecoderExp3POE(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        
        modules_x = []
        modules_x.append(nn.Sequential(nn.Linear(dim_latent + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_x.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.decoder_x = nn.Sequential(*modules_x)     

        self.scale_decoder_x = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )
        
        modules_y = []
        modules_y.append(nn.Sequential(nn.Linear(dim_latent + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_y.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.decoder_y = nn.Sequential(*modules_y)     

        self.scale_decoder_y = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )
        
        modules_z = []
        modules_z.append(nn.Sequential(nn.Linear(dim_latent + num_sample, num_hidden_units), 
                                       nn.BatchNorm1d(num_hidden_units), 
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules_z.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                           nn.BatchNorm1d(num_hidden_units), 
                                           nn.ReLU(),
                                           nn.Dropout(p=dropout_rate)))

        self.decoder_z = nn.Sequential(*modules_z)     

        self.scale_decoder_z = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )  

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        px = self.decoder_x(torch.cat((z, batch_id), dim=-1))
        px = self.scale_decoder_x(px)
        
        py = self.decoder_y(torch.cat((z, batch_id), dim=-1))
        py = self.scale_decoder_y(py)
        
        pz = self.decoder_z(torch.cat((z, batch_id), dim=-1))
        pz = self.scale_decoder_z(pz)
        
        px_z_[0]['mean'] = px * library_size_factor[:,0,:] 
        px_z_[1]['mean'] = py * library_size_factor[:,1,:] 
        px_z_[2]['mean'] = pz * library_size_factor[:,2,:] 

        return px_z_


class EncoderFC(nn.Module):
    """
    Encodes data of ``n_bin`` dimensions into a latent space of ``n_output`` dimensions.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    """

    def __init__(
        self, 
        num_feature: int,      
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_feature = num_feature
        self.num_modality = num_modality
        
        modules = []
        modules.append(nn.Sequential(nn.Linear(num_feature+num_sample, num_hidden_units), 
                                     nn.BatchNorm1d(num_hidden_units), 
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                         nn.BatchNorm1d(num_hidden_units), 
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout_rate)))

        self.encoder = nn.Sequential(*modules)

        self.z_mean_encoder = nn.Sequential(
            nn.Linear(num_hidden_units, dim_latent)
        )
        
        self.z_logvar_encoder = nn.Linear(num_hidden_units, dim_latent) 

    # calculate library size
    def calculate_size_factor(self, data, num_feature):
        size_factor = torch.sum(data[:,0,:], axis=1).unsqueeze(1)
        size_factor = size_factor.repeat(1, num_feature)
        return size_factor
        
    def forward(self, data, batch_id):
        library_size_factor = self.calculate_size_factor(data, self.num_feature)

        data = torch.log(data[:,0,:] + 1)
        
        if batch_id.size()[1] > 1:
            q = self.encoder(torch.cat((data, batch_id), dim=-1))
        else:
            q = self.encoder(data)
        qz_x_mean = self.z_mean_encoder(q)
        qz_x_var = torch.exp(self.z_logvar_encoder(q))

        z = Normal(qz_x_mean, qz_x_var.sqrt()).rsample()

        return qz_x_mean, qz_x_var, z, library_size_factor


class DecoderFC(nn.Module):
    """
    Decodes data from latent space of ``dim_latent`` dimensions ``n_output`` dimensions.
    Parameters
    ----------
    n_input
        The dimensionality of the input: number of bins on the genome of interest
    dim_latent
    n_hidden
    """

    def __init__(
        self, 
        num_feature: int,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.num_modality = num_modality
        
        modules = []
        modules.append(nn.Sequential(nn.Linear(dim_latent+num_sample, num_hidden_units), 
                                     nn.BatchNorm1d(num_hidden_units), 
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout_rate)))
        for _ in range(num_hidden_layers - 1):
            modules.append(nn.Sequential(nn.Linear(num_hidden_units, num_hidden_units), 
                                         nn.BatchNorm1d(num_hidden_units), 
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout_rate)))

        self.decoder = nn.Sequential(*modules)     

        self.scale_decoder = nn.Sequential(
            nn.Linear(num_hidden_units, num_feature),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, batch_id, library_size_factor):
        px_z_ = [{} for _ in range(self.num_modality)]
        
        if batch_id.size()[1] > 1:
            px = self.decoder(torch.cat((z, batch_id), dim=-1))
        else:
            px = self.decoder(z)
        px = self.scale_decoder(px)
        
        px_z_[0]['mean'] = px * library_size_factor

        return px_z_