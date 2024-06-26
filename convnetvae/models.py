# line ending: unix

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal, Poisson

from .modules import *
from .distributions import *
from .convNetVAE_utils import *


class ConvNetL1VAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_stride: int = 1,
        size_padding_enc: list = [0],
        size_padding_dec: list = [0],
        size_padding_dec_extra: int = 0,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        #self.gene_dispersion = gene_dispersion
        self.num_feature = num_feature
        self.dim_latent = dim_latent
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderConvNetL1(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel, 
            size_stride=size_stride,
            size_padding=size_padding_enc,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderConvNetL1(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            convEnc_1_out=self.encoder.convEnc_1_out,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel,
            size_stride=size_stride,
            size_padding=size_padding_dec,
            size_padding_extra=size_padding_dec_extra,
            dropout_rate=dropout_rate
        )

    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
            
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']
        
        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs,theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence= kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )

        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss

class ConvNetL1VAE_alt(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_stride: int = 1,
        size_padding_enc: list = [0],
        size_padding_dec: list = [0],
        size_padding_dec_extra: int = 0,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        #self.gene_dispersion = gene_dispersion
        self.num_feature = num_feature
        self.dim_latent = dim_latent
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderConvNetL1Alt(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel, 
            size_stride=size_stride,
            size_padding=size_padding_enc,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderConvNetL1(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            convEnc_1_out=self.encoder.convEnc_1_out,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel,
            size_stride=size_stride,
            size_padding=size_padding_dec,
            size_padding_extra=size_padding_dec_extra,
            dropout_rate=dropout_rate
        )

    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
            
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']
        
        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs,theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence= kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )

        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss

class ConvNetL2VAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_stride: int = 1,
        size_padding_enc: list = [0],
        size_padding_dec: list = [0],
        size_padding_dec_extra: int = 0,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        #self.gene_dispersion = gene_dispersion
        self.num_feature = num_feature
        self.dim_latent = dim_latent
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderConvNetL2(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel, 
            size_stride=size_stride,
            size_padding=size_padding_enc,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderConvNetL2(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            convEnc_1_out=self.encoder.convEnc_1_out,
            convEnc_2_out=self.encoder.convEnc_2_out, 
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel,
            size_stride=size_stride,
            size_padding=size_padding_dec,
            size_padding_extra=size_padding_dec_extra,
            dropout_rate=dropout_rate
        )

    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
            
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']
        
        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs, theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence = kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )

        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss


class ConvNetL3VAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 128,
        num_channel: list = [0],
        size_conv_kernel: int = 3, 
        size_stride: int = 1,
        size_padding_enc: list = [0],
        size_padding_dec: list = [0],
        size_padding_dec_extra: int = 0,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        #self.gene_dispersion = gene_dispersion
        self.num_feature = num_feature
        self.dim_latent = dim_latent
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderConvNetL3(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel, 
            size_stride=size_stride,
            size_padding=size_padding_enc,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderConvNetL3(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            convEnc_1_out=self.encoder.convEnc_1_out,
            convEnc_2_out=self.encoder.convEnc_2_out, 
            convEnc_3_out=self.encoder.convEnc_3_out, 
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_channel=num_channel,
            size_conv_kernel=size_conv_kernel,
            size_stride=size_stride,
            size_padding=size_padding_dec,
            size_padding_extra=size_padding_dec_extra,
            dropout_rate=dropout_rate
        )

    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
            
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']
        
        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs, theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence = kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )

        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss

class poeExp2VAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderExp2POE(
            num_feature=num_feature,
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderExp2POE(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
 
    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        #px_ = px_dict
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']

        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs, theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence = kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )
        
        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss

class poeExp3VAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderExp3POE(
            num_feature=num_feature,
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderExp3POE(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
 
    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        #px_ = px_dict
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']

        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs, theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence= kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )
        
        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss


class fcVAE(nn.Module):
    r"""
    Variational Autoencoder for 10x multiome ATAC and gene expression data.
    Parameters
    ----------
    n_bin
        The dimensionality of the input: number of bins on the genome of interest
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    dim_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        num_feature: int,
        count_dist: str,
        num_modality: int,
        num_sample: int,
        dim_latent: int = 20,
        num_hidden_units: int = 256,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.2,
        beta: float = 1
    ):

        super().__init__()
        
        self.beta = beta
        self.count_dist = count_dist
        
        if self.count_dist == 'NegativeBinomial':
            self.px_r = torch.nn.Parameter(torch.randn(num_feature)) # all cells share the same dispersion parameter of NB for each gene
        if self.count_dist == 'Poisson':
            self.px_r = None
        
        self.encoder = EncoderFC(
            num_feature=num_feature,
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
        
        self.decoder = DecoderFC(
            num_feature=num_feature, 
            num_modality=num_modality,
            num_sample=num_sample,
            dim_latent=dim_latent,
            num_hidden_units=num_hidden_units,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
 
    def inference(
        self,
        data,
        batch_id
    ):
        
        qz_x_mean, qz_x_var, z, library_size_factor = self.encoder(data, batch_id)
        qz_x_ = dict(qz_x_mean=qz_x_mean, 
                     qz_x_var=qz_x_var, 
                     z=z, 
                     library_size_factor=library_size_factor)

        return qz_x_
        
    def generative(
        self,
        z,
        batch_id,
        library_size_factor
    ):

        px_z_ = self.decoder(z, batch_id, library_size_factor)
        
        return px_z_, torch.exp(self.px_r) if self.count_dist == 'NegativeBinomial' else None

    def get_reconstruction_loss(
        self,
        data,
        px_z_,
        px_r
    ):
        """Compute reconstruction loss."""
        #px_ = px_dict
        reconst_loss_ = [{} for _ in range(len(px_z_))]
        
        for modality_idx in range(len(reconst_loss_)):
            if px_r is None:
                reconst_loss_[modality_idx] = -Poisson(px_z_[modality_idx]['mean']).log_prob(data[:,modality_idx,:]).sum(dim=-1) # dim=batch_size
            else:
                reconst_loss_[modality_idx] = -NegativeBinomial(mu=px_z_[modality_idx]['mean'],
                                                                theta=px_r,
                                                                ).log_prob(data[:,modality_idx,:]).sum(dim=-1)
            
        return reconst_loss_
    
    def get_loss(
        self,
        data,
        inference_outputs,
        generative_outputs,
        theta,
        beta
    ):
        """
        """
        qz_x_mean = inference_outputs['qz_x_mean']
        qz_x_var = inference_outputs['qz_x_var']
        library_size_factor = inference_outputs['library_size_factor']

        # Reconstruction loss
        reconst_loss_ = self.get_reconstruction_loss(data, generative_outputs, theta)
        reconst_loss_sum = torch.stack(reconst_loss_, dim=-1).sum(-1)
        
        # KL divergence
        kl_divergence= kl(Normal(qz_x_mean, torch.sqrt(qz_x_var)), Normal(0, 1)).sum(dim=1) 

        loss = torch.mean(
            reconst_loss_sum
            + beta * kl_divergence
        )
        
        return loss
        
    def forward(self, data, batch_id):
        inference_outputs = self.inference(data=data,
                                           batch_id=batch_id)
        generative_outputs, theta = self.generative(z=inference_outputs['z'], 
                                                    batch_id=batch_id,
                                                    library_size_factor=inference_outputs['library_size_factor'])
        loss = self.get_loss(data, inference_outputs, generative_outputs, theta, beta = self.beta)
        
        return inference_outputs, generative_outputs, loss