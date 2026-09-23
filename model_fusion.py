"""Experimental image+packet fusion, isolated from the original OpenDetectNet."""

import torch
from torch import nn
from torch.nn import functional as F

from model import OpenDetectNet
from networks.temporal import TemporalBiGRU


class FusionEncoder(nn.Module):
    def __init__(self, image_encoder, latent_dim=128, sequence_features=3):
        super().__init__()
        self.image_encoder = image_encoder
        # The fusion heads replace the original image-only latent heads.
        del self.image_encoder.mu
        del self.image_encoder.logvar
        self.temporal = TemporalBiGRU(input_features=sequence_features, hidden_size=64)
        self.image_projection = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.temporal_projection = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, image, sequence, lengths):
        pooled, lateral = self.image_encoder.forward_features(image)
        packet_vector = self.temporal(sequence, lengths)
        fused = torch.cat((self.image_projection(pooled),
                           self.temporal_projection(packet_vector)), dim=1)
        return self.mu(fused), self.logvar(fused), lateral


class OpenDetectFusionNet(OpenDetectNet):
    """Keep the original decoder, Gaussian prototypes, and loss terms."""

    def __init__(self, sequence_features=3, **kwargs):
        super().__init__(**kwargs)
        if self.arch != 'resnet18':
            raise ValueError('Fusion currently supports resnet18 only')
        self.encoder = FusionEncoder(self.encoder, self.latent_dim, sequence_features)

    def forward(self, image, sequence, lengths):
        mu, logvar, lateral = self.encoder(image, sequence, lengths)
        z = self.sampler(mu, logvar)
        dist = self.distance(z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes)
        recon = self.decoder(z, lateral)
        return z, dist, kl_div, recon

    def loss(self, image, sequence, lengths, label):
        z, _, kl_div, recon = self.forward(image, sequence, lengths)
        if self.temp_inter <= 0:
            raise ValueError('temp_inter must be greater than zero')
        target_kl = kl_div.gather(1, label[:, None]).mean()
        terms = {'dis': F.cross_entropy(-kl_div / self.temp_inter, label),
                 'rec': F.mse_loss(recon, image), 'kld': target_kl}
        return z, recon, kl_div.argmin(dim=1), terms
