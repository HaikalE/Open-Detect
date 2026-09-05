import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import net


class OpenDetectNet(nn.Module):
    def __init__(self, arch='resnet18', channel=3, latent_dim=128, n_classes=10, temp_inter=1.0, temp_intra=1, init=True, decoder_version=2):
        super(OpenDetectNet, self).__init__()
        self.arch = arch
        self.channel = channel
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.temp_inter = temp_inter
        self.temp_intra = temp_intra
        self.decoder_version = decoder_version
        self.encoder, self.decoder = net(self.arch, self.channel, self.latent_dim, decoder_version=decoder_version)
        # Keep parameters device agnostic. The caller moves the complete model
        # to CUDA or CPU with model.to(device).
        self.prototypes = nn.Parameter(torch.randn(self.n_classes, self.latent_dim), requires_grad=True)
        if init:
            nn.init.kaiming_normal_(self.prototypes)

    def sampler(self, mu, logvar):
        # Reparameterization trick for sampling latent variable z
        std = torch.exp(0.5 * logvar)
        if self.training:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z

    def distance(self, latent_z, prototypes):
        # Compute squared Euclidean distance between latent_z and prototypes
        matrixA_square = torch.sum(latent_z ** 2, 1, keepdim=True)
        matrixB_square = torch.sum(prototypes ** 2, 1).unsqueeze(0)
        product_A_B = torch.matmul(latent_z, prototypes.t())
        return matrixA_square + matrixB_square - 2 * product_A_B

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        # KL divergence between N(mu, sigma) and all prototype Gaussians N(mu_w, I)
        kl_div = self.distance(mean, prototypes) + torch.sum(logvar.exp() - logvar - 1, dim=1, keepdim=True)
        return 0.5 * kl_div

    def forward(self, x):
        mu, logvar, lateral_z = self.encoder(x)
        latent_z = self.sampler(mu, logvar)
        dist = self.distance(latent_z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes)
        recon_x = self.decoder(latent_z, lateral_z)
        return latent_z, dist, kl_div, recon_x

    def loss(self, x, y):
        latent_z, _, kl_div, x_recon = self.forward(x)

        # Equations 17-18 classify samples using the KL divergence between
        # q(z|x)=N(mu_x, sigma_x) and every Gaussian prototype N(mu_y, I).
        preds = torch.argmin(kl_div, dim=1)

        # KL divergence to the ground-truth class prototype (Eq. 15).
        y_one_hot = F.one_hot(y, num_classes=self.n_classes).bool()
        kl_div_y = kl_div[y_one_hot].view(len(kl_div), 1)

        # Generative constraint (Eq. 13): reconstruction + conditional KL.
        rec_loss = F.mse_loss(x_recon, x)
        kld_loss = kl_div_y.mean()

        # Discriminative constraint (Eq. 18): negative log q(y|x), where
        # q(y|x) is a softmax over negative KL divergences to all prototypes.
        if self.temp_inter <= 0:
            raise ValueError('temp_inter must be greater than zero')
        class_logits = -kl_div / self.temp_inter
        dis_loss = F.cross_entropy(class_logits, y)

        # The paper does not add a separate entropy term to Eq. 13 or Eq. 20.
        loss = {'dis': dis_loss, 'rec': rec_loss, 'kld': kld_loss}
        return latent_z, x_recon, preds, loss


def train_model(model, args, train_loader, epoch, optimizer):
    # Training loop for one epoch
    num_epochs = args.epoch
    model.train()
    print('Current learning rate is {}'.format(optimizer.param_groups[0]['lr']))
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('*' * 70)
    train_corrects = 0
    running_loss = {}
    for i, (image, label) in enumerate(train_loader):
        device = next(model.parameters()).device
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        _, _, preds, loss = model.loss(image, label)
        total_loss = args.lamda * (loss['rec'] + loss['kld']) + (1 - args.lamda) * loss['dis']
        if not torch.isfinite(total_loss):
            raise FloatingPointError('Non-finite training loss at epoch {}, batch {}'.format(epoch + 1, i))
        loss['total'] = total_loss
        total_loss.backward()
        optimizer.step()
        for k in loss.keys():
            running_loss[k] = loss.get(k, 0).item() + running_loss.get(k, 0)
        train_corrects += torch.sum(preds == label.data)
    train_acc = train_corrects.item() / len(train_loader.dataset)
    train_loss = {k: running_loss.get(k, 0) / len(train_loader) for k in running_loss.keys()}
    print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(
        train_corrects, len(train_loader.dataset), train_acc))
    print('Train loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f})] + (1-{})*dis({:.3f})'.format(
        train_loss['total'], args.lamda, train_loss['rec'], train_loss['kld'],
        args.lamda, train_loss['dis']))


def validate_model(model, args, val_loader, epoch):
    # Validation loop for one epoch
    model.eval()
    val_corrects = 0.0
    val_running_loss = {'total': 0.0, 'rec': 0.0, 'kld': 0.0, 'dis': 0.0}
    for image, label in val_loader:
        with torch.no_grad():
            device = next(model.parameters()).device
            image, label = image.to(device), label.to(device)
            latent_z, x_recon, preds, loss = model.loss(image, label)
            total_loss = args.lamda * (loss['rec'] + loss['kld']) + (1 - args.lamda) * loss['dis']
            loss['total'] = total_loss
            for k in loss.keys():
                val_running_loss[k] = loss.get(k, 0).item() + val_running_loss.get(k, 0)
            val_corrects += torch.sum(preds == label.data)
    val_acc = val_corrects.item() / len(val_loader.dataset)
    val_loss = {k: val_running_loss.get(k, 0) / len(val_loader) for k in val_running_loss.keys()}
    print('Val corrects: {} Val samples: {} Val accuracy: {}'.format(
        val_corrects, len(val_loader.dataset), val_acc))
    print('Val loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f})] + (1-{})*dis({:.3f})'.format(
        val_loss['total'], args.lamda, val_loss['rec'], val_loss['kld'],
        args.lamda, val_loss['dis']))
    print('*' * 70)
    return val_acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = OpenDetectNet('resnet18', 1, 128, 24, 0.1, 1)
    net.to(device)
    input = torch.randn(1, 1, 40, 40).to(device)
    print(net(input))
    
    # print(net.parameters)
