import torch
import torch.nn as nn
import torch.nn.functional as F
class KLDivergenceLoss(nn.Module):
    def forward(self, mu, logvar):
        # KL(q(z|x)||p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_divergence.mean()

class ReconstructionLoss(nn.Module):
    def forward(self, x, x_recon):
        return F.mse_loss(x_recon, x)

class GANLossD(nn.Module):
    def forward(self, real, fake):
        # Hinge loss for discriminator
        loss_real = torch.mean(F.relu(1.0 - real))
        loss_fake = torch.mean(F.relu(1.0 + fake))
        loss = loss_real + loss_fake
        return loss.mean()

class GANLossG(nn.Module):
    def forward(self, fake):
        # Generator tries to maximize D(G(z))
        loss = -torch.mean(fake)
        return loss.mean()

