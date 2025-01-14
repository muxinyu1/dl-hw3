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
        # 判别器损失 = 0.5 * (max(0, 1-D(x)) + max(0, 1+D(x̂)))  
        # real: D(x) - 真实样本的判别器输出  
        # fake: D(x̂) - 重构样本的判别器输出  
        
        loss_real = torch.mean(torch.clamp(1 - real, min=0))  # max(0, 1-D(x))  
        loss_fake = torch.mean(torch.clamp(1 + fake, min=0))  # max(0, 1+D(x̂))  
        
        loss = 0.5 * (loss_real + loss_fake)  
        return loss  


class GANLossG(nn.Module):  
    def forward(self, fake):  
        # 生成器损失 = -D(x̂)  
        # fake: D(x̂) - 重构样本的判别器输出  
        
        loss = -fake  # 直接取负  
        return loss.mean()  

