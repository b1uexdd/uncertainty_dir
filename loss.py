import torch
import numpy as np
import torch.nn as nn




def beta_nll_components(mean, variance, target, beta=0.5):
    """Return beta-NLL and its two per-element components."""
    mse_component = 0.5 * ((target - mean) ** 2 / variance)
    var_component = 0.5 * variance.log()

    if beta > 0:
        weight = variance.detach() ** beta
        mse_component = mse_component * weight
        var_component = var_component * weight

    total_loss = mse_component + var_component
    return total_loss, mse_component, var_component


def beta_nll_loss(mean, variance, target, beta=0.5, e=0):
    """Compute beta-NLL loss."""
    loss, _, _ = beta_nll_components(mean, variance, target, beta=beta)
    return loss



def reverse_ent_to_var(ent, feature_dim=64):
    # log_var = 2h/d - log(2e*pi)
    ent_rescale = 2*ent/feature_dim
    log_const = torch.Tensor([np.log(2*np.pi) + 1]).unsqueeze(-1)
    log_const = log_const.cuda()
    log_const = log_const.repeat(ent.shape[0], 1)
    biased_logvar = ent_rescale - log_const
    var = torch.exp(biased_logvar)
    return var



# estimate the p(z_d)
class KNIFE(nn.Module):
    def __init__(self, args, zc_dim):
        super(KNIFE, self).__init__()
        self.kernel_marg = MargKernel(args.batch_size, zc_dim)

    def forward(self, z_c):  # samples have shape [sample_size, dim]
        marg_ent = self.kernel_marg(z_c)
        condition_ent = self.kernel_marg(z_c)
        return marg_ent

    def learning_loss(self, z_c):
        marg_ent = self.kernel_marg(z_c)
        return marg_ent 




class MargKernel(nn.Module):
    """
    Used to compute p(z_d) but seems to compute the  -log p(z)
    """

    def __init__(self, batch_size, zc_dim, init_samples=None):
        #
        self.K = batch_size
        self.d = zc_dim
        # K is the batch size, d is the feature dimentsion
        self.init_std = 0.01
        super(MargKernel, self).__init__()
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])
        # mean : a
        init_samples = self.init_std * torch.randn(self.K, self.d)
        self.means = nn.Parameter(init_samples, requires_grad=True)  # [K, db]
        # var : A
        diag = self.init_std * torch.randn((1, self.K, self.d))
        self.logvar = nn.Parameter(diag, requires_grad=True)
        #
        tri = self.init_std * torch.randn((1, self.K, self.d, self.d))
        tri = tri.to(init_samples.dtype)
        self.tri = nn.Parameter(tri, requires_grad=True)
        # weight : w
        weigh = torch.ones((1, self.K))
        self.weigh = nn.Parameter(weigh, requires_grad=True)


    def logpdf(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.d, 'x has to have shape [N, d]'
        x = x[:, None, :]
        w = torch.log_softmax(self.weigh, dim=1)
        #
        y = x - self.means
        #
        logvar = self.logvar
        #
        var = logvar.exp()
        y = y * var
        # print(f"Marg : {var.min()} | {var.max()} | {var.mean()}")
        #tri_size = torch.tril(self.tri, diagonal=-1).size()
        #y_size = y[:, :, :, None].size()
        #print(f' the size of the tri is {tri_size} y size is {y_size}')
        #
        y = y + torch.squeeze(torch.matmul(torch.tril(self.tri, diagonal=-1), y[:, :, :, None]), 3)
        #
        y = torch.sum(y ** 2, dim=2)
        #
        y = -y / 2 + torch.sum(torch.log(torch.abs(var) + 1e-8), dim=-1) + w
        y = torch.logsumexp(y, dim=-1)
        return self.logC.to(y.device) + y

    def update_parameters(self, z):
        self.means = z



    def forward(self, x):
        y = -self.logpdf(x)
        #
        #return torch.mean(y)
        return y.unsqueeze(-1)
        # return a (bs, 1)

class CondKernel(nn.Module):
    """
    Used to compute p(z_d | z_c)
    """

    def __init__(self, args, zc_dim, zd_dim, layers=1):
        super(CondKernel, self).__init__()
        self.K, self.d = args.cond_modes, zd_dim
        self.use_tanh = args.use_tanh
        self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])

        self.mu = FF(args, zc_dim, self.d, self.K * zd_dim)
        self.logvar = FF(args, zc_dim, self.d, self.K * zd_dim)

        self.weight = FF(args, zc_dim, self.d, self.K)
        self.tri = None
        if args.cov_off_diagonal == 'var':
            self.tri = FF(args, zc_dim, self.d, self.K * zd_dim ** 2)
        self.zc_dim = zc_dim

    def logpdf(self, z_c, z_d):  # H(z_d|z_c)

        z_d = z_d[:, None, :]  # [N, 1, d]

        w = torch.log_softmax(self.weight(z_c), dim=-1)  # [N, K]
        mu = self.mu(z_c)
        logvar = self.logvar(z_c)
        if self.use_tanh:
            logvar = logvar.tanh()
        var = logvar.exp().reshape(-1, self.K, self.d)
        mu = mu.reshape(-1, self.K, self.d)
        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri is not None:
            tri = self.tri(z_c).reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3)
        z = torch.sum(z ** 2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC.to(z.device) + z

    def forward(self, z_c, z_d):
        z = -self.logpdf(z_c, z_d)
        return torch.mean(z)
    



# (1/1-alpha)*(log \sum z_I^{alpha})
def Renyi_alpha(z, alpha=2):
    # num, dea_dim
    # do we need norm?
    norm_z = torch.nn.functional.softmax(z, dim=-1)
    # num, 1
    renyi_ent = (1/1-alpha)*torch.log(torch.sum(torch.pow(norm_z, alpha), dim=1).unsqueeze(-1))
    #
    return renyi_ent
