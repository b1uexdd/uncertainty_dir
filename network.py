import torch.nn as nn
import torchvision
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# group based network arch : output group*2
class ResNet_regression(nn.Module):
    def __init__(self, args=None):
        super(ResNet_regression, self).__init__()
        self.args = args
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = args.groups * 2
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z
    

# single network arch, output 1
class ResNets(nn.Module):
    def __init__(self, args=None):
        super(ResNets, self).__init__()
        self.args = args
        self.groups = args.groups
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        output_dim = 1
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, output_dim))
        #

        #self.mode = args.mode
        self.sigma = args.sigma
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        # the ouput dim of the embed is : 512
        #
        return y_hat, z
    

# reparameterization for mu and sigma in ResNet
'''
class reparam_ResNet(nn.Module):
    def __init__(self, name, args=None):
        super(ResNet_regression, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        # mean ,std estimation networks
        self.emb2mu = nn.Linear(dim_in, 512)
        self.emb2std = nn.Linear(dim_in, 512)
        #


    def estimate(self, emb):
        """Estimates mu and std from the given input embeddings."""
        mean = self.emb2mu(emb)
        std = torch.nn.functional.softplus(self.emb2std(emb))
        return mean, std


    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1])
        return mu + std * z
    

    def forward(self, x, y):
        feat = self.encoder(x)
        mean, std = self.estimate(feat)
        z_reparameterized = self.reparameterize(mean, std)
        y_pred = self.regressor(feat)
        #
        return z_reparameterized, y_pred
'''


#########################################################################################################
'''
class Guassian_uncertain_ResNet(nn.Module):
    def __init__(self, name='resnet50', norm=False, weight_norm= False):
        super(Guassian_uncertain_ResNet, self).__init__()
        backbone, dim_in = model_dict[name]
        self.encoder = backbone()
        self.norm = norm
        self.weight_norm = weight_norm
        #
        self.feature_rescale = nn.Linear(dim_in, 64)
        
        if self.weight_norm:
            self.regressor = torch.nn.utils.weight_norm(nn.Linear(dim_in, 2), name='weight')
        else:
           self.regressor = nn.Linear(dim_in, 2)
        
        self.guassian_head = GaussianLikelihoodHead(inp_dim=64, outp_dim=1, use_spectral_norm_mean=weight_norm)
        #
        self.feature_dim = 64       

    def forward(self, x):
        feat = self.encoder(x)
        if self.norm:
            feat = F.normalize(feat, dim=-1)
        feat = self.feature_rescale(feat)

        mean, var = self.guassian_head(feat)

        return feat, mean, var  
'''



class GaussianLikelihoodHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        outp_dim,
        initial_var=1,
        min_var=1e-8,
        max_var=100,
        mean_scale=1,
        var_scale=1,
        use_spectral_norm_mean=False,
        use_spectral_norm_var=False,
    ):
        super().__init__()
        assert min_var <= initial_var <= max_var

        self.min_var = min_var
        self.max_var = max_var
        self.init_var_offset = np.log(np.exp(initial_var - min_var) - 1)

        self.mean_scale = mean_scale
        self.var_scale = var_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.mean = nn.Linear(inp_dim, outp_dim)

        if use_spectral_norm_var:
            self.var = nn.utils.spectral_norm(nn.Linear(inp_dim, outp_dim))
        else:
            self.var = nn.Linear(inp_dim, outp_dim)

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = self.var(inp) * self.var_scale

        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)


        return mean, var

    

##################################################

class ResNet_conformal(nn.Module):
    def __init__(self, args=None):
        super(ResNet_conformal, self).__init__()
        self.args = args
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        #self.norm, self.weight_norm = args.norm, args.weight_norm
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.pred_head =  nn.Sequential(nn.Linear(fc_inputs, 1))
        self.interval_upper = nn.Sequential(nn.Linear(fc_inputs, 1))
        self.interval_lower = nn.Sequential(nn.Linear(fc_inputs, 1))
        #self.pred_head = nn.Linear(fc_inputs, 1)
        #self.interval_head = nn.Linear(fc_inputs, 2)

        
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        y_pred = self.pred_head(z)
        #z_cp = z.detach()
        #lower_upper = self.interval_head(z)
        y_lower, y_upper = self.interval_lower(z), self.interval_upper(z)
        #
        #y_preds = self.model_linear(z)
        #
        #y_pred, y_lower, y_upper = torch.chunk(y_preds, 3, dim=-1)
        # the ouput dim of the embed is : bs,3
        #print(f' y pred shape {y_pred.shape} y preds shape {y_preds.shape}')
        #
        return y_pred, y_lower, y_upper, z


##################################################

class ResNet_cls_uncertain(nn.Module):
    def __init__(self, args=None):
        super(ResNet_cls_uncertain, self).__init__()
        self.args = args
        exec('self.model = torchvision.models.resnet{}(pretrained=False)'.format(args.model_depth))
        #
        #self.norm, self.weight_norm = args.norm, args.weight_norm
        #
        fc_inputs = self.model.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(self.model.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.reg_head =  nn.Sequential(nn.Linear(fc_inputs, 1))
        self.cls_head =  nn.Sequential(nn.Linear(fc_inputs, 10))
        #self.pred_head = nn.Linear(fc_inputs, 1)
        #self.interval_head = nn.Linear(fc_inputs, 2)

        
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        y_pred = self.pred_head(z)
        #z_cp = z.detach()
        #lower_upper = self.interval_head(z)
        cls_pred = self.cls_head(z)
        #
        return y_pred, cls_pred, z