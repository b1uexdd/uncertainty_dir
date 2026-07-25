import torch.nn as nn
import torchvision
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import logging
import math

import torch.nn as nn

try:
    from fds import FDS
except ImportError:
    FDS = None


print = logging.info


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_conformal_new(nn.Module):
    """AgeDB ResNet with one point head and two conformal interval heads."""

    def __init__(
            self,
            block,
            layers,
            fds,
            bucket_num,
            bucket_start,
            start_update,
            start_smooth,
            kernel,
            ks,
            sigma,
            momentum,
            dropout=None):
        self.inplanes = 64
        super(ResNet_conformal_new, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        feature_dim = 512 * block.expansion
        self.linear = nn.Linear(feature_dim, 1)
        self.interval_lower = nn.Linear(feature_dim, 1)
        self.interval_upper = nn.Linear(feature_dim, 1)

        if fds:
            if FDS is None:
                raise ImportError(
                    'FDS was enabled, but fds.py is not available.'
                )
            self.FDS = FDS(
                feature_dim=feature_dim,
                bucket_num=bucket_num,
                bucket_start=bucket_start,
                start_update=start_update,
                start_smooth=start_smooth,
                kernel=kernel,
                ks=ks,
                sigma=sigma,
                momentum=momentum,
            )
        self.fds = fds
        self.start_smooth = start_smooth

        self.use_dropout = bool(dropout)
        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                )
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        """Return the shared 2048-dimensional ResNet50 representation."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(
            self,
            x,
            targets=None,
            epoch=None,
            return_intervals=False):
        encoding = self.extract_features(x)
        head_features = encoding

        if self.training and self.fds:
            if epoch is None:
                raise ValueError('epoch is required when training with FDS')
            if epoch >= self.start_smooth:
                head_features = self.FDS.smooth(
                    head_features, targets, epoch
                )

        if self.use_dropout:
            head_features = self.dropout(head_features)

        prediction = self.linear(head_features)

        if return_intervals:
            lower = self.interval_lower(head_features)
            upper = self.interval_upper(head_features)
            return prediction, lower, upper, encoding

        # Keep the original AgeDB-DIR/FDS return contract for train.py.
        if self.training and self.fds:
            return prediction, encoding
        return prediction


def resnet50(**kwargs):
    return ResNet_conformal_new(Bottleneck, [3, 4, 6, 3], **kwargs)


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

##################################################
class ResNet_lenet_conformal(nn.Module):
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
        
        self.features = nn.Sequential(
                    nn.Conv2d(3, 6, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 120, kernel_size=5),
                    nn.ReLU(inplace=True),
                    # Makes the MLP input independent of the input image size.
                    nn.AdaptiveAvgPool2d((1, 1)))
        
        self.mlp = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(120, 84),
                    nn.ReLU(inplace=True))
        #
        self.pred_head =  nn.Sequential(nn.Linear(fc_inputs, 1))
        self.interval_upper = nn.Sequential(nn.Linear(84, 1))
        self.interval_lower = nn.Sequential(nn.Linear(84, 1))
        #self.pred_head = nn.Linear(fc_inputs, 1)
        #self.interval_head = nn.Linear(fc_inputs, 2)

        
        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #interval z
        z_interval = self.features(x)
        z_interval = self.mlp(x)
        #pred head
        z = self.Flatten(z)
        y_pred = self.pred_head(x)
        #z_cp = z.detach()
        #lower_upper = self.interval_head(z)
        z_interval = self.Flatten(z_interval)
        y_lower, y_upper = self.interval_lower(z_interval), self.interval_upper(z_interval)
        #
        #y_preds = self.model_linear(z)
        #
        #y_pred, y_lower, y_upper = torch.chunk(y_preds, 3, dim=-1)
        # the ouput dim of the embed is : bs,3
        #print(f' y pred shape {y_pred.shape} y preds shape {y_preds.shape}')
        #
        return y_pred, y_lower, y_upper, z, z_interval
