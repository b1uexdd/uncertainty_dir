import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from agedb import *
from utils import AverageMeter, shot_count, shot_metric, setup_seed, per_label_var, per_label_mae, per_label_frobenius_norm, label_uncertainty_accumulation, uncertainty_accumulation
import torch
from loss import *
from network import resnet50
import torch.optim as optim
import time
from scipy.stats import gmean
from split_CP import coverage_loss, calibrate_qhat_from_batch, calibrate_qhat_splitCP, cqr_pinball, interval_minimization
import torch.nn.functional as F
import itertools


# current sota 7.73, 7.46, 7.76, 10.08
# g 10 lr 0.0002 epoch 450 sigma 2 temp 0.02

import os
os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training/optimization related
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/root/autodl-tmp/data', help='data directory')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint',
                    help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='',
                    help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--warmup_lr', type=float, default=1e-3,
                    help='initial learning rate used only by Stage 1')
parser.add_argument(
    '--stage2_lr',
    type=float,
    default=1e-4,
    help='Stage 2 learning rate for the first phase',
)
parser.add_argument(
    '--stage2_lr_after',
    type=float,
    default=5e-5,
    help='Stage 2 learning rate after switching',
)
parser.add_argument(
    '--stage2_lr_switch_epoch',
    type=int,
    default=5,
    help='number of Stage 2 epochs using --stage2_lr',
)
parser.add_argument('--stage3_extractor_lr', type=float, default=None,
                    help='Stage 3 extractor learning rate; defaults to --lr')
parser.add_argument('--stage3_regressor_lr', type=float, default=None,
                    help='Stage 3 point-head learning rate; defaults to --lr')
parser.add_argument('--stage3_weight_decay', type=float, default=1e-5,
                    help='weight decay used by both Stage 3 optimizers')

parser.add_argument('--warmup_epoch', default=90, type=int, help='warm-up epochs')
parser.add_argument('--quantile_epoch',default=30,type=int,)
parser.add_argument('--nll_epoch',default=30,type=int,)

parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*',
                    default=[60, 80], help='lr schedule (when to drop lr by 10x)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--print_freq', type=int,
                    default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
#
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--la', action='store_true',
                    help='if use logit adj to train the imbalance')
parser.add_argument('--model_depth', type=int, default=50,
                    help='resnet 18 or resnnet 50')
parser.add_argument('--init_noise_sigma', type=float,
                    default=1., help='initial scale of the noise')
parser.add_argument('--tsne', type=bool, default=False,
                    help='draw tsne or not')
parser.add_argument('--g_dis', action='store_true',
                    help='if dynamically adjust the tradeoff')
parser.add_argument('--gamma', type=float, default=5, help='tradeoff rate')
#
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the wole datasets')
#
parser.add_argument('--tau', default=1, type=float,
                    help=' tau for logit adjustment ')
parser.add_argument('--ranked_contra', action='store_true')
parser.add_argument('--temp', type=float, help='temperature for contrastive loss', default=0.07)
parser.add_argument('--contra_ratio', type=float, help='ratio fo contrastive loss', default=1)
#
parser.add_argument('--soft_label', action='store_true')
parser.add_argument('--ce', action='store_true',  help='if use the cross_entropy /la or not')
parser.add_argument('--output_file', type=str, default='result_')
parser.add_argument('--scale', type=float, default=1, help='scale of the sharpness in soft label')
#parser.add_argument('--diversity', type=float, default=0, help='scale of the diversity loss in regressor output')
parser.add_argument('--fd_ratio', type=float, default=0, help='scale of the diversity loss in z')
parser.add_argument('--beta', default=0.5, type=float,  help='beta for nll')
parser.add_argument('--variance_mse_threshold', type=float, default=1,
                    help='after warmup, switch samples with variance below this threshold from NLL to MSE')
parser.add_argument('--ema_variance_alpha', type=float, default=0.9,
                    help='distance decay for majority-label prediction variances used to replace median/low variances')

parser.add_argument('--ema_epoch_decay', type=float, default=0.9,
                    help='temporal EMA decay across NLL epochs')
parser.add_argument('--variance_floor', type=float, default=1e-6,
                    help='minimum variance used by beta-NLL and the EMA lookup')
parser.add_argument('--disable_ema_variance', action='store_true',
                    help='use detached per-sample interval variance instead of the per-label EMA lookup')
parser.add_argument('--lamb', default=0.8, type=float,  help='lamb for coverage')
parser.add_argument('--weight', default=0.1, type=float,  help='weight for interval_loss in total loss')
parser.add_argument('--alpha', default=0.1, type=float,  help='miscoverage level for conformal calibration')
parser.add_argument('--cp_mode', type=str, default='hybrid', choices=['cqr', 'split', 'hybrid'])
parser.add_argument('--warmup_ckpt_path', type=str, default='',
                    help='path to save the final warmup checkpoint; defaults to <store_root>/<store_name>/warmup_final.pth.tar')
parser.add_argument('--resume_warmup_ckpt', type=str, default='',
                    help='path to a saved warmup checkpoint to resume from')
parser.add_argument('--stage2_ckpt_path', type=str, default='',
                    help='path to save the quantile-stage checkpoint; defaults to <store_root>/<store_name>/stage2_final.pth.tar')
parser.add_argument('--resume_stage2_ckpt', type=str, default='',
                    help='load a quantile-stage checkpoint and skip directly to Stage 3')
parser.add_argument('--skip_stage2_checkpoint_save', action='store_true',
                    help='do not save the Stage 2 checkpoint; useful for grid search')
parser.add_argument('--skip_final_model_save', action='store_true',
                    help='evaluate the final model without saving it; useful for grid search')
parser.add_argument('--lightweight_stage3_checkpoint', action='store_true',
                    help='save only model weights and selection metadata for the Stage 3 best checkpoint')
parser.add_argument('--skip_stage1_test', action='store_true',
                    help='skip the Stage 1 test-set pass; useful for grid search')
#
parser.add_argument('--asymm', action='store_true', help='if use the asymmetric soft label')
parser.add_argument('--weight_norm', action='store_true', help='if use the weight norm for train')
parser.add_argument('--feature_norm', action='store_true', help='if use the feature norm for train')
#
# MSE only, else NLL
parser.add_argument('--MSE', action='store_true', help='only use MSE or not')
parser.add_argument('--MAE', action='store_true', help='only use MAE or not')
# first reweight and then judge if we can use LDS
parser.add_argument('--reweight', type=str, default='inv',  choices=['inv', 'sqrt_inverse'],
                    help='weight : inv or sqrt_inv')
parser.add_argument('--smooth', default='none', choices=['lds', 'none'], help='use LDS or not')
parser.add_argument('--inv_method', default='cqr_pinball', choices=['split_cp', 'cqr_pinball', 'cqr_coverage'], help='use which method to train interval module')
#
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad_(trainable)


BACKBONE_MODULE_NAMES = (
    'conv1',
    'bn1',
    'layer1',
    'layer2',
    'layer3',
    'layer4',
)


def backbone_parameters(model):
    """Return only the trainable ResNet50 feature-extractor parameters."""
    return itertools.chain.from_iterable(
        getattr(model, name).parameters()
        for name in BACKBONE_MODULE_NAMES
    )


def set_backbone_trainable(model, trainable):
    for name in BACKBONE_MODULE_NAMES:
        set_trainable(getattr(model, name), trainable)


def get_checkpoint_model_state(checkpoint):
    """Accept train.py, EMA, and bare state-dict checkpoint formats."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f'Checkpoint must be a dict, got {type(checkpoint).__name__}.'
        )

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if state_dict and all(
            key.startswith('module.') for key in state_dict):
        state_dict = {
            key[len('module.'):]: value
            for key, value in state_dict.items()
        }
    return state_dict


def load_checkpoint_model(model, checkpoint, checkpoint_path):
    """Load an exactly matching enhanced-ResNet50 checkpoint."""
    state_dict = get_checkpoint_model_state(checkpoint)
    model_state = model.state_dict()

    missing_keys = sorted(set(model_state) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(model_state))
    shape_mismatches = sorted(
        (
            key,
            tuple(state_dict[key].shape),
            tuple(model_state[key].shape),
        )
        for key in set(model_state).intersection(state_dict)
        if state_dict[key].shape != model_state[key].shape
    )

    if missing_keys or unexpected_keys or shape_mismatches:
        raise RuntimeError(
            f'Checkpoint architecture mismatch: {checkpoint_path}\n'
            f'Missing keys: {missing_keys}\n'
            f'Unexpected keys: {unexpected_keys}\n'
            f'Shape mismatches (key, checkpoint, model): '
            f'{shape_mismatches}\n'
            'Use a checkpoint produced by train.py with the same enhanced '
            'ResNet50 and the same FDS setting.'
        )

    model.load_state_dict(state_dict, strict=True)


def adjust_stage1_learning_rate(optimizer, epoch, args):
    lr = args.warmup_lr
    for milestone in args.schedule:
        if epoch >= milestone:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_data_loader(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                        split='train', reweight=args.reweight, group_num=args.groups, smooth=args.smooth)   
    # Deterministic view of the training split for interval evaluation.
    train_eval_dataset = AgeDB(data_dir=args.data_dir, df=df_train,
                        img_size=args.img_size, split='val', group_num=args.groups)
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                        img_size=args.img_size, split='test', group_num=args.groups)
    #
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, train_eval_loader, val_loader, test_loader, train_labels


def resolve_stage_mode(args, epoch):
    if args.cp_mode == 'hybrid':
        return 'warmup' if epoch < args.warmup_epoch else 'train'
    return args.cp_mode


def get_checkpoint_dir(args):
    return os.path.join(args.store_root, args.store_name) if args.store_name else args.store_root

def get_warmup_checkpoint_path(args):
    if args.warmup_ckpt_path:
        return args.warmup_ckpt_path
    return os.path.join(get_checkpoint_dir(args), 'warmup_final.pth.tar')


def get_stage2_checkpoint_path(args):
    if args.stage2_ckpt_path:
        return args.stage2_ckpt_path
    return os.path.join(get_checkpoint_dir(args), 'stage2_final.pth.tar')


def reduce_batch_loss(loss, weight, smooth_mode):
    if smooth_mode == 'lds':
        return (loss * weight.expand_as(loss)).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


@torch.no_grad()
def get_prediction_variance(model, train_loader):
    """Compute the mean squared prediction residual for each label."""
    model.eval()
    label_squared_residuals = defaultdict(list)

    for x, y, _ in train_loader:
        x = x.to(device)
        y_pred, _, _, _ = model(x, return_intervals=True)

        predictions = y_pred.reshape(-1).cpu().tolist()
        labels = y.reshape(-1).cpu().tolist()

        for label, prediction in zip(labels, predictions):
            squared_residual = (float(prediction) - float(label)) ** 2
            label_squared_residuals[int(label)].append(squared_residual)

    label_variance = {}
    for label, squared_residuals in label_squared_residuals.items():
        residual_tensor = torch.tensor(
            squared_residuals, dtype=torch.float32
        )
        label_variance[label] = residual_tensor.mean().item()

    return label_variance
        
@torch.no_grad()
def infer_per_label_prediction_variance(model, data_loader):
    model.eval()

    uppers = []
    lowers = []
    label_list = []

    for x, y, _ in data_loader:
        x = x.to(device)
        z = model.extract_features(x)
        upper = model.interval_upper(z)
        lower = model.interval_lower(z)

        label_list.append(y.detach())
        uppers.append(upper.detach())
        lowers.append(lower.detach())

    uppers = torch.cat(uppers, dim=0).cpu()
    lowers = torch.cat(lowers, dim=0).cpu()
    labels = torch.cat(label_list, dim=0).cpu()

    interval_variance = {}
    variance_sum = defaultdict(float)
    label_count = defaultdict(int)

    for upper, lower, label in zip(uppers, lowers, labels):
        variance = ((upper - lower) / 2.5632) ** 2

        label = int(label.item())
        variance = variance.item()

        variance_sum[label] += float(variance)
        label_count[label] += 1

    interval_variance = {label: variance_sum[label] / label_count[label] for label in label_count}

    return interval_variance, dict(label_count)


def using_ema_replace_low_and_med_and_maj(
        interval_variance, prediciton_variance, previous_variance,
        maj, med, low,
        alpha=0.9, variance_floor=1e-6):

    if not 0 < alpha <= 1:
        raise ValueError('--ema_variance_alpha must be in (0, 1].')
    if variance_floor <= 0:
        raise ValueError('--variance_floor must be positive.')

    if previous_variance is None:
        previous_variance = dict(interval_variance)
    else:
        previous_variance = {int(k): float(v) for k, v in previous_variance.items()}

    majority_labels = sorted(int(label) for label in maj if int(label) in interval_variance)
    tail_labels = sorted(set(map(int, med)) | set(map(int, low)))

    updated_variance = dict(interval_variance)
    ema_sources = {}
    
    for label in majority_labels:
        pred_var = previous_variance.get(label, interval_variance[label])
        updated_variance[label] = max((1 - alpha) * pred_var + alpha * prediciton_variance[label], variance_floor)

    for label in tail_labels:
        #find nearest k majs
        nearest_maj = sorted(majority_labels,key=lambda source_label: (abs(source_label - label), source_label))[:1]

        v_ema = sum(interval_variance[source] for source in nearest_maj) 

        previous = previous_variance.get(label, interval_variance.get(label, v_ema))

        updated_variance[label] = max((1.0 - alpha) * previous + alpha * v_ema, variance_floor)

        ema_sources[label] = nearest_maj

    return updated_variance, ema_sources


def build_label_variance_lookup(label_variance, target_device):

    fallback = float(np.mean(list(label_variance.values())))
    lookup = torch.full(
        (max(label_variance) + 1,),
        fallback,
        dtype=torch.float32,
        device=target_device
    )
    for label, variance in label_variance.items():
        lookup[int(label)] = float(variance)
    return lookup


def compute_nll_ema_variance(
        args, model, data_loader, prediction_variance, maj,
        med, low, previous_ema_variance=None):
    """Refresh majority-guided temporal EMA variance for one NLL epoch."""
    interval_variance, label_count = infer_per_label_prediction_variance(model, data_loader)

    updated_variance, ema_sources = using_ema_replace_low_and_med_and_maj(
        interval_variance, prediction_variance, previous_ema_variance, 
        maj, med, low,
        alpha=args.ema_variance_alpha,
        variance_floor=args.variance_floor)

    lookup = build_label_variance_lookup(updated_variance, device)

    return {
        'lookup': lookup,
        'interval_variance': interval_variance,
        'prediction_variance': prediction_variance,
        'label_count': label_count,
        'ema_variance': updated_variance,
        'ema_source_labels': ema_sources,
    }


def apply_per_label_ema_variance(labels, ema_lookup, target_dtype):
    """Read constant per-label temporal EMA variance for every sample."""
    label_indices = labels.long()
    if label_indices.numel() > 0:
        min_label = int(label_indices.min().item())
        max_label = int(label_indices.max().item())
        if min_label < 0 or max_label >= ema_lookup.numel():
            raise IndexError(
                f'Label range [{min_label}, {max_label}] is outside '
                f'the EMA lookup of length {ema_lookup.numel()}.'
            )
    return ema_lookup[label_indices].to(target_dtype).detach()


def compute_interval_coverage(lower, upper, label, maj, med, low, device):
    covered = ((label >= lower) & (label <= upper)).to(torch.float)

    def group_coverage(group_labels):
        group_tensor = torch.as_tensor(group_labels, device=device)
        group_indices = torch.nonzero(torch.isin(label, group_tensor), as_tuple=False)
        if group_indices.numel() == 0:
            return float('nan')
        return covered[group_indices[:, 0]].squeeze(-1).mean().item()

    maj_cov = group_coverage(maj)
    med_cov = group_coverage(med)
    low_cov = group_coverage(low)
    total_cov = covered.squeeze(-1).mean().item()
    return maj_cov, med_cov, low_cov, total_cov


def evaluate_interval(model, data_loader, maj, med, low):
    """Evaluate one fixed model over a complete loader without updating it."""
    model.eval()

    lower_list = []
    upper_list = []
    prediction_list = []
    label_list = []

    with torch.no_grad():
        for x, y, _ in data_loader:
            x = x.to(device)
            y = y.to(device)

            prediction, lower, upper, _ = model(
                x, return_intervals=True
            )
            prediction_list.append(prediction)
            lower_list.append(lower)
            upper_list.append(upper)
            label_list.append(y)

    if not label_list:
        raise ValueError('Cannot evaluate interval metrics on an empty loader.')

    lowers = torch.cat(lower_list, dim=0)
    uppers = torch.cat(upper_list, dim=0)
    predictions = torch.cat(prediction_list, dim=0)
    labels = torch.cat(label_list, dim=0)

    coverage_maj, coverage_med, coverage_low, coverage_total = \
        compute_interval_coverage(
            lowers, uppers, labels, maj, med, low, device
        )

    return {
        'coverage_maj': coverage_maj,
        'coverage_med': coverage_med,
        'coverage_low': coverage_low,
        'coverage_total': coverage_total,
        'interval_width': torch.abs(uppers - lowers).mean().item(),
        'crossing_rate': (lowers > uppers).float().mean().item(),
        'point_mae': torch.abs(predictions - labels).mean().item(),
    }


def switch_low_variance_to_mse(args, point_loss, mse_component, var_component, y_pred, interval, y):
    threshold = args.variance_mse_threshold
    if args.MSE or args.MAE or threshold is None:
        return point_loss, mse_component, var_component

    mse_loss = (y_pred - y) ** 2
    low_variance_mask = interval < threshold
    point_loss = torch.where(low_variance_mask, mse_loss, point_loss)
    mse_component = torch.where(low_variance_mask, mse_loss, mse_component)
    var_component = torch.where(low_variance_mask, torch.zeros_like(var_component), var_component)
    return point_loss, mse_component, var_component

################################
###########train part###########
################################

@torch.no_grad()
def evaluate_point_mae(model, data_loader):
    """Compute ordinary, unweighted point-prediction MAE."""
    model.eval()
    absolute_error_sum = 0.0
    sample_count = 0

    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)

        z = model.extract_features(x)
        y_pred = model.linear(z)

        absolute_error_sum += torch.abs(y_pred - y).sum().item()
        sample_count += y.numel()

    if sample_count == 0:
        raise ValueError('Cannot evaluate MAE on an empty loader.')

    return absolute_error_sum / sample_count


#stage 1
def warm_up_using_mse(args, model, train_loader, opt_extractor, opt_regressor):
    model.train()
    model.interval_lower.eval()
    model.interval_upper.eval()

    mse_history = []
    
    for x, y, w in train_loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        z = model.extract_features(x)
        y_pred = model.linear(z)

        mse = (y_pred - y) ** 2
        mse_loss = reduce_batch_loss(mse, w, args.smooth)

        opt_extractor.zero_grad()
        opt_regressor.zero_grad()
        mse_loss.backward()
        opt_extractor.step()
        opt_regressor.step()

        mse_history.append(mse_loss.item())

    return float(np.mean(mse_history))

#stage 2
def train_quantile_by_cal(args, model, cal_loader, opt_cp_lower, opt_cp_upper):
    model.eval()
    model.interval_lower.train()
    model.interval_upper.train()

    lower_loss_history = []
    upper_loss_history = []

    for x_cal, y_cal, _ in cal_loader:
        x_cal = x_cal.to(device)
        y_cal = y_cal.to(device)

        with torch.no_grad():
            z_cal = model.extract_features(x_cal)

        lower = model.interval_lower(z_cal)
        upper = model.interval_upper(z_cal)

        loss_lower, loss_upper = cqr_pinball(y_cal, upper, lower, lamb=args.lamb)
        
        quantile_loss = loss_lower + loss_upper

        opt_cp_lower.zero_grad()
        opt_cp_upper.zero_grad()

        quantile_loss.backward()

        opt_cp_lower.step()
        opt_cp_upper.step()

        lower_loss_history.append(loss_lower.item())
        upper_loss_history.append(loss_upper.item())
    
    return (float(np.mean(lower_loss_history)), float(np.mean(upper_loss_history)))

#stage 3
def train_one_epoch_using_nll(
        args, model, train_loader, opt_extractor, opt_regressor,
        maj, med, low, ema_variance_lookup=None):
    model.train()
    model.interval_upper.eval()
    model.interval_lower.eval()

    nll_history = []
    mse_history = []
    var_history = []

    for x, y, w in train_loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        if ema_variance_lookup is not None:
            # The default EMA path needs only the point prediction. Avoid
            # building an unused interval graph in Stage 3.
            y_pred = model(x)
            variance = apply_per_label_ema_variance(
                y,
                ema_variance_lookup,
                y_pred.dtype
            )
        else:
            y_pred, lower, upper, _ = model(
                x, return_intervals=True
            )
            variance = ((upper - lower) / 2.5632) ** 2
            # The interval heads provide a fixed uncertainty target in
            # Stage 3. Do not backpropagate through variance into the
            # backbone, even when per-sample variance is requested.
            variance = torch.clamp(
                variance, min=args.variance_floor
            ).detach()

        nll, mse_part, var_part = beta_nll_components(y_pred, variance, y, beta=args.beta)
        nll_loss = reduce_batch_loss(nll, w, args.smooth)
        
        mse_component = reduce_batch_loss(mse_part, w, args.smooth)
        var_component = reduce_batch_loss(var_part, w, args.smooth)
        
        opt_extractor.zero_grad()
        opt_regressor.zero_grad()
        nll_loss.backward()
        opt_extractor.step()
        opt_regressor.step()

        nll_history.append(nll_loss.item())
        mse_history.append(mse_component.item())
        var_history.append(var_component.item())

    return {
        'nll': float(np.mean(nll_history)),
        'nll_mse': float(np.mean(mse_history)),
        'nll_var': float(np.mean(var_history))}

        
#fuse all three stages
def train(
        args, model, train_loader, train_eval_loader, cal_loader,
        test_loader, train_labels):
    #write log for nll
    log_dir = get_checkpoint_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'nll_beta_{args.beta}_{args.inv_method}.txt')
    with open(log_path, 'w') as file:
        file.write(
        'epoch '
        'nll nll_mse nll_var cal_mae '
        #'uncer_maj uncer_med uncer_low uncer_total '
        'coverage_maj coverage_med coverage_low coverage_total '
        #'uncer_pred_maj uncer_pred_med '
        #'uncer_pred_low uncer_pred_total '
        'interval_width\n')

    maj, med, low = shot_count(train_labels)

    history = {
            'warmup_mse': [],
            'warmup_cal_mae': [],
            'best_warmup_epoch': None,
            'best_warmup_cal_mae': None,
            'stage1_test_mae': None,
            'quantile_lower': [],
            'quantile_upper': [],
            'stage2_train_metrics': None,
            'stage2_cal_metrics': None,
            'nll_metrics': [],
            'best_stage3_epoch': None,
            'best_stage3_cal_mae': None,
            'best_stage3_checkpoint': None,
            'nll_ema_variance': {
                'enabled': not args.disable_ema_variance,
                'alpha': args.ema_variance_alpha,
                'epoch_decay': args.ema_epoch_decay,
                'variance_floor': args.variance_floor,
                'epochs':[]
            },
            'final_quantile_lower': [],
            'final_quantile_upper': []
               }

    resume_stage2 = bool(args.resume_stage2_ckpt)
    res_pred_variance = None
    previous_ema_variance = None
    stage2_train_metrics = None
    stage2_cal_metrics = None

######################stage 1 warmup round###########################
    # A Stage 2 checkpoint already contains all four model modules, so loading
    # it bypasses both warmup and quantile training.
    if resume_stage2:
        stage2_path = args.resume_stage2_ckpt

        if not os.path.isfile(stage2_path):
            raise FileNotFoundError(
                f'Stage 2 checkpoint not found: {stage2_path}'
            )

        checkpoint = torch.load(stage2_path, map_location=device)
        load_checkpoint_model(model, checkpoint, stage2_path)
        history['loaded_stage2_checkpoint'] = stage2_path

        saved_history = checkpoint.get('history', {})
        for key in (
                'warmup_mse', 'warmup_cal_mae',
                'best_warmup_epoch', 'best_warmup_cal_mae',
                'quantile_lower', 'quantile_upper',
                'stage2_train_metrics', 'stage2_cal_metrics'):
            if key in saved_history:
                history[key] = saved_history[key]

        res_pred_variance = checkpoint.get('prediction_variance')
        if res_pred_variance is not None:
            res_pred_variance = {
                int(label): max(float(variance), args.variance_floor)
                for label, variance in res_pred_variance.items()
            }

        previous_ema_variance = checkpoint.get('initial_ema_variance')
        if previous_ema_variance is not None:
            previous_ema_variance = {
                int(label): max(float(variance), args.variance_floor)
                for label, variance in previous_ema_variance.items()
            }

        stage2_train_metrics = checkpoint.get(
            'stage2_train_metrics',
            saved_history.get('stage2_train_metrics')
        )
        stage2_cal_metrics = checkpoint.get(
            'stage2_cal_metrics',
            saved_history.get('stage2_cal_metrics')
        )

        print(
            f'Loaded Stage 2 checkpoint from {stage2_path}, '
            f'quantile epoch={checkpoint.get("epoch", "unknown")}; '
            'skipping Stage 1 and Stage 2 training'
        )

    # Otherwise, load a warmup checkpoint or train Stage 1 from scratch.
    elif args.resume_warmup_ckpt:
        warmup_path = args.resume_warmup_ckpt

        if not os.path.isfile(warmup_path):
            raise FileNotFoundError(f'Warmup checkpoint not found: {warmup_path}')

        checkpoint = torch.load(warmup_path, map_location=device)
        load_checkpoint_model(model, checkpoint, warmup_path)
        history['loaded_warmup_checkpoint'] = warmup_path
        saved_warmup_history = checkpoint.get('history', {})
        for key in (
                'warmup_mse', 'warmup_cal_mae',
                'best_warmup_epoch', 'best_warmup_cal_mae'):
            if key in saved_warmup_history:
                history[key] = saved_warmup_history[key]

        print(
            f'Loaded warmup checkpoint from {warmup_path}, '
            f'saved epoch={checkpoint.get("epoch", "unknown")}, '
            f'cal-MAE={checkpoint.get("cal_mae", "unknown")}')

    else:
        print('stage 1: warm up')

        set_backbone_trainable(model, True)
        set_trainable(model.linear, True)
        set_trainable(model.interval_lower, False)
        set_trainable(model.interval_upper, False)

        opt_extractor = optim.Adam(
            backbone_parameters(model),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay
        )
        opt_regressor = optim.Adam(
            model.linear.parameters(),
            lr=args.warmup_lr,
            weight_decay=args.weight_decay
        )

        warmup_path = get_warmup_checkpoint_path(args)
        warmup_dir = os.path.dirname(warmup_path)
        if warmup_dir:
            os.makedirs(warmup_dir, exist_ok=True)

        best_cal_mae = float('inf')

        for epoch in tqdm(range(args.warmup_epoch), desc='warm up'):
            adjust_stage1_learning_rate(opt_extractor, epoch, args)
            adjust_stage1_learning_rate(opt_regressor, epoch, args)

            mse_loss = warm_up_using_mse(args, model, train_loader, opt_extractor, opt_regressor)
            cal_mae = evaluate_point_mae(model, cal_loader)

            if not np.isfinite(cal_mae):
                raise ValueError(
                    f'Non-finite Stage 1 cal MAE at epoch {epoch + 1}.'
                )

            history['warmup_mse'].append(mse_loss)
            history['warmup_cal_mae'].append(cal_mae)

            is_best = cal_mae < best_cal_mae
            if is_best:
                best_cal_mae = cal_mae
                history['best_warmup_epoch'] = epoch + 1
                history['best_warmup_cal_mae'] = best_cal_mae

                torch.save(
                    {
                        'stage': 'stage1_best_cal_mae',
                        'epoch': epoch,
                        'cal_mae': best_cal_mae,
                        'model_state_dict': model.state_dict(),
                        'opt_extractor_state_dict':
                            opt_extractor.state_dict(),
                        'opt_regressor_state_dict':
                            opt_regressor.state_dict(),
                        'history': history,
                    },
                    warmup_path
                )

            print(
                f'Warmup epoch {epoch + 1}/'
                f'{args.warmup_epoch}, '
                f'MSE={mse_loss:.6f}, '
                f'cal-MAE={cal_mae:.6f}'
                f'{" [best]" if is_best else ""}'
            )

        best_warmup_checkpoint = torch.load(
            warmup_path, map_location=device
        )
        load_checkpoint_model(
            model,
            best_warmup_checkpoint,
            warmup_path,
        )
        print(
            f'Restored best Stage 1 model from epoch '
            f'{history["best_warmup_epoch"]}, '
            f'cal-MAE={best_cal_mae:.6f}: {warmup_path}'
        )

    if not resume_stage2:
        if args.skip_stage1_test:
            print('Skipped Stage 1 checkpoint test (--skip_stage1_test)')
        else:
            print(
                'Evaluating the loaded/trained Stage 1 checkpoint on the '
                'test set before Stage 2...'
            )
            stage1_test_mae = test(
                model,
                test_loader,
                train_labels,
                args
            )
            history['stage1_test_mae'] = stage1_test_mae
            print(
                f'Stage 1 checkpoint Test MAE: '
                f'{stage1_test_mae:.6f}'
            )

    if res_pred_variance is None:
        res_pred_variance = get_prediction_variance(
            model, train_eval_loader
        )

    ####################stage 2 quantile training using pinball loss####################
    if not resume_stage2:
        print('stage 2: quantile training')
        set_backbone_trainable(model, False)
        set_trainable(model.linear, False)
        set_trainable(model.interval_lower, True)
        set_trainable(model.interval_upper, True)

        stage2_lr = args.stage2_lr

        opt_cp_lower = optim.Adam(
    model.interval_lower.parameters(),
    lr=stage2_lr,
    weight_decay=5e-4,
)
    opt_cp_upper = optim.Adam(
    model.interval_upper.parameters(),
    lr=stage2_lr,
    weight_decay=5e-4,
)

    quantile_progress = tqdm(
    range(args.quantile_epoch),
    desc='quantile training',
    dynamic_ncols=True,
)

    for epoch in quantile_progress:
    # epoch 0~4: 1e-4
    # epoch 5~19: 5e-5
        if epoch < args.stage2_lr_switch_epoch:
            current_lr = args.stage2_lr
        else:
            current_lr = args.stage2_lr_after

        for optimizer in (opt_cp_lower, opt_cp_upper):
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        lower_loss, upper_loss = train_quantile_by_cal(
        args,
        model,
        cal_loader,
        opt_cp_lower,
        opt_cp_upper,
    )

        history['quantile_lower'].append(lower_loss)
        history['quantile_upper'].append(upper_loss)

        quantile_progress.set_postfix(
        lr=f'{current_lr:.1e}',
        lower=f'{lower_loss:.6f}',
        upper=f'{upper_loss:.6f}',
    )

        tqdm.write(
        f'Quantile epoch {epoch + 1}/{args.quantile_epoch}, '
        f'lr={current_lr:.1e}, '
        f'lower={lower_loss:.6f}, '
        f'upper={upper_loss:.6f}'
    )

    # Old Stage 2 checkpoints may not contain metrics; evaluate only when
    # needed. New checkpoints restore them without an extra full evaluation.
    if stage2_train_metrics is None:
        stage2_train_metrics = evaluate_interval(
            model, train_eval_loader, maj, med, low
        )
    if stage2_cal_metrics is None:
        stage2_cal_metrics = evaluate_interval(
            model, cal_loader, maj, med, low
        )

    history['stage2_train_metrics'] = stage2_train_metrics
    history['stage2_cal_metrics'] = stage2_cal_metrics

    print(
        'Stage 2 train-set metrics: '
        f'many={stage2_train_metrics["coverage_maj"]:.4f}, '
        f'median={stage2_train_metrics["coverage_med"]:.4f}, '
        f'low={stage2_train_metrics["coverage_low"]:.4f}, '
        f'total={stage2_train_metrics["coverage_total"]:.4f}, '
        f'width={stage2_train_metrics["interval_width"]:.4f}, '
        f'crossing={stage2_train_metrics["crossing_rate"]:.4f}'
    )
    print(
        'Stage 2 calibration-set metrics: '
        f'many={stage2_cal_metrics["coverage_maj"]:.4f}, '
        f'median={stage2_cal_metrics["coverage_med"]:.4f}, '
        f'low={stage2_cal_metrics["coverage_low"]:.4f}, '
        f'total={stage2_cal_metrics["coverage_total"]:.4f}, '
        f'width={stage2_cal_metrics["interval_width"]:.4f}, '
        f'crossing={stage2_cal_metrics["crossing_rate"]:.4f}'
    )

    stage2_metrics_path = os.path.join(log_dir, 'stage2_coverage_metrics.csv')
    pd.DataFrame([
        {'split': 'train', **stage2_train_metrics},
        {'split': 'cal', **stage2_cal_metrics},
    ]).to_csv(stage2_metrics_path, index=False)
    print(f'Saved Stage 2 coverage metrics to {stage2_metrics_path}')

    # V0: per-label interval variance inferred from the trained Stage 2
    # quantile model. Store plain Python values so it is outside autograd.
    if previous_ema_variance is None:
        with torch.no_grad():
            previous_ema_variance, _ = (
                infer_per_label_prediction_variance(
                    model, train_eval_loader
                )
            )
        previous_ema_variance = {
            int(label): max(float(variance), args.variance_floor)
            for label, variance in previous_ema_variance.items()
        }

    if not resume_stage2 and not args.skip_stage2_checkpoint_save:
        stage2_path = get_stage2_checkpoint_path(args)
        stage2_dir = os.path.dirname(stage2_path)
        if stage2_dir:
            os.makedirs(stage2_dir, exist_ok=True)

        history['saved_stage2_checkpoint'] = stage2_path
        torch.save(
            {
                'stage': 'stage2_quantile',
                'epoch': args.quantile_epoch - 1,
                'model_state_dict': model.state_dict(),
                'opt_cp_lower_state_dict': opt_cp_lower.state_dict(),
                'opt_cp_upper_state_dict': opt_cp_upper.state_dict(),
                'prediction_variance': res_pred_variance,
                'initial_ema_variance': previous_ema_variance,
                'stage2_train_metrics': stage2_train_metrics,
                'stage2_cal_metrics': stage2_cal_metrics,
                'history': history,
                'config': {
                    'dataset': args.dataset,
                    'seed': args.seed,
                    'lamb': args.lamb,
                    'quantile_epoch': args.quantile_epoch,
                },
            },
            stage2_path
        )
        print(f'Saved Stage 2 checkpoint to {stage2_path}')
    elif not resume_stage2:
        print('Skipped Stage 2 checkpoint save')
    
    ###########################stage 3 training backbone using nll#############################
    print('stage 3: backbone training')
    set_trainable(model.interval_lower, False)
    set_trainable(model.interval_upper, False)
    set_backbone_trainable(model, True)
    set_trainable(model.linear, True)

    stage3_extractor_lr = (
        args.lr
        if args.stage3_extractor_lr is None
        else args.stage3_extractor_lr
    )
    stage3_regressor_lr = (
        args.lr
        if args.stage3_regressor_lr is None
        else args.stage3_regressor_lr
    )
    print(
        'Stage 3 optimizer: '
        f'extractor_lr={stage3_extractor_lr}, '
        f'regressor_lr={stage3_regressor_lr}, '
        f'weight_decay={args.stage3_weight_decay}'
    )
    opt_extractor = optim.Adam(
        backbone_parameters(model),
        lr=stage3_extractor_lr,
        weight_decay=args.stage3_weight_decay
    )
    opt_regressor = optim.Adam(
        model.linear.parameters(),
        lr=stage3_regressor_lr,
        weight_decay=args.stage3_weight_decay
    )

    best_cal_mae = float('inf')
    best_stage3_epoch = None
    best_stage3_path = os.path.join(
        log_dir, 'best_stage3_by_cal_mae.pth.tar'
    )

    for epoch in tqdm(range(args.nll_epoch), desc='nll training'):
        if args.disable_ema_variance:
            # train_one_epoch_using_nll() will use detached, per-sample
            # interval variance instead of the constant per-label lookup.
            ema_variance_lookup = None
        else:
            ema_state = compute_nll_ema_variance(
                args,
                model,
                train_eval_loader,
                res_pred_variance,
                maj,
                med,
                low,
                previous_ema_variance,
            )
            ema_variance_lookup = ema_state['lookup']
            previous_ema_variance = dict(ema_state['ema_variance'])

            epoch_ema_state = {
                key: value
                for key, value in ema_state.items()
                if key != 'lookup'
            }
            epoch_ema_state['epoch'] = epoch + 1
            history['nll_ema_variance']['epochs'].append(
                epoch_ema_state
            )

        metrics = train_one_epoch_using_nll(args, model, train_loader,
                                            opt_extractor, opt_regressor,
                                            maj, med, low,
                                            ema_variance_lookup)

        # One no-gradient cal-set pass provides both interval metrics and the
        # ordinary (unweighted) point-prediction MAE used for model selection.
        cal_metrics = evaluate_interval(
            model, test_loader, maj, med, low
        )
        metrics['cal_mae'] = cal_metrics.pop('point_mae')
        metrics.update(cal_metrics)

        history['nll_metrics'].append(metrics)

        is_best = metrics['cal_mae'] < best_cal_mae
        if is_best:
            best_cal_mae = metrics['cal_mae']
            best_stage3_epoch = epoch + 1
            history['best_stage3_epoch'] = best_stage3_epoch
            history['best_stage3_cal_mae'] = best_cal_mae
            history['best_stage3_checkpoint'] = best_stage3_path

            best_stage3_checkpoint = {
                'stage': 'stage3_best_cal_mae',
                'epoch': epoch,
                'cal_mae': best_cal_mae,
                'model_state_dict': model.state_dict(),
            }
            if not args.lightweight_stage3_checkpoint:
                best_stage3_checkpoint.update({
                    'opt_extractor_state_dict':
                        opt_extractor.state_dict(),
                    'opt_regressor_state_dict':
                        opt_regressor.state_dict(),
                    'previous_ema_variance':
                        previous_ema_variance,
                    'history': history,
                })

            torch.save(
                best_stage3_checkpoint,
                best_stage3_path
            )

        print(f'NLL epoch {epoch + 1}/'
              f'{args.nll_epoch}, '
              f'NLL={metrics["nll"]:.6f}, '
              f'MSE-part={metrics["nll_mse"]:.6f}, '
              f'VAR-part={metrics["nll_var"]:.6f}, '
              f'cal-MAE={metrics["cal_mae"]:.6f}, '
              f'coverage={metrics["coverage_total"]:.4f}'
              f'{" [best]" if is_best else ""}')
        
        values = [epoch + 1,

                metrics['nll'],
                metrics['nll_mse'],
                metrics['nll_var'],
                metrics['cal_mae'],

                metrics['coverage_maj'],
                metrics['coverage_med'],
                metrics['coverage_low'],
                metrics['coverage_total'],

                metrics['interval_width']
                ]


        with open(log_path, 'a') as file:
            file.write(' '.join(map(str, values)) + '\n')

    if best_stage3_epoch is not None:
        best_checkpoint = torch.load(
            best_stage3_path, map_location=device
        )
        load_checkpoint_model(
            model,
            best_checkpoint,
            best_stage3_path,
        )
        print(
            f'Restored best Stage 3 model from epoch '
            f'{best_stage3_epoch}, cal-MAE={best_cal_mae:.6f}: '
            f'{best_stage3_path}'
        )

    return model, history

def test(model, test_loader, train_labels, args):
    model.eval()
    #
    mse_pred = AverageMeter()
    mae_pred = AverageMeter()
    # gmean
    criterion_gmean_pred = nn.L1Loss(reduction='none')
    gmean_loss_all_pred = []
    #
    pred, labels = [], []
    #
    pred_list, label_list, z_list = [], [], []
    #
    with torch.no_grad():
        for idx, (x, y, _) in enumerate(test_loader):
            bsz = x.shape[0]
            x, y= x.to(device), y.to(device)
            #
            labels.extend(y.data.cpu().numpy())
            #
            y_pred, lower, upper, z = model(
                x, return_intervals=True
            )
            interval = torch.clamp(torch.abs(upper - lower), min=1e-6)
            #
            #print(f' y shape is  {y_output.shape}')
            #
            mae_y = torch.mean(torch.abs(y_pred- y))
            mse_y_pred = F.mse_loss(y_pred, y)
            #
            pred.extend(y_pred.data.cpu().numpy())
            # gmean
            loss_all_pred = criterion_gmean_pred(y_pred, y)
            gmean_loss_all_pred.extend(loss_all_pred.cpu().numpy())
            #
            mse_pred.update(mse_y_pred.item(), bsz)
            #
            mae_pred.update(mae_y.item(), bsz)
            #
            label_list.append(y)
            pred_list.append(y_pred)
            z_list.append(z)
        label_, pred_, z_  = torch.cat(label_list, 0), torch.cat(pred_list, 0), torch.cat(z_list, 0)
        #
        # gmean
        gmean_pred = gmean(np.hstack(gmean_loss_all_pred), axis=None).astype(float)
        shot_pred = shot_metric(pred, labels, train_labels)
    print(f' MSE is {mse_pred.avg}')
    #
    #mae_dict = per_label_mae(pred_, label_)
    #mae_dict = per_label_frobenius_norm(z_, label_)
    #var_per_label = per_label_var(pred, labels)
    #mae_per_label = per_label_mae(pred_, label_)
    #
    #
    return mae_pred.avg#, shot_pred, gmean_pred, var_per_label, mae_per_label
        # np.hstack(group), np.hstack(group_pred) #newly added

######################
# write log for the test
#####################
def write_log(store_name, mae_pred, shot_pred, gmean_pred):
    with open(store_name, 'a+') as f:
        f.write('=---------------------------------------------------------------------=\n')
        f.write(f' store name is {store_name}')
        #
        f.write(' Prediction ALL MAE {} Many: MAE {} Median: MAE {} Low: MAE {}'.format(mae_pred, shot_pred['many']['l1'],
                                                                            shot_pred['median']['l1'], shot_pred['low']['l1']) + "\n")
        #
        f.write(' G-mean Prediction {}, Many : G-Mean {}, Median : G-Mean {}, Low : G-Mean {}'.format(gmean_pred, shot_pred['many']['gmean'],
                                                                        shot_pred['median']['gmean'], shot_pred['low']['gmean'])+ "\n")     
        f.write('---------------------------------------------------------------------\n')
        f.close()
#############################
 
def main():
    args = parser.parse_args()

    if args.resume_stage2_ckpt and args.resume_warmup_ckpt:
        raise ValueError(
            'Use only one of --resume_stage2_ckpt and '
            '--resume_warmup_ckpt.'
        )
    if args.warmup_epoch <= 0:
        raise ValueError('--warmup_epoch must be positive.')
    if args.variance_mse_threshold is not None and args.variance_mse_threshold < 0:
        raise ValueError('--variance_mse_threshold must be non-negative.')
    if not 0 < args.ema_variance_alpha <= 1:
        raise ValueError('--ema_variance_alpha must be in (0, 1].')
    if not 0 <= args.ema_epoch_decay < 1:
        raise ValueError('--ema_epoch_decay must be in [0, 1).')
    if args.variance_floor <= 0:
        raise ValueError('--variance_floor must be positive.')
    if args.stage2_lr is not None and args.stage2_lr <= 0:
        raise ValueError('--stage2_lr must be positive.')
    if args.stage3_extractor_lr is not None and args.stage3_extractor_lr <= 0:
        raise ValueError('--stage3_extractor_lr must be positive.')
    if args.stage3_regressor_lr is not None and args.stage3_regressor_lr <= 0:
        raise ValueError('--stage3_regressor_lr must be positive.')
    if args.stage3_weight_decay < 0:
        raise ValueError('--stage3_weight_decay must be non-negative.')
    
    setup_seed(args.seed)
    #
    train_loader, train_eval_loader, val_loader, test_loader, train_labels = get_data_loader(args)
    #
    loss_mse = nn.MSELoss()
    #
    maj, med, low = shot_count(train_labels)
    #
    # This must match the enhanced ResNet50 used by vanilla train.py.
    # Vanilla checkpoints produced without --fds use this configuration.
    model = resnet50(
        fds=False,
        bucket_num=100,
        bucket_start=3,
        start_update=0,
        start_smooth=1,
        kernel='gaussian',
        ks=9,
        sigma=1,
        momentum=0.9,
    ).to(device)

    model, history = train(
        args, model, train_loader, train_eval_loader, val_loader,
        test_loader, train_labels
    )
    mae_pred = test(model, test_loader, train_labels, args)

    print(f'Final test MAE: {mae_pred:.6f}')

    if args.skip_final_model_save:
        print('Skipped final model save')
    else:
        final_path = os.path.join(
            get_checkpoint_dir(args),
            'final_model.pth.tar'
        )
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'history': history,
            },
            final_path
        )
        print(f'Saved final model to {final_path}')
if __name__ == '__main__':
    main()
