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
from network import *
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
parser.add_argument('--lamb', default=0.8, type=float,  help='lamb for coverage')
parser.add_argument('--weight', default=0.1, type=float,  help='weight for interval_loss in total loss')
parser.add_argument('--alpha', default=0.1, type=float,  help='miscoverage level for conformal calibration')
parser.add_argument('--cp_mode', type=str, default='hybrid', choices=['cqr', 'split', 'hybrid'])
parser.add_argument('--warmup_ckpt_path', type=str, default='',
                    help='path to save the final warmup checkpoint; defaults to <store_root>/<store_name>/warmup_final.pth.tar')
parser.add_argument('--resume_warmup_ckpt', type=str, default='',
                    help='path to a saved warmup checkpoint to resume from')
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

def get_data_loader(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size,
                        split='train', reweight=args.reweight, group_num=args.groups, smooth=args.smooth)   
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val,
                        img_size=args.img_size, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                        img_size=args.img_size, split='test', group_num=args.groups)
    #
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_labels


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

def reduce_batch_loss(loss, weight, smooth_mode):
    if smooth_mode == 'lds':
        return (loss * weight.expand_as(loss)).sum() / weight.sum().clamp_min(1e-12)
    return loss.mean()


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

#stage 1
def warm_up_using_mse(args, model, train_loader, opt_extractor, opt_regressor):
    model.model_extractor.train()
    model.pred_head.train()
    model.interval_lower.eval()
    model.interval_upper.eval()

    mse_history = []
    
    for x, y, w in train_loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)
        z = model.model_extractor(x)
        z = model.Flatten(z)
        y_pred = model.pred_head(z)

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
    model.model_extractor.eval()
    model.pred_head.eval()

    model.interval_lower.train()
    model.interval_upper.train()

    lower_loss_history = []
    upper_loss_history = []

    for x_cal, y_cal, _ in cal_loader:
        x_cal = x_cal.to(device)
        y_cal = y_cal.to(device)

        with torch.no_grad():
            z_cal = model.model_extractor(x_cal)
            z_cal = model.Flatten(z_cal)

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
def train_one_epoch_using_nll(args, model, train_loader, opt_extractor, opt_regressor, maj, med, low):
    model.model_extractor.train()
    model.pred_head.train()

    model.interval_upper.eval()
    model.interval_lower.eval()

    nll_history = []
    mse_history = []
    var_history = []

    interval_var_list = []
    lower_list = []
    upper_list = []
    label_list = []
    pred_list = []

    for x, y, w in train_loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        y_pred, lower, upper, _= model(x)
        
        variance = ((upper - lower) / 2.5632) ** 2
        variance = torch.clamp(variance, min=1e-6)

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
        interval_var_list.append(variance.detach())
        lower_list.append(lower.detach())
        upper_list.append(upper.detach())
        label_list.append(y.detach())
        pred_list.append(y_pred.detach())
    
    interval_vars = torch.cat(interval_var_list, dim=0)
    lowers = torch.cat(lower_list, dim=0)
    uppers = torch.cat(upper_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    preds = torch.cat(pred_list, dim=0)

    interval_width = torch.abs(uppers - lowers).mean().item()

    #origin uncertainty output
    uncer_maj, uncer_med, uncer_low, uncer_total = uncertainty_accumulation(interval_vars, labels, maj, med, low, device)

    #coverage output
    coverage_maj, coverage_med, coverage_low, coverage_total = compute_interval_coverage(lowers,uppers,labels, maj, med, low, device)

    #prediction uncertainty output
    uncer_pred_maj, uncer_pred_med, uncer_pred_low, uncer_pred_total = label_uncertainty_accumulation(preds, labels, maj, med, low, device)

    return {
        'nll': float(np.mean(nll_history)),
        'nll_mse': float(np.mean(mse_history)),
        'nll_var': float(np.mean(var_history)),

        'uncer_maj': uncer_maj,
        'uncer_med': uncer_med,
        'uncer_low': uncer_low,
        'uncer_total': uncer_total,

        'coverage_maj': coverage_maj,
        'coverage_med': coverage_med,
        'coverage_low': coverage_low,
        'coverage_total': coverage_total,

        'uncer_pred_maj': uncer_pred_maj,
        'uncer_pred_med': uncer_pred_med,
        'uncer_pred_low': uncer_pred_low,
        'uncer_pred_total': uncer_pred_total,

        'interval_width': interval_width
    }

#fuse all three stages
def train(args, model, train_loader, cal_loader, train_labels):
    #write log for nll
    log_dir = get_checkpoint_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'nll_beta_{args.beta}_{args.inv_method}.txt')
    with open(log_path, 'w') as file:
        file.write(
        'epoch '
        'nll nll_mse nll_var '
        'uncer_maj uncer_med uncer_low uncer_total '
        'coverage_maj coverage_med coverage_low coverage_total '
        'uncer_pred_maj uncer_pred_med '
        'uncer_pred_low uncer_pred_total '
        'interval_width\n')

    maj, med, low = shot_count(train_labels)

    history = {
            'warmup_mse': [],
            'quantile_lower': [],
            'quantile_upper': [],
            'nll_metrics': [],
            'final_quantile_lower': [],
            'final_quantile_upper': []
               }
    
    #stage 1 warmup round
    # stage 1: warmup or load warmup checkpoint
    if args.resume_warmup_ckpt:
        warmup_path = args.resume_warmup_ckpt

        if not os.path.isfile(warmup_path):
            raise FileNotFoundError(
                f'Warmup checkpoint not found: {warmup_path}'
            )

        checkpoint = torch.load(
            warmup_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=True
        )

        history['loaded_warmup_checkpoint'] = warmup_path

        print(
            f'Loaded warmup checkpoint from {warmup_path}, '
            f'saved epoch={checkpoint.get("epoch", "unknown")}'
        )

    else:
        print('stage 1: warm up')

        set_trainable(model.model_extractor, True)
        set_trainable(model.pred_head, True)
        set_trainable(model.interval_lower, False)
        set_trainable(model.interval_upper, False)

        opt_extractor = optim.Adam(
            model.model_extractor.parameters(),
            lr=args.lr,
            weight_decay=5e-4
        )

        opt_regressor = optim.Adam(
            model.pred_head.parameters(),
            lr=args.lr,
            weight_decay=5e-4
        )

        for epoch in tqdm(
            range(args.warmup_epoch),
            desc='warm up'
        ):
            mse_loss = warm_up_using_mse(
                args,
                model,
                train_loader,
                opt_extractor,
                opt_regressor
            )

            history['warmup_mse'].append(mse_loss)

            print(
                f'Warmup epoch {epoch + 1}/'
                f'{args.warmup_epoch}, '
                f'MSE={mse_loss:.6f}'
            )

        warmup_path = get_warmup_checkpoint_path(args)
        warmup_dir = os.path.dirname(warmup_path)

        if warmup_dir:
            os.makedirs(warmup_dir, exist_ok=True)

        torch.save(
            {
                'epoch': args.warmup_epoch - 1,
                'model_state_dict': model.state_dict(),
                'opt_extractor_state_dict':
                    opt_extractor.state_dict(),
                'opt_regressor_state_dict':
                    opt_regressor.state_dict()
            },
            warmup_path
        )

        print(f'Saved warmup checkpoint to {warmup_path}')

    #stage 2 quantile training using pinball loss
    print('stage 2: quantile training')
    set_trainable(model.model_extractor, False)
    set_trainable(model.pred_head, False)
    set_trainable(model.interval_lower, True)
    set_trainable(model.interval_upper, True)

    opt_cp_lower = optim.Adam(model.interval_lower.parameters(), lr=args.lr, weight_decay=5e-4,)
    opt_cp_upper = optim.Adam(model.interval_upper.parameters(), lr=args.lr, weight_decay=5e-4,)

    for epoch in tqdm(range(args.quantile_epoch), desc='quantile training'):
        lower_loss, upper_loss = train_quantile_by_cal(args, model, cal_loader, opt_cp_lower, opt_cp_upper)

        history['quantile_lower'].append(lower_loss)
        history['quantile_upper'].append(upper_loss)

        print(f'Quantile epoch {epoch + 1}/'
              f'{args.quantile_epoch}, '
              f'lower={lower_loss:.6f}, '
              f'upper={upper_loss:.6f}')
    
    #stage 3 training backbone using nll
    print('stage 3: backbone training')
    set_trainable(model.interval_lower, False)
    set_trainable(model.interval_upper, False)
    set_trainable(model.model_extractor, True)
    set_trainable(model.pred_head, True)

    opt_extractor = optim.Adam(model.model_extractor.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_regressor = optim.Adam(model.pred_head.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in tqdm(range(args.nll_epoch), desc='NLL training'):
        metrics = train_one_epoch_using_nll(args,model, train_loader, opt_extractor, opt_regressor, maj, med, low)

        history['nll_metrics'].append(metrics)

        print(f'NLL epoch {epoch + 1}/'
              f'{args.nll_epoch}, '
              f'NLL={metrics["nll"]:.6f}, '
              f'MSE-part={metrics["nll_mse"]:.6f}, '
              f'VAR-part={metrics["nll_var"]:.6f}, '
              f'uncer={metrics["uncer_total"]:.6f}, '
              f'coverage={metrics["coverage_total"]:.4f}')
        
        values = [epoch + 1,

                metrics['nll'],
                metrics['nll_mse'],
                metrics['nll_var'],

                metrics['uncer_maj'],
                metrics['uncer_med'],
                metrics['uncer_low'],
                metrics['uncer_total'],

                metrics['coverage_maj'],
                metrics['coverage_med'],
                metrics['coverage_low'],
                metrics['coverage_total'],

                metrics['uncer_pred_maj'],
                metrics['uncer_pred_med'],
                metrics['uncer_pred_low'],
                metrics['uncer_pred_total'],
                metrics['interval_width']]

        with open(log_path, 'a') as file:
            file.write(' '.join(map(str, values)) + '\n')
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
            y_pred, lower, upper, z = model(x)
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

    if args.variance_mse_threshold is not None and args.variance_mse_threshold < 0:
        raise ValueError('--variance_mse_threshold must be non-negative.')
    
    setup_seed(args.seed)
    #
    train_loader, val_loader, test_loader,  train_labels = get_data_loader(args)
    #
    loss_mse = nn.MSELoss()
    #
    maj, med, low = shot_count(train_labels)
    #
    model = ResNet_conformal(args).to(device)

    model, history = train(args, model, train_loader, val_loader, train_labels)
    mae_pred = test(model, test_loader, train_labels, args)

    print(f'Final test MAE: {mae_pred:.6f}')

    final_path = os.path.join(
    get_checkpoint_dir(args),
    'final_model.pth.tar')

    torch.save(
    {'model_state_dict': model.state_dict(),
        'history': history,},final_path)
    print(f'Saved final model to {final_path}')
if __name__ == '__main__':
    main()
