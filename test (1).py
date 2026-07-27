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
parser.add_argument('--reweight', type=str, default='none',  choices=['none','inv', 'sqrt_inverse'],
                    help='weight : inv or sqrt_inv')
parser.add_argument('--smooth', default='none', choices=['lds', 'none'], help='use LDS or not')
parser.add_argument('--inv_method', default='cqr_pinball', choices=['split_cp', 'cqr_pinball', 'cqr_coverage'], help='use which method to train interval module')
#
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    mae_dict = per_label_mae(pred_, label_)
    mae_dict = per_label_frobenius_norm(z_, label_)
    var_per_label = per_label_var(pred, labels)
    mae_per_label = per_label_mae(pred_, label_)
    #
    #
    return mae_pred.avg, shot_pred, gmean_pred, var_per_label, mae_per_label
        # np.hstack(group), np.hstack(group_pred) #newly added
    
def write_log(store_name, mae_pred, shot_pred, gmean_pred):
    with open(store_name, 'w') as f:
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
        
args = parser.parse_args()
train_loader, train_eval_loader, val_loader, test_loader, train_labels = get_data_loader(args)
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

path = './checkpoint/ema_resnet50_q80_nll15/best_stage3_by_cal_mae.pth.tar'
checkpoint = torch.load(path, map_location=device)
load_checkpoint_model(model, checkpoint, path)

mae, shot, gmean, var, mae_label = test(model, test_loader, train_labels, args)

store_name = 'result.txt'

write_log(store_name, mae, shot, gmean)

