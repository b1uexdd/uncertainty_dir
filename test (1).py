import time
import os
import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import gmean
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import *
import torch
from loss import *
from rnc_model import RnCIntervalRegressor
import torch.optim as optim
import time
from scipy.stats import gmean
from split_CP import coverage_loss, calibrate_qhat_from_batch, calibrate_qhat_splitCP, cqr_pinball, interval_minimization
import torch.nn.functional as F
import itertools
from dataset import AgeDB


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# training/optimization related
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--dataset', type=str, default='AgeDB',
                    choices=['imdb_wiki', 'AgeDB'], help='dataset name')
parser.add_argument('--data_folder', type=str,
                    default='/root/autodl-tmp/data', help='data directory')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
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

parser.add_argument('--stage2_lr', type=float, default=5e-3, help='Stage 2 learning rate for the first phase',)
parser.add_argument('--stage2_lr_after', type=float, default=5e-4, help='Stage 2 learning rate after switching',)
parser.add_argument('--stage2_lr_switch_epoch', type=int, default=40, help='number of Stage 2 epochs using --stage2_lr',)

parser.add_argument('--stage3_extractor_lr', type=float, default=5e-5,
                    help='Stage 3 extractor learning rate; defaults to --lr')
parser.add_argument('--stage3_regressor_lr', type=float, default=5e-4,
                    help='Stage 3 point-head learning rate; defaults to --lr')
parser.add_argument('--stage3_weight_decay', type=float, default=1e-5,
                    help='weight decay used by both Stage 3 optimizers')

parser.add_argument('--warmup_epoch', default=90, type=int, help='warm-up epochs')
parser.add_argument('--quantile_epoch',default=50,type=int,)
parser.add_argument('--nll_epoch',default=15,type=int,)

parser.add_argument('--momentum', type=float, default=0.9,
                    help='optimizer momentum')
parser.add_argument('--weight_decay', type=float,
                    default=1e-4, help='optimizer weight decay')

parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--print_freq', type=int,
                    default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--num_workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--per_label_plot', type=str,
                    default='per_label_metrics.pdf',
                    help='base output path for separate per-label variance and MAE figures')
#
parser.add_argument('--sigma', default=0.5, type=float)
parser.add_argument('--la', action='store_true',
                    help='if use logit adj to train the imbalance')
parser.add_argument('--model_depth', type=int, default=18,
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
parser.add_argument('--fd_ratio', type=float, default=0, help='scale of the diversity loss in z')
parser.add_argument('--beta', default=3.4, type=float,  help='beta for nll')
parser.add_argument('--variance_mse_threshold', type=float, default=1,
                    help='after warmup, switch samples with variance below this threshold from NLL to MSE')
parser.add_argument('--ema_variance_alpha', type=float, default=0.01,
                    help='distance decay for majority-label prediction variances used to replace median/low variances')

parser.add_argument('--ema_epoch_decay', type=float, default=0.95,
                    help='temporal EMA decay across NLL epochs')
parser.add_argument('--variance_floor', type=float, default=1e-6,
                    help='minimum variance used by beta-NLL and the EMA lookup')
parser.add_argument('--lamb', default=0.9, type=float,  help='lamb for coverage')
parser.add_argument('--weight', default=0.1, type=float,  help='weight for interval_loss in total loss')
parser.add_argument('--cp_mode', type=str, default='hybrid', choices=['cqr', 'split', 'hybrid'])
parser.add_argument('--warmup_ckpt_path', type=str, default='',
                    help='path to save the final warmup checkpoint; defaults to <store_root>/<store_name>/warmup_final.pth.tar')
parser.add_argument('--resume_warmup_ckpt', type=str, default='./save_true/stage1_rnc_final.pth',
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
# MSE only, else NLL
parser.add_argument('--MSE', action='store_true', help='only use MSE or not')
parser.add_argument('--MAE', action='store_true', help='only use MAE or not')
# first reweight and then judge if we can use LDS
parser.add_argument('--reweight', type=str, default='none',  choices=['none','inv', 'sqrt_inverse'],
                    help='weight : inv or sqrt_inv')
parser.add_argument('--smooth', default='none', choices=['lds', 'none'], help='use LDS or not')
parser.add_argument('--inv_method', default='cqr_pinball', choices=['split_cp', 'cqr_pinball', 'cqr_coverage'], help='use which method to train interval module')
parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_loader(args):
    train_transform = get_transforms(split='train', aug=args.aug)
    val_transform = get_transforms(split='val', aug=args.aug)
    print(f"Train Transforms: {train_transform}")
    print(f"Val Transforms: {val_transform}")
    df = pd.read_csv('/root/my_project/rnc/data/agedb.csv')
    df_train, df_val, df_test = df[df['split'] ==
                                'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    train_dataset = globals()[args.dataset](data_folder=args.data_folder, transform=train_transform, split='train')
    train_eval_dataset = AgeDB(data_folder=args.data_folder,transform=val_transform,split='train')
    val_dataset = globals()[args.dataset](data_folder=args.data_folder, transform=val_transform, split='val')
    test_dataset = globals()[args.dataset](data_folder=args.data_folder, transform=val_transform, split='test')

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_eval_loader, train_labels

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
        for idx, (x, y) in enumerate(test_loader):
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


def plot_per_label_metrics(
        var_per_label, mae_per_label, maj_labels, med_labels, low_labels,
        output_path):
    """Plot per-label prediction variance and MAE as separate bar charts."""
    if not var_per_label and not mae_per_label:
        print('No per-label metrics found; skipping the figure.')
        return

    def to_float(value):
        if torch.is_tensor(value):
            return value.detach().cpu().item()
        return float(value)

    # Colorblind-safe, low-saturation colors suitable for a paper figure.
    blue = '#0072B2'
    vermillion = '#D55E00'
    shot_groups = {
        'many': set(int(label) for label in maj_labels),
        'medium': set(int(label) for label in med_labels),
        'low': set(int(label) for label in low_labels),
    }
    shot_colors = {
        'many': '#DCE8D5',
        'medium': '#F3E5C3',
        'low': '#E4D9EB',
    }
    shot_legend = [
        Patch(facecolor=shot_colors['many'], edgecolor='none',
              label='Many-shot'),
        Patch(facecolor=shot_colors['medium'], edgecolor='none',
              label='Medium-shot'),
        Patch(facecolor=shot_colors['low'], edgecolor='none',
              label='Low-shot'),
    ]

    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_stem, output_ext = os.path.splitext(output_path)
    if output_ext.lower() not in {'.pdf', '.png'}:
        output_stem = output_path

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        def save_bar_chart(metric_dict, ylabel, color, name):
            if not metric_dict:
                print(f'No per-label {name} values found; skipping its figure.')
                return

            labels = sorted(metric_dict)
            values = [to_float(metric_dict[label]) for label in labels]
            fig, ax = plt.subplots(figsize=(7.0, 3.8))

            # Shade each label according to its training-set shot group.
            for label in labels:
                label_int = int(label)
                group = next(
                    (group_name for group_name, group_labels
                     in shot_groups.items() if label_int in group_labels),
                    None,
                )
                if group is not None:
                    ax.axvspan(
                        label - 0.5, label + 0.5,
                        facecolor=shot_colors[group], alpha=0.72,
                        edgecolor='none', zorder=0,
                    )

            ax.bar(
                labels, values, width=0.78, color=color, alpha=0.9,
                edgecolor='white', linewidth=0.3, zorder=3,
            )
            ax.set_xlabel('Label')
            ax.set_ylabel(ylabel)
            ax.grid(
                axis='y', color='#D9D9D9', linewidth=0.7,
                alpha=0.7, zorder=0,
            )
            ax.set_axisbelow(True)
            ax.margins(x=0.01)
            ax.set_xlim(labels[0] - 0.5, labels[-1] + 0.5)

            if len(labels) <= 20:
                ax.set_xticks(labels)
            else:
                tick_step = max(1, int(np.ceil(len(labels) / 12)))
                ax.set_xticks(labels[::tick_step])

            ax.legend(
                handles=shot_legend, loc='upper center',
                bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False,
            )
            fig.tight_layout()
            pdf_path = f'{output_stem}_{name}.pdf'
            png_path = f'{output_stem}_{name}.png'
            fig.savefig(pdf_path, bbox_inches='tight')
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved per-label {name} figure to {pdf_path} and {png_path}')

        save_bar_chart(
            var_per_label, 'Prediction variance', blue, 'variance',
        )
        save_bar_chart(
            mae_per_label, 'MAE', vermillion, 'mae',
        )
        
args = parser.parse_args()
train_loader, train_eval_loader, val_loader, test_loader, train_labels = set_loader(args)
model = RnCIntervalRegressor(name=args.model).to(device)
path = './checkpoint/final_model.pth.tar'
checkpoint = torch.load(path, map_location=device)
load_checkpoint_model(model, checkpoint, path)

mae, shot, gmean, var, mae_label = test(model, test_loader, train_labels, args)

maj, med, low = shot_count(train_labels)

plot_per_label_metrics(
    var, mae_label, maj, med, low, args.per_label_plot,
)

store_name = 'result.txt'

write_log(store_name, mae, shot, gmean)
