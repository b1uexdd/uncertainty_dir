import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from collections import defaultdict
from scipy.stats import gmean
import random
from torch.distributions import Categorical, kl
import torch.nn as nn
softmax = nn.Softmax(dim=-1)
import torch.nn.functional as F
from utils import *
import statistics
import matplotlib.pyplot as plt
import math
from model import *


class AverageMeter(object):
    def __init__(self,  name = '', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question):
    """ Ask a yes/no question via input() and return their answer. """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, state, is_best, prefix=''):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def shot_metric(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(labels[labels == l]))
        l1_per_class.append(
            np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(
            np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            #print(train_class_count[i])
            #print(l1_per_class[i])
            #print(l1_all_per_class[i])
        else:
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    #
    shot_dict = defaultdict(dict)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


def balanced_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    mse_per_class, l1_per_class = [], []
    for l in np.unique(labels):
        mse_per_class.append(np.mean((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.mean(np.abs(preds[labels == l] - labels[labels == l])))

    mean_mse = sum(mse_per_class) / len(mse_per_class)
    mean_l1 = sum(l1_per_class) / len(l1_per_class)
    return mean_mse, mean_l1


def setup_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabale = False


def shot_metric_balanced(pred, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    # input of the pred & labels are all numpy.darray
    # train_labels is from csv , e.g. df['age']
    #
    preds = np.hstack(pred)
    labels = np.hstack(labels)
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count, test_class_count = [], []
    #
    l1_per_class, l1_all_per_class = [], []
    #
    for l in np.unique(labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
        test_class_count.append(
            len(labels[labels == l]))
        l1_per_class.append(
            np.mean(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(
            np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
            #print(train_class_count[i])
            #print(l1_per_class[i])
            #print(l1_all_per_class[i])
        else:
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])
    
    
    shot_dict = defaultdict(dict)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / len(many_shot_l1)
    #shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    #
    shot_dict['median']['l1'] = np.sum(
        median_shot_l1) / len(median_shot_cnt)
    #shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    #
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / len(low_shot_cnt)
    #shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict






# calculate the majority, median and low shot labels   
def shot_count(train_labels, many_shot_thr=100, low_shot_thr=20):
    #
    train_labels = np.array(train_labels).astype(int)
    #
    train_class_count = []
    #
    maj_class, med_class, min_class = [], [], []
    #
    for l in np.unique(train_labels):
        train_class_count.append(len(
            train_labels[train_labels == l]))
    #
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            maj_class.append(np.unique(train_labels)[i])
        elif train_class_count[i] < low_shot_thr:
            min_class.append(np.unique(train_labels)[i])
        else:
            med_class.append(np.unique(train_labels)[i]) 
    #
    return maj_class, med_class, min_class


def shot_reg(label, pred, maj, med, min):
    # how many preditions in this shots
    pred_dict = {'maj':0, 'med':0, 'min':0}
    # how many preditions from min to med, min to maj, med to maj, min to med
    pred_label_dict = {'min to med':0, 'min to maj':0, 'med to maj':0, 'med to min':0, 'maj to min':0, 'maj to med':0}
    #
    pred = int_tensors(pred)
    #
    #print(maj)
    #print(med)
    #print(min)
    #print(f' size of label {type(label)} size of pred {type(pred)}')
    #
    labels, preds = np.stack(label), np.stack(pred)
    #
    #print(f' labels {labels[:100]} preds {preds[:100]}')
    #dis = np.floor(np.abs(labels - preds)).tolist()
    bsz = labels.shape[0]
    for i in range(bsz):
        k_pred = check_shot(preds[i],maj, med, min)
        k_label = check_shot(labels[i],maj, med, min)
        if k_pred in pred_dict.keys():
            pred_dict[k_pred] = pred_dict[k_pred] + 1
        pred_shift = check_pred_shift(k_pred, k_label)
        if pred_shift in pred_label_dict.keys():
            pred_label_dict[pred_shift] = pred_label_dict[pred_shift] + 1
    return pred_dict['maj'], pred_dict['med'], pred_dict['min'], \
        pred_label_dict['min to med'], pred_label_dict['min to maj'], pred_label_dict['med to maj'],pred_label_dict['med to min'],pred_label_dict['maj to min'],pred_label_dict['maj to med']


def check_shot(e, maj, med, min):
    if e in maj:
        return 'maj'
    elif e in med:
        return 'med'
    else:
        return 'min'
    
# check reditions from min to med, min to maj, med to maj
def check_pred_shift(k_pred, k_label):
    if k_pred == 'med' and k_label == 'min':
        return 'min to med'
    elif k_pred == 'maj' and k_label == 'min':
        return 'min to maj'
    elif k_pred == 'maj' and k_label == 'med':
        return 'med to maj'
    elif k_pred == 'min' and k_label == 'med':
        return 'med to min'
    elif k_pred == 'min' and k_label == 'maj':
        return 'maj to min'
    elif k_pred == 'med' and k_label == 'maj':
        return 'maj to med'
    else:
        return 'others'
      
# invert the pred to its closest interger
def int_tensors(pred):
    pred = torch.Tensor(pred)
    #pred = pred - torch.floor(pred)
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    diff = pred - torch.floor(pred)
    diff = torch.where(diff > 0.5, one, diff)
    diff = torch.where(diff < 0.5, zero, diff)
    pred = torch.floor(pred) + diff
    pred = torch.clamp(pred, 0, 100)
    pred = pred.tolist()
    return pred


 # calculate the frob norm of test on different shots (frobs norm and nuc norm)
def cal_frob_norm(y, feat, majs, meds, mino, maj_shot, med_shot, min_shot, maj_shot_nuc, med_shot_nuc, min_shot_nuc, device):
    bsz = y.shape[0]
    maj_index, med_index, min_index = [], [], []
    for i in range(bsz):
        if y[i] in majs:
            maj_index.append(i)
        elif y[i] in meds:
            med_index.append(i)
        else:
            min_index.append(i)
    #
    if len(maj_index) != 0:
        majority = torch.index_select(feat, dim=0, index=torch.LongTensor(maj_index).to(device))
        ma = torch.mean(torch.norm(majority, p='fro', dim=-1))
        ma_nuc = torch.norm(majority, p='nuc')/majority.shape[0]
        #maj_shot = math.sqrt(maj_shot**2 + ma)
        maj_shot.update(ma.item(), majority.shape[0])
        maj_shot_nuc.update(ma_nuc.item(), majority.shape[0])
    if len(med_index) != 0:
        median = torch.index_select(feat, dim=0, index=torch.LongTensor(med_index).to(device))
        md = torch.mean(torch.norm(median, p='fro', dim=-1))
        md_nuc = torch.norm(median, p='nuc')/median.shape[0]
        #med_shot = math.sqrt(med_shot**2 + md)
        med_shot.update(md.item(), median.shape[0])
        med_shot_nuc.update(md_nuc.item(), median.shape[0])
    if len(min_index) != 0:
        minority = torch.index_select(feat, dim=0, index=torch.LongTensor(min_index).to(device))
        mi = torch.mean(torch.norm(minority, p='fro', dim=-1))
        mi_nuc = torch.norm(minority, p='nuc')/minority.shape[0]
        #min_shot = math.sqrt(mi**2 + mi)
        min_shot.update(mi.item(), minority.shape[0])
        min_shot_nuc.update(mi_nuc.item(), minority.shape[0])
    return maj_shot, med_shot, min_shot, maj_shot_nuc, med_shot_nuc, min_shot_nuc


# calculate the prediction variance w.r.t the ground truth labels
# return a dictionary : {key (is the label): prediction list}
def cal_pred_L1_distance(preds, labels):
    #
    preds = np.hstack(preds)
    labels = np.hstack(labels)
    preds_tesnor = torch.Tensor(preds)
    #
    label_to_pred_index = {}
    for l in np.unique(labels):
        pred_index = np.argwhere(labels==l)[:,0].tolist()
        label_to_pred_index[l] = torch.index_select(preds_tesnor, dim=0, index=torch.Tensor(pred_index).to(torch.int32)).squeeze(-1).tolist()
    return label_to_pred_index



# calculate the l1 distance between the prediction and ground truth
# calculate variacne of predictions
def variance_mean_cal(label_to_pred_index, train_labels):
    maj, med, low = shot_count(train_labels)
    #label_to_pred_index = cal_pred_L1_distance(preds, labels)
    #
    index_list = []
    #shot_list_mean, shot_list_variance = [], []
    #mean_list = []
    #variance_list = []
    maj_mean, med_mean, low_mean = [], [], []
    maj_var, med_var, low_var =  [], [], []
    maj_index, med_index, low_index = [], [], []
    #
    for k in label_to_pred_index.keys():
        mean = statistics.mean(label_to_pred_index[k])
        l1 = abs(mean -  k)
        variance = statistics.variance(label_to_pred_index[k])
        if k in maj:
            #shot_list_mean.append('r')
            #shot_list_variance.append('y')
            maj_mean.append(l1)
            med_mean.append(0)
            low_mean.append(0)
            maj_var.append(variance)
            med_var.append(0)
            low_var.append(0)
        elif k in med:
            #shot_list_mean.append('g')
            #shot_list_variance.append('c')
            maj_mean.append(0)
            med_mean.append(l1)
            low_mean.append(0)
            maj_var.append(0)
            med_var.append(variance)
            low_var.append(0)
        else:
            #shot_list_mean.append('b')
            #shot_list_variance.append('w')
            maj_mean.append(0)
            med_mean.append(0)
            low_mean.append(l1)
            maj_var.append(0)
            med_var.append(0)
            low_var.append(variance)
        index_list.append(k)
    mean_list = [maj_mean, med_mean, low_mean]
    var_list = [maj_var, med_var, low_var]   
    return index_list, mean_list, var_list

# draw the mean and variance given their correpsonding maj, med, and low groups
def draw_bias_bar(index_list, mean_list, var_list, prefix='ce'):
    xx = [i for i in range(len(index_list))]
    [maj_mean, med_mean, low_mean] = mean_list
    [maj_var, med_var, low_var] = var_list
    rect1 = plt.bar(xx, height=maj_mean, width=0.2, label='maj')
    rect2 = plt.bar(xx, height=med_mean, width=0.2, label='med')
    rect3 = plt.bar(xx, height=low_mean, width=0.2, label='low')
    plt.ylabel("mean_difference")
    #
    plt.xticks(xx[::20], index_list[::20])
    plt.xlabel("Age")
    plt.legend()
    plt.savefig(f'./{prefix}_mean.png')
    plt.show()
    plt.close()
    rect1 = plt.bar(xx, height=maj_var, width=0.2, label='maj')
    rect2 = plt.bar(xx, height=med_var, width=0.2, label='med')
    rect3 = plt.bar(xx, height=low_var, width=0.2, label='low')
    plt.ylabel("var_difference")
    #
    plt.xticks(xx[::20], index_list[::20])
    plt.xlabel("Age")
    plt.legend()
    plt.savefig(f'./{prefix}_variance.png')
    plt.show()
    plt.close()


# pearson correlation between two lists
def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den



# no currently used
def test_group_acc(model, train_loader, prefix, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder_regression(groups=args.groups, name='resnet18')
    model = torch.load(f'./models/best_{prefix}.pth')
    model.eval()
    pred, labels = [], []
    for idx, (x, y, g) in enumerate(train_loader):
        x, y, g = x.to(device), y.to(device), g.to(device)
        with torch.no_grad():
            y_output,  z = model(x)
            y_chunk = torch.chunk(y_output, 2, dim=1)
            g_hat, y_pred = y_chunk[0], y_chunk[1]
            g_index = torch.argmax(g_hat, dim=1).unsqueeze(-1)
            pred.extend(g_index.data.cpu().numpy())
            labels.extend(g.data.cpu().numpy())
    pred = np.array(pred)
    labels = np.array(labels)
    np.save(f'./acc/pred{prefix}.npy', pred)
    np.save(f'./acc/labels{prefix}.npy', labels)




# in a mini-batch, calculate the variance in each shot and total
# var, prediction variances
# label, ground truth target labels
# maj, med, low labels
def uncertainty_accumulation(var, label, maj, med, low, device):
    maj_indice = torch.nonzero(torch.isin(label, torch.Tensor(maj).to(device)), as_tuple=False)
    med_indice = torch.nonzero(torch.isin(label, torch.Tensor(med).to(device)), as_tuple=False)
    low_indice = torch.nonzero(torch.isin(label, torch.Tensor(low).to(device)), as_tuple=False)
    maj_var = torch.mean(var[maj_indice[:, 0]].squeeze(-1).to(torch.float))
    med_var = torch.mean(var[med_indice[:, 0]].squeeze(-1).to(torch.float))
    low_var = torch.mean(var[low_indice[:, 0]].squeeze(-1).to(torch.float))
    total_var = torch.mean(var.squeeze(-1).to(torch.float))
    return maj_var.item(), med_var.item(), low_var.item(), total_var.item()



# in a mini-batch, calculate the prediction variance from predicted values in each shot and total
# pred, target lable prediction 
# label, ground truth target labels
# maj, med, low labels
def label_uncertainty_accumulation(pred, label, maj, med, low, device):
    maj_indice = torch.nonzero(torch.isin(label, torch.Tensor(maj).to(device)), as_tuple=False)
    med_indice = torch.nonzero(torch.isin(label, torch.Tensor(med).to(device)), as_tuple=False)
    low_indice = torch.nonzero(torch.isin(label, torch.Tensor(low).to(device)), as_tuple=False)
    maj_var = torch.var(pred[maj_indice[:, 0]].squeeze(-1).to(torch.float))
    med_var = torch.var(pred[med_indice[:, 0]].squeeze(-1).to(torch.float))
    low_var = torch.var(pred[low_indice[:, 0]].squeeze(-1).to(torch.float))
    #
    total_var = (maj_var*maj_indice.shape[0]+ med_var*med_indice.shape[0] + low_var*low_indice.shape[0])\
        /(maj_indice.shape[0] + med_indice.shape[0] + low_indice.shape[0])
    #
    return maj_var.item(), med_var.item(), low_var.item(), total_var.item()


#
#
#
def per_label_mae(output, target):
    """
    output: Tensor of shape (N, 1)
    target: Tensor of shape (N,) with M unique labels
    Returns: dict mapping each label to its MAE
    """
    output = output.view(-1)  # (N,)
    target = target.view(-1)  # (N,)

    unique_labels = target.unique()
    mae_dict = {}

    for label in unique_labels:
        mask = target == label
        if mask.sum() == 0:
            continue
        pred_subset = output[mask]
        true_subset = target[mask].float()
        mae = torch.abs(pred_subset - true_subset).mean()
        mae_dict[int(label.item())] = mae.item()

    return mae_dict




#
# per label forbenius norm
#
def per_label_frobenius_norm(features, labels):
    """
    features: Tensor of shape (N, D)
    labels: Tensor of shape (N,)
    Returns: dict {label: avg Frobenius norm}
    """
    features = features.view(features.size(0), -1)  # Ensure shape (N, D)
    labels = labels.view(-1)  # Ensure shape (N,)

    unique_labels = labels.unique()
    frob_norms = {}

    for label in unique_labels:
        mask = labels == label
        feats = features[mask]  # (n_c, D)
        if feats.size(0) == 0:
            continue
        norms = torch.norm(feats, p='fro', dim=1)  # L2 norm per row
        avg = norms.mean().item()
        frob_norms[int(label.item())] = avg

    return frob_norms



####
# return the variance of per label
# return a dictionary
###
def per_label_var(preds, labels):
    label_to_preds = defaultdict(list)
    for pred, label in zip(preds, labels):
        label_to_preds[int(label.item())].append(pred.item())
    label_variance = {}
    for label, group_preds in label_to_preds.items():
        group_tensor = torch.tensor(group_preds)
        label_variance[label] = torch.var(group_tensor, unbiased=True)  # sample variance
    #
    all_labels, all_vars = [], []
    for k in sorted(label_variance.keys()):
        all_labels.append(k)
        all_vars.append(label_variance[k].item())
    print('----labels in var per label-----')
    print(all_labels)
    print('-----vars per label-----')
    print(all_vars)
        #print(f"Label {label}: prediction variance = {label_variance[label].item():.4f}")
    return label_variance