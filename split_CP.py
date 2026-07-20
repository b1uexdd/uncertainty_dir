import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from conform_cqr import pinball_loss

#CQR-based interval estimation
#revise no gradient
def calibrate_qhat_from_batch(model, cal_batch, device, alpha=0.1):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_cal, y_cal, _ = cal_batch
        x_cal, y_cal = x_cal.to(device), y_cal.to(device)
        y_pred, lower, upper, _ = model(x_cal)
        score = torch.maximum(lower - y_cal, y_cal - upper)
        score = torch.clamp(score, min=0.0)
        q_hat = torch.quantile(score.flatten(), 1 - alpha)
    model.train(was_training)
    # calculate the loss
    #cp_loss = coverage_loss(y_cal, y_pred, lower, upper, alpha)
    #
    return q_hat

#split-CP based interval estimation, 90% coverage 
def calibrate_qhat_splitCP(model, cal_batch, device, alpha=0.1):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_cal, y_cal, _ = cal_batch
        x_cal, y_cal = x_cal.to(device), y_cal.to(device)
        y_pred, _, _, _ = model(x_cal)
        q_hat_list = torch.abs(y_cal - y_pred).flatten()
        q_hat = torch.quantile(q_hat_list, 1 - alpha)
    model.train(was_training)
    #lower, upper = y_pred-q_hat, y_pred+q_hat
    #cp_loss = coverage_loss(y_cal, y_pred, lower, upper, alpha)
    return q_hat


# return interval
def get_interval(model, cal_batch, device, alpha=0.1, pattern='cqr'):
    assert pattern in ['cqr', 'split']
    if pattern == 'cqr':
        return calibrate_qhat_from_batch(model, cal_batch, device, alpha=0.1)
    elif pattern == 'split':
        return calibrate_qhat_splitCP(model, cal_batch, device, alpha=0.1)
    else:
        raise NotImplementedError



def predict_with_interval(
    model: nn.Module,
    x: torch.Tensor,
    q_hat: float,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns point prediction, lower bound, upper bound.
    """
    model.eval()
    x = x.to(device)
    pred, lower, upper, _ = model(x)
    lower = lower - q_hat
    upper = upper + q_hat
    return pred.cpu(), lower.cpu(), upper.cpu()



def evaluate_conformal(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    q_hat: float,
    device: str
) -> Tuple[float, float]:
    """
    Returns:
    - empirical coverage
    - average interval width
    """
    pred, lower, upper = predict_with_interval(model, x_test, q_hat, device)

    y_test_cpu = y_test.cpu()
    covered = ((y_test_cpu >= lower) & (y_test_cpu <= upper)).float()
    coverage = covered.mean().item()

    avg_width = (upper - lower).mean().item()
    return coverage, avg_width


# split CP loss (dual output)
def coverage_loss(
        true: torch.Tensor,
        prediction: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        lamb: float
    ):
    '''
        This loss can be used for both CQR and Split CP, as you only need revise the upper and lower predictions
            ---- from model prediction: upper and lower --> CQR
            ---- from model y_pred --> construct the upper and lower --> Split CP
    '''
    #
    bound_penalty = F.relu(lower - prediction) + F.relu(prediction - upper)
    #coverage
    k=5
    soft_covered = torch.sigmoid(k * (true - lower)) * torch.sigmoid(k * (upper - true))
    coverage = soft_covered.mean()
    coverage_penalty = torch.relu(lamb - coverage)
    #
    loss = bound_penalty.mean() + coverage_penalty.mean()  
    #
    return loss


# dual output, with different tau set up, return both upper and lower loss bar
def cqr_pinball(
        y_true : torch.Tensor,
        upper,
        lower,
        lamb: float
    ):
    #
    tau = 1 - lamb
    high, low  = 1 - tau/2, tau/2
    loss_upper_quantile = pinball_loss(y_true, upper, high)
    loss_lower_quantile = pinball_loss(y_true, lower, low)
    #loss = loss_lower_quantile + loss_upper_quantile
    return loss_lower_quantile, loss_upper_quantile


# minimize the prediction interval
def interval_minimization(
        lower: torch.Tensor,
        upper: torch.Tensor,
    ):
    loss = torch.abs(upper-lower).mean()
    return loss