import time
import os
import argparse
import logging
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib.patches import Patch
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INTERVAL_SCALE = 2.5632
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
    
def get_stage3_ema_variance(ckpt, target_epoch=None):
    """
    target_epoch 使用训练日志里的 1-based epoch，例如 epoch 1、epoch 5。
    不指定时，优先读取该 checkpoint 对应 epoch 的 EMA。
    """
    epoch_states = (
        ckpt.get("history", {})
        .get("nll_ema_variance", {})
        .get("epochs", [])
    )

    if target_epoch is not None:
        for state in epoch_states:
            if int(state["epoch"]) == target_epoch:
                return state["ema_variance"], target_epoch

        raise KeyError(
            f"checkpoint history 中没有 Stage 3 epoch {target_epoch}；"
            f"现有 epoch: {[x.get('epoch') for x in epoch_states]}"
        )

    if "previous_ema_variance" in ckpt:
        # checkpoint['epoch'] 是 0-based
        saved_epoch = int(ckpt["epoch"]) + 1
        return ckpt["previous_ema_variance"], saved_epoch

    if epoch_states:
        state = epoch_states[-1]
        return state["ema_variance"], int(state["epoch"])

    raise KeyError(
        "checkpoint 中没有 previous_ema_variance 或 EMA history。"
        "它可能是通过 --lightweight_stage3_checkpoint 保存的。"
    )
            
def variance_to_dataframe(label_variance):
    rows = []

    for label, variance in sorted(
        label_variance.items(),
        key=lambda item: int(item[0]),
    ):
        label = int(label)
        variance = max(float(variance), 0.0)
        std = math.sqrt(variance)

        # 训练时的定义：
        # variance = ((upper - lower) / 2.5632) ** 2
        width = INTERVAL_SCALE * std
        half_width = width / 2.0

        rows.append({
            "label": label,
            "ema_variance": variance,
            "ema_std": std,
            "effective_interval_width": width,
            "effective_half_width": half_width,
        })

    return pd.DataFrame(rows)


def variance_to_interval(label_variance):
    """Recover per-label interval width from the saved Stage 2 variance."""
    return {
        int(label): INTERVAL_SCALE * math.sqrt(
            max(float(variance), 0.0)
        )
        for label, variance in label_variance.items()
    }


def plot_per_label_interval_comparison(
        stage2_df, stage3_df, maj_labels, med_labels, low_labels,
        output_path):
    """Draw Stage 2 and Stage 3 EMA interval widths in one figure."""
    comparison_df = stage2_df.merge(
        stage3_df[
            ["label", "effective_interval_width"]
        ].rename(
            columns={
                "effective_interval_width": "stage3_ema_interval"
            }
        ),
        on="label",
        how="outer",
    ).sort_values("label")

    shot_groups = {
        "many": set(int(label) for label in maj_labels),
        "medium": set(int(label) for label in med_labels),
        "low": set(int(label) for label in low_labels),
    }
    shot_colors = {
        "many": "#DCE8D5",
        "medium": "#F3E5C3",
        "low": "#E4D9EB",
    }

    output_path = os.path.abspath(os.path.expanduser(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    output_stem, output_ext = os.path.splitext(output_path)
    if output_ext.lower() not in {".pdf", ".png"}:
        output_stem = output_path

    comparison_df.to_csv(
        f"{output_stem}.csv",
        index=False,
    )

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(9.0, 4.2))

        labels = comparison_df["label"].astype(int).tolist()
        for label in labels:
            group = next(
                (
                    name for name, group_labels in shot_groups.items()
                    if label in group_labels
                ),
                None,
            )
            if group is not None:
                ax.axvspan(
                    label - 0.5,
                    label + 0.5,
                    facecolor=shot_colors[group],
                    alpha=0.72,
                    edgecolor="none",
                    zorder=0,
                )

        bar_width = 0.38
        stage2_bars = ax.bar(
            comparison_df["label"] - bar_width / 2,
            comparison_df["interval"],
            width=bar_width,
            color="#0072B2",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
            label="Stage 2",
            zorder=3,
        )
        stage3_bars = ax.bar(
            comparison_df["label"] + bar_width / 2,
            comparison_df["stage3_ema_interval"],
            width=bar_width,
            color="#D55E00",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.35,
            label="Stage 3 EMA",
            zorder=3,
        )

        shot_legend = [
            Patch(
                facecolor=shot_colors["many"],
                edgecolor="none",
                label="Many-shot",
            ),
            Patch(
                facecolor=shot_colors["medium"],
                edgecolor="none",
                label="Medium-shot",
            ),
            Patch(
                facecolor=shot_colors["low"],
                edgecolor="none",
                label="Low-shot",
            ),
        ]
        ax.legend(
            handles=[stage2_bars, stage3_bars, *shot_legend],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=5,
            frameon=False,
        )

        ax.set_xlabel("Label")
        ax.set_ylabel("Interval width (log scale)")
        ax.set_yscale("log")
        ax.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.7,
            alpha=0.7,
            zorder=0,
        )
        ax.set_axisbelow(True)
        ax.margins(x=0.01)

        if labels:
            ax.set_xlim(min(labels) - 0.5, max(labels) + 0.5)
            tick_step = max(1, int(math.ceil(len(labels) / 12)))
            ax.set_xticks(labels[::tick_step])

        fig.tight_layout()
        pdf_path = f"{output_stem}.pdf"
        png_path = f"{output_stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved interval comparison CSV to {output_stem}.csv")
    print(f"Saved interval comparison figure to {pdf_path} and {png_path}")


path_stage2 = './checkpoint/ema_resnet50_q80_nll15/stage2_final.pth.tar'
checkpoint_stage2 = torch.load(path_stage2, map_location=device)
stage2_interval = variance_to_interval(
    checkpoint_stage2["initial_ema_variance"]
)
stage2_df = pd.DataFrame(
    sorted(stage2_interval.items()),
    columns=["label", "interval"],
)


path_stage3 = './checkpoint/ema_resnet50_q80_nll15/best_stage3_by_cal_mae.pth.tar'
checkpoint_stage3 = torch.load(path_stage3, map_location=device)
ema_variance, epoch = get_stage3_ema_variance(checkpoint_stage3)
df = variance_to_dataframe(ema_variance)

data_dir = '/root/autodl-tmp/data'
label_dataframe = pd.read_csv(os.path.join(data_dir, 'agedb.csv'))
train_labels = label_dataframe[
    label_dataframe['split'] == 'train'
]['age']
maj, med, low = shot_count(train_labels)

plot_per_label_interval_comparison(
    stage2_df,
    df,
    maj,
    med,
    low,
    (
        './checkpoint/ema_resnet50_q80_nll15/'
        f'stage2_vs_stage3_epoch_{epoch}_per_label_interval'
    ),
)
