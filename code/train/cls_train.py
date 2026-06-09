"""
cls_train.py
二分类（D3/M3）任务训练引擎，损失函数直接沿用 ACMIL 官方实现：
  https://github.com/dazhangyu123/ACMIL/blob/main/Step3_WSI_classification_ACMIL.py

官方损失（等权三项相加，无额外权重参数）:
    loss = diff_loss + loss0 + loss1

    loss1     : CE(slide_preds, label)          全局 bag 分类损失
    loss0     : CE(sub_preds, label × n_token)  子分支分类损失（n_token==1 时为 0）
    diff_loss : 各 token 注意力两两余弦相似度均值   多样性正则

评估指标:
    acc        : 整体准确率
    auc_macro  : 宏平均 AUC（OvR）
    f1_weighted: 加权 F1
    kappa      : Cohen's Kappa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)


# ──────────────────────────────────────────────────────────────────────
# 训练与评估
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataset, optimizer, device, accum_steps=8):
    """
    训练一个 epoch，损失公式与 ACMIL 官方完全一致。

    由于 WSI 变长序列不能直接 stack，采用梯度累积：
    每 accum_steps 个样本执行一次参数更新，等效 batch_size = accum_steps。

    Args:
        model       : ACMIL_GA（输出 sub_preds, slide_preds, attn）
        dataset     : ClsDataset
        optimizer   : 优化器
        device      : 设备
        accum_steps : 梯度累积步数

    Returns:
        avg_loss : float，每样本平均总损失
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    indices = np.random.permutation(len(dataset))
    total_loss = 0.0
    optimizer.zero_grad()

    for step, idx in enumerate(indices):
        feats, label, _ = dataset[int(idx)]
        feats = feats.to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        sub_preds, slide_preds, attn = model([feats])
        # sub_preds  : [n_token, n_classes]
        # slide_preds: [1, n_classes]
        # attn       : [1, n_token, N]

        n_token = sub_preds.shape[0]

        # ── 官方损失 ──────────────────────────────────────────────────
        # loss1: 全局 bag 分类损失
        loss1 = criterion(slide_preds, label_t)

        # loss0: 子分支分类损失（n_token==1 时 loss0=0，与官方一致）
        if n_token > 1:
            loss0 = criterion(sub_preds, label_t.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.0, device=device)

        # diff_loss: 各 token 注意力两两余弦相似度均值（官方原版）
        diff_loss_val = torch.tensor(0.0, device=device, dtype=torch.float)
        attn_soft = torch.softmax(attn, dim=-1)  # [1, n_token, N]
        if n_token > 1:
            n_pairs = n_token * (n_token - 1) / 2
            for i in range(n_token):
                for j in range(i + 1, n_token):
                    diff_loss_val += (
                        torch.cosine_similarity(
                            attn_soft[:, i], attn_soft[:, j], dim=-1
                        ).mean()
                        / n_pairs
                    )

        loss = (diff_loss_val + loss0 + loss1) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0 or step == len(indices) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(dataset)


@torch.no_grad()
def evaluate_cls(model, dataset, device, n_classes=2):
    """
    评估模型，使用 slide_preds（全局 bag）进行预测，与官方 evaluate() 一致。

    Returns:
        dict with keys:
            acc, auc_macro, f1_weighted, kappa,
            probs [N, n_classes], preds [N], labels [N],
            confusion_matrix [n_classes, n_classes]
    """
    model.eval()

    all_labels = []
    all_probs = []

    for idx in range(len(dataset)):
        feats, label, _ = dataset[idx]
        feats = feats.to(device)

        _, slide_preds, _ = model([feats])
        prob = torch.softmax(slide_preds, dim=-1).cpu().numpy()[0]  # [n_classes]

        all_probs.append(prob)
        all_labels.append(label)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)      # [N, n_classes]
    all_preds = all_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    try:
        if n_classes == 2:
            # 二分类：传正类概率，标准 AUC
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # 多分类：OvR macro AUC
            auc = roc_auc_score(
                all_labels, all_probs,
                multi_class="ovr", average="macro",
                labels=list(range(n_classes)),
            )
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))

    return {
        "acc": float(acc),
        "auc_macro": float(auc),
        "f1_weighted": float(f1),
        "kappa": float(kappa),
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
        "confusion_matrix": cm,
    }


def train_cls_model(model, train_dataset, val_dataset, device,
                    lr=1e-4, weight_decay=1e-3, max_epochs=100,
                    patience=20, n_classes=2, accum_steps=8,
                    save_path=None, verbose=True):
    """
    完整训练流程，支持早停（基于验证集 Macro AUC）。

    Args:
        model         : ACMIL_GA
        train_dataset : 训练集 ClsDataset
        val_dataset   : 验证集 ClsDataset
        device        : 设备
        lr            : 学习率
        weight_decay  : L2 权重衰减
        max_epochs    : 最大训练轮数
        patience      : 早停耐心值（Val Macro AUC 无提升的最大 epoch 数）
        n_classes     : 分类数
        accum_steps   : 梯度累积步数
        save_path     : 最优模型保存路径
        verbose       : 是否打印日志

    Returns:
        best_val_auc  : float
        best_metrics  : dict
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    best_val_auc = 0.0
    best_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_dataset, optimizer, device, accum_steps
        )
        val_metrics = evaluate_cls(model, val_dataset, device, n_classes)
        val_auc = val_metrics["auc_macro"]
        scheduler.step()

        if verbose:
            lr_cur = optimizer.param_groups[0]["lr"]
            auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else " nan "
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val AUC: {auc_str} | "
                f"Acc: {val_metrics['acc']:.4f} | "
                f"F1: {val_metrics['f1_weighted']:.4f} | "
                f"LR: {lr_cur:.2e}"
            )

        improved = not np.isnan(val_auc) and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_metrics = val_metrics
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"    -> 保存最优模型 (Val AUC: {val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"\n  早停: {patience} epochs 无改善, "
                        f"最优 Val AUC: {best_val_auc:.4f}"
                    )
                break

    return best_val_auc, best_metrics
