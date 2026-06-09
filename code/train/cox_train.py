"""
cox_train.py
Cox回归训练引擎：负对数部分似然损失 + ACMIL注意力多样性正则 + 训练循环。
"""

import torch
import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────

def cox_loss(risk_scores, times, events):
    """
    负对数部分似然（Cox loss）。

    Args:
        risk_scores: [batch] 各样本的risk score
        times: [batch] 生存时间
        events: [batch] 事件指示 (1=事件发生, 0=删失)

    Returns:
        loss: scalar
    """
    # 按时间降序排列（最长时间在前）
    order = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    # 数值保护：clamp risk scores 防止 exp() 溢出
    risk_scores = risk_scores.clamp(-20, 20)

    # cumsum(exp(risk)) 从最长时间累积到最短时间
    hazard_ratio = torch.cumsum(torch.exp(risk_scores), dim=0)
    log_risk = torch.log(hazard_ratio + 1e-7)
    uncensored_likelihood = risk_scores - log_risk

    # 仅对event=1的样本计算损失
    censored_loss = uncensored_likelihood * events
    n_events = events.sum()
    if n_events > 0:
        loss = -censored_loss.sum() / n_events
    else:
        loss = torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    return loss


def diff_loss(attention):
    """
    注意力多样性正则损失（来自ACMIL）。
    鼓励不同注意力头关注不同的patch区域。

    Args:
        attention: [1, n_token, N] 注意力权重

    Returns:
        loss: scalar
    """
    if attention.shape[1] <= 1:
        return torch.tensor(0.0, device=attention.device)

    attention = attention.squeeze(0)  # [n_token, N]
    attention = torch.softmax(attention, dim=1)

    # 计算各头之间的余弦相似度
    n_token = attention.shape[0]
    loss = torch.tensor(0.0, device=attention.device)
    count = 0
    for i in range(n_token):
        for j in range(i + 1, n_token):
            cos_sim = torch.cosine_similarity(
                attention[i].unsqueeze(0), attention[j].unsqueeze(0)
            )
            loss = loss + cos_sim.squeeze()
            count += 1
    if count > 0:
        loss = loss / count
    return loss


def compute_cindex(risk_scores, times, events):
    """
    计算Harrell's C-index。

    Args:
        risk_scores: numpy array or list
        times: numpy array or list
        events: numpy array or list

    Returns:
        c_index: float
    """
    try:
        # concordance_index(event_times, predicted_scores, event_observed)
        # 注意: 高risk score应对应更早事件，所以取负
        ci = concordance_index(times, -np.array(risk_scores), events)
        return ci
    except Exception:
        return 0.5


# ──────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataset, optimizer, device,
                    diff_weight=0.0, sub_weight=0.0):
    """
    训练一个epoch。
    由于Cox loss需要全部样本参与排序，采用全数据集训练（逐样本前向，收集后统一计算loss）。

    与官方ACMIL一致：每个子分支(token)独立计算Cox loss，而非取均值后计算一次。

    Args:
        model: ACMIL_Cox模型
        dataset: SurvivalDataset
        optimizer: 优化器
        device: 设备
        diff_weight: diff_loss权重
        sub_weight: sub-branch cox loss权重（官方默认=1.0）

    Returns:
        epoch_loss: float, 平均损失
        epoch_cindex: float, C-index
    """
    model.train()

    n_token = model.n_token

    all_slide_risks = []
    # 每个token独立收集risk scores（与官方ACMIL一致）
    all_sub_risks_per_token = [[] for _ in range(n_token)]
    all_times = []
    all_events = []
    all_attns = []

    # 逐样本前向传播，收集risk scores
    indices = np.random.permutation(len(dataset))
    for idx in indices:
        feats, time, event, sid = dataset[idx]
        feats = feats.to(device)

        sub_risks, slide_risk, attn = model(feats)

        all_slide_risks.append(slide_risk)
        # 每个token的risk score独立保存
        for t in range(n_token):
            all_sub_risks_per_token[t].append(sub_risks[t])
        all_times.append(time)
        all_events.append(event)
        if diff_weight > 0:
            all_attns.append(attn)

    # 组装tensor
    slide_risks_t = torch.stack(all_slide_risks)  # [N_samples]
    times_t = torch.tensor(all_times, dtype=torch.float32, device=device)
    events_t = torch.tensor(all_events, dtype=torch.float32, device=device)

    # 计算损失
    loss_slide = cox_loss(slide_risks_t, times_t, events_t)

    # 子分支损失：每个token独立计算Cox loss后求平均（与官方ACMIL一致）
    loss_sub = torch.tensor(0.0, device=device)
    if sub_weight > 0:
        for t in range(n_token):
            token_risks_t = torch.stack(all_sub_risks_per_token[t])  # [N_samples]
            loss_sub = loss_sub + cox_loss(token_risks_t, times_t, events_t)
        loss_sub = loss_sub / n_token

    # diff loss: 取所有样本的平均
    loss_diff = torch.tensor(0.0, device=device)
    if diff_weight > 0 and len(all_attns) > 0:
        for attn in all_attns:
            loss_diff = loss_diff + diff_loss(attn)
        loss_diff = loss_diff / len(all_attns)

    total_loss = loss_slide + sub_weight * loss_sub + diff_weight * loss_diff

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    # 放宽梯度裁剪：之前max_norm=1.0导致dimreduction独占梯度预算，
    # attention层(grad_norm~0.004)和cox_head几乎得不到更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    # C-index
    with torch.no_grad():
        risks_np = slide_risks_t.detach().cpu().numpy()
        times_np = times_t.cpu().numpy()
        events_np = events_t.cpu().numpy()
        ci = compute_cindex(risks_np, times_np, events_np)

    return total_loss.item(), ci


@torch.no_grad()
def evaluate(model, dataset, device):
    """
    评估模型：计算loss和C-index，返回所有risk scores。

    Returns:
        eval_loss: float
        c_index: float
        results: dict with keys slide_ids, risk_scores, times, events
    """
    model.eval()

    all_slide_risks = []
    all_times = []
    all_events = []
    all_sids = []

    for idx in range(len(dataset)):
        feats, time, event, sid = dataset[idx]
        feats = feats.to(device)

        sub_risks, slide_risk, attn = model(feats)

        all_slide_risks.append(slide_risk.item())
        all_times.append(time)
        all_events.append(event)
        all_sids.append(sid)

    # 计算loss
    risks_t = torch.tensor(all_slide_risks, dtype=torch.float32, device=device)
    times_t = torch.tensor(all_times, dtype=torch.float32, device=device)
    events_t = torch.tensor(all_events, dtype=torch.float32, device=device)
    eval_loss = cox_loss(risks_t, times_t, events_t).item()

    # C-index
    ci = compute_cindex(all_slide_risks, all_times, all_events)

    results = {
        "slide_ids": all_sids,
        "risk_scores": np.array(all_slide_risks),
        "times": np.array(all_times),
        "events": np.array(all_events),
    }

    return eval_loss, ci, results


def train_cox_model(model, train_dataset, val_dataset, device,
                    lr=5e-5, weight_decay=1e-3, max_epochs=200,
                    patience=30, diff_weight=0.0, sub_weight=0.0,
                    save_path=None):
    """
    完整的Cox模型训练流程，支持早停。

    Args:
        model: ACMIL_Cox模型
        train_dataset: 训练集SurvivalDataset
        val_dataset: 验证集SurvivalDataset
        device: 设备
        lr: 学习率
        weight_decay: 权重衰减
        max_epochs: 最大epoch数
        patience: 早停耐心值
        diff_weight: diff_loss权重
        sub_weight: sub-branch cox loss权重
        save_path: 最优模型保存路径

    Returns:
        best_val_cindex: float, 最优验证集C-index
        train_history: dict, 训练历史
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 纯CosineAnnealing衰减（无warmup），lr从初始值平滑衰减到接近0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    best_val_cindex = 0.0
    epochs_no_improve = 0
    train_history = {"train_loss": [], "train_ci": [], "val_loss": [], "val_ci": []}

    for epoch in range(1, max_epochs + 1):
        # 训练
        train_loss, train_ci = train_one_epoch(
            model, train_dataset, optimizer, device,
            diff_weight=diff_weight, sub_weight=sub_weight
        )

        # 验证
        val_loss, val_ci, _ = evaluate(model, val_dataset, device)

        scheduler.step()

        # 记录
        train_history["train_loss"].append(train_loss)
        train_history["train_ci"].append(train_ci)
        train_history["val_loss"].append(val_loss)
        train_history["val_ci"].append(val_ci)

        lr_current = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{max_epochs} | "
              f"Train Loss: {train_loss:.4f}, C-index: {train_ci:.4f} | "
              f"Val Loss: {val_loss:.4f}, C-index: {val_ci:.4f} | "
              f"LR: {lr_current:.2e}")

        # 早停
        if val_ci > best_val_cindex:
            best_val_cindex = val_ci
            epochs_no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  -> 保存最优模型 (Val C-index: {val_ci:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n早停: {patience} epochs无改善, 最优Val C-index: {best_val_cindex:.4f}")
                break

    return best_val_cindex, train_history
