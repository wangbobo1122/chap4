"""
拜占庭攻击方法实现

包含多种针对联邦学习的攻击方法
"""

import torch
import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Dict


def trimmed_mean(updates: torch.Tensor, trim_ratio: float = 0.1) -> torch.Tensor:
    """
    Trimmed Mean 聚合函数（用于测试攻击效果）

    参数:
        updates: 所有客户端的更新，shape: (n_clients, ...)
        trim_ratio: 两端各修剪的比例

    返回:
        聚合后的更新
    """
    n_clients = updates.shape[0]
    trim_count = int(n_clients * trim_ratio)

    if trim_count == 0:
        return torch.mean(updates, dim=0)

    # 对每个参数维度 separately 进行 trimmed mean
    sorted_updates, _ = torch.sort(updates, dim=0)
    trimmed = sorted_updates[trim_count:n_clients - trim_count]

    return torch.mean(trimmed, dim=0)


def multi_krum(updates: torch.Tensor, n_byzantine: int, keep: int = 1) -> Tuple[torch.Tensor, int]:
    """
    Multi-Krum 聚合函数（用于测试攻击效果）

    参数:
        updates: 所有客户端的更新，shape: (n_clients, params_dim)
        n_byzantine: 恶意客户端数量
        keep: 保留的客户端数量

    返回:
        (聚合后的更新，选中客户端的索引)
    """
    n_clients = updates.shape[0]
    n_keep = max(n_clients - n_byzantine, 1)

    # 计算每对客户端之间的距离
    distances = torch.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            dist = torch.norm(updates[i] - updates[j])
            distances[i, j] = dist
            distances[j, i] = dist

    # 对每个客户端，计算到最近的 n_keep 个客户端的距离和
    scores = []
    for i in range(n_clients):
        sorted_distances = torch.sort(distances[i])[0]
        score = torch.sum(sorted_distances[1:n_keep + 1])  # 排除自己
        scores.append(score)

    # 选择分数最低的 keep 个客户端
    scores = torch.tensor(scores)
    selected_indices = torch.argsort(scores)[:keep]

    # 聚合选中的更新
    aggregated = torch.mean(updates[selected_indices], dim=0)

    return aggregated, selected_indices[0].item() if keep == 1 else selected_indices


def compute_lambda_fang(
    all_updates: torch.Tensor,
    model_re: torch.Tensor,
    n_attackers: int
) -> float:
    """
    计算 Fang 攻击中的最优 lambda 值

    参数:
        all_updates: 所有客户端的更新
        model_re: 全局模型参数
        n_attackers: 攻击者数量

    返回:
        最优的 lambda 值
    """
    n_total = len(all_updates)
    n_benign = n_total - n_attackers

    # 计算方差
    variance = torch.var(all_updates, dim=0)
    std = torch.sqrt(variance)

    # Fang 攻击的 lambda 上界
    lambda_max = n_total / (2 * n_attackers)

    # 使用自适应 lambda
    lamda = lambda_max * 0.5

    return lamda


def get_malicious_updates_fang(
    all_updates: torch.Tensor,
    model_re: torch.Tensor,
    n_attackers: int,
    threshold: float = 0.01
) -> torch.Tensor:
    """
    Fang 攻击 - 针对 Krum 的定制攻击

    核心思想：最大化聚合结果与原始均值的距离，同时能欺骗 Krum

    参数:
        all_updates: 所有良性客户端的更新
        model_re: 全局模型参数
        n_attackers: 攻击者数量
        threshold: 收敛阈值

    返回:
        恶意更新
    """
    deviation = torch.sign(model_re)
    lamda = compute_lambda_fang(all_updates, model_re, n_attackers)

    # 逐步减小 lambda 直到能欺骗 Krum
    while lamda > threshold:
        mal_update = -lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        # 测试是否能欺骗 Krum
        _, krum_candidate = multi_krum(mal_updates, n_attackers)

        if krum_candidate < n_attackers:  # Krum 选中了恶意更新
            return mal_updates[:n_attackers]

        lamda *= 0.5

    # 如果无法欺骗 Krum，返回最后的尝试
    mal_update = -lamda * deviation
    return torch.stack([mal_update] * n_attackers)


def get_malicious_updates_ipm(
    all_updates: torch.Tensor,
    model_re: torch.Tensor,
    n_attackers: int,
    attack_type: str = 'trimmed_mean',
    max_iterations: int = 100,
    threshold_diff: float = 1e-4
) -> torch.Tensor:
    """
    IPM (Inner Product Manipulation) 攻击 - 针对 Trimmed Mean 的定制攻击

    核心思想：最大化聚合结果与原始均值的距离

    参数:
        all_updates: 所有良性客户端的更新
        model_re: 全局模型参数
        n_attackers: 攻击者数量
        attack_type: 'trimmed_mean' 或 'krum'
        max_iterations: 最大迭代次数
        threshold_diff: 收敛阈值

    返回:
        恶意更新
    """
    lamda = 1.0
    lamda_succ = 0.0
    lamda_fail = 1.0
    prev_loss = 0.0

    deviation = torch.sign(model_re)

    for iteration in range(max_iterations):
        if torch.abs(lamda_succ - lamda) <= threshold_diff:
            break

        mal_update = model_re - lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        if attack_type == 'trimmed_mean':
            agg_grads = trimmed_mean(mal_updates)
        else:
            agg_grads, _ = multi_krum(mal_updates, n_attackers)

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            # 成功，尝试更大的 lambda
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            # 失败，减小 lambda
            lamda = lamda - lamda_fail / 2
            lamda_fail /= 2

        prev_loss = loss

    # 返回最终成功的恶意更新
    mal_update = model_re - lamda_succ * deviation
    return torch.stack([mal_update] * n_attackers)


def lie_attack(
    all_updates: torch.Tensor,
    n_attackers: int,
    n_total: int = None
) -> torch.Tensor:
    """
    LIE (Little Is Enough) 攻击

    原理：恶意更新被构造为略低于均值若干个标准差，刚好在检测阈值边缘

    参数:
        all_updates: 所有良性客户端的更新
        n_attackers: 攻击者数量
        n_total: 总客户端数量（包括攻击者）

    返回:
        恶意更新
    """
    if n_total is None:
        n_total = len(all_updates) + n_attackers

    n_benign = len(all_updates)

    # 计算统计信息
    mu = torch.mean(all_updates, dim=0)
    sigma = torch.std(all_updates, dim=0)

    # 计算攻击系数
    # 确保恶意更新在统计分布的边缘但不过于明显
    z_score = norm.ppf((n_total / 2 - 1) / (n_total - n_attackers))
    z_score = max(z_score, 1.0)  # 至少 1 个标准差

    # 生成恶意更新
    malicious_updates = []
    for i in range(n_attackers):
        # 恶意更新 = mu - z_score * sigma
        mal_update = mu - z_score * sigma
        malicious_updates.append(mal_update)

    return torch.stack(malicious_updates)


def scaling_attack(
    all_updates: torch.Tensor,
    n_attackers: int,
    factor: float = 10.0
) -> torch.Tensor:
    """
    Scaling 攻击 - 放大梯度

    参数:
        all_updates: 所有良性客户端的更新
        n_attackers: 攻击者数量
        factor: 放大因子

    返回:
        恶意更新
    """
    # 恶意更新 = 良性更新的均值 * factor
    mean_update = torch.mean(all_updates, dim=0)
    mal_update = mean_update * factor

    return torch.stack([mal_update] * n_attackers)


def mean_attack(
    all_updates: torch.Tensor,
    n_attackers: int
) -> torch.Tensor:
    """
    Mean 攻击 - 梯度反转

    参数:
        all_updates: 所有良性客户端的更新
        n_attackers: 攻击者数量

    返回:
        恶意更新
    """
    mean_update = torch.mean(all_updates, dim=0)
    mal_update = -mean_update

    return torch.stack([mal_update] * n_attackers)


def full_mean_attack(
    all_updates: torch.Tensor,
    n_attackers: int
) -> torch.Tensor:
    """
    Full Mean 攻击 - 完全抵消良性更新

    参数:
        all_updates: 所有良性客户端的更新
        n_attackers: 攻击者数量

    返回:
        恶意更新
    """
    n_benign = len(all_updates)
    all_sum = torch.sum(all_updates, dim=0)

    # 恶意更新的总和应该抵消良性更新的总和
    # m * mal_update = -all_sum
    mal_update = -all_sum / n_attackers

    return torch.stack([mal_update] * n_attackers)


def get_malicious_updates(
    all_updates: torch.Tensor,
    model_re: torch.Tensor,
    n_attackers: int,
    attack_method: str = 'fang'
) -> torch.Tensor:
    """
    获取恶意更新的统一接口

    参数:
        all_updates: 所有良性客户端的更新
        model_re: 全局模型参数
        n_attackers: 攻击者数量
        attack_method: 攻击方法，可选：'fang', 'ipm', 'lie', 'scaling', 'mean', 'full_mean'

    返回:
        恶意更新
    """
    attack_methods = {
        'fang': lambda: get_malicious_updates_fang(all_updates, model_re, n_attackers),
        'ipm': lambda: get_malicious_updates_ipm(all_updates, model_re, n_attackers),
        'lie': lambda: lie_attack(all_updates, n_attackers),
        'scaling': lambda: scaling_attack(all_updates, n_attackers, factor=10.0),
        'mean': lambda: mean_attack(all_updates, n_attackers),
        'full_mean': lambda: full_mean_attack(all_updates, n_attackers),
    }

    if attack_method not in attack_methods:
        raise ValueError(f"未知的攻击方法：{attack_method}")

    return attack_methods[attack_method]()
