"""
TD (Truth Discovery) 真值发现算法

用于计算客户端和边缘服务器的声誉权重
离真值越近的更新，权重越大

算法流程:
1. 初始化：输入 n 个客户端模型 {θ₁, θ₂, ..., θₙ}，随机初始化真值模型 θ
2. 权重估计：wᵢ = log(Σ||θⱼ - θ||₂) - log(||θᵢ - θ||₂)
3. 真值更新：θ = (Σwᵢ·θᵢ) / (Σwᵢ)
4. 迭代收敛：||θ_new - θ_old||₂ < ε (建议 ε = 10⁻⁶)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


# 收敛阈值常量
EPSILON_DEFAULT = 1e-6


def calculate_model_distance(model1: Dict, model2: Dict) -> float:
    """
    计算两个模型之间的 L2 距离
    度量：d(θᵢ, θ) = ||θᵢ - θ||₂

    参数:
        model1: 模型参数字典 1
        model2: 模型参数字典 2

    返回:
        L2 距离
    """
    total_distance = 0.0

    for key in model1.keys():
        diff = model1[key] - model2[key]
        total_distance += torch.sum(diff ** 2)

    return torch.sqrt(total_distance).item()


def flatten_model(model_params: Dict) -> torch.Tensor:
    """
    将模型参数字典扁平化为一维张量

    参数:
        model_params: 模型参数据字典

    返回:
        扁平化的一维张量
    """
    flattened = []
    for key in sorted(model_params.keys()):
        flattened.append(model_params[key].flatten())
    return torch.cat(flattened)


def unflatten_model(flattened: torch.Tensor, template: Dict) -> Dict:
    """
    将扁平化张量恢复为模型参数字典

    参数:
        flattened: 扁平化的一维张量
        template: 模板参数字典（用于获取形状信息）

    返回:
        模型参数字典
    """
    result = {}
    idx = 0

    for key in sorted(template.keys()):
        shape = template[key].shape
        size = template[key].numel()
        result[key] = flattened[idx:idx + size].reshape(shape)
        idx += size

    return result


def td_weight_estimation(
    models: List[Dict],
    truth_model: Dict = None,
    max_iterations: int = 100,
    epsilon: float = EPSILON_DEFAULT,
    regularizer: float = 1e-10
) -> Tuple[List[float], Dict]:
    """
    TD 权重估计和真值更新算法

    算法流程:
    1. 初始化：输入 n 个客户端模型 {θ₁, θ₂, ..., θₙ}，初始化权重 {w₁, w₂, ..., wₙ}，随机初始化真值模型 θ
    2. 权重估计：距离真值 θ 越近的模型，权重 wi 越大
       度量：d(θᵢ, θ) = ||θᵢ - θ||₂
       公式：wᵢ = log(Σ||θⱼ - θ||₂) - log(||θᵢ - θ||₂)
    3. 真值更新：对所有本地模型进行加权平均
       公式：θ = (Σwᵢ · θᵢ) / (Σwᵢ)
    4. 迭代收敛：重复步骤 2 和 3，直到 ||θ_new - θ_old||₂ < ε

    参数:
        models: 所有客户端/服务器的模型参数字典列表 {θ₁, θ₂, ..., θₙ}
        truth_model: 初始真值模型（可选，默认随机初始化）
        max_iterations: 最大迭代次数
        epsilon: 收敛阈值 (建议 ε = 10⁻⁶)
        regularizer: 正则化项，防止分母为零

    返回:
        (权重列表 {w₁, w₂, ..., wₙ}, 最终真值模型 θ)
    """
    n_models = len(models)

    if n_models == 0:
        return [], None

    if n_models == 1:
        return [1.0], models[0]

    # 扁平化所有模型
    flattened_models = [flatten_model(m) for m in models]
    stacked_models = torch.stack(flattened_models)

    # 1. 初始化
    # 如果没有初始真值模型，随机初始化或使用所有模型的均值
    if truth_model is None:
        # 随机初始化真值模型（在模型均值附近添加小的随机扰动）
        truth = torch.mean(stacked_models, dim=0)
        # 添加小的随机扰动以增加随机性
        noise = torch.randn_like(truth) * 0.01
        truth = truth + noise
    else:
        truth = flatten_model(truth_model)

    # 初始化权重 {w₁, w₂, ..., wₙ}
    weights = [1.0 / n_models] * n_models

    # 迭代优化
    for iteration in range(max_iterations):
        truth_old = truth.clone()

        # 2. 权重估计 (Weight Estimation)
        # 计算每个模型与真值的 L2 距离：d(θᵢ, θ) = ||θᵢ - θ||₂
        distances = []
        for i in range(n_models):
            dist = torch.norm(stacked_models[i] - truth, p=2)
            distances.append(dist)

        # 计算总距离：Σ||θⱼ - θ||₂
        total_distance = sum(distances)

        # 计算权重：wᵢ = log(Σ||θⱼ - θ||₂) - log(||θᵢ - θ||₂)
        # 距离越小，权重越大
        weights = []
        for i, dist in enumerate(distances):
            # 添加正则化项防止 log(0)
            dist_reg = dist + regularizer
            total_dist_reg = total_distance + regularizer * n_models
            w = torch.log(total_dist_reg) - torch.log(dist_reg)
            # 确保权重为正
            w = max(0.0, w.item())
            weights.append(w)

        # 计算权重和：Σwᵢ
        weight_sum = sum(weights)

        # 处理权重和为零的情况
        if weight_sum < regularizer:
            # 如果所有权重都接近零，使用均匀权重
            weights = [1.0 / n_models] * n_models
            weight_sum = 1.0

        # 3. 真值更新 (Truth Update)
        # 公式：θ = (Σwᵢ · θᵢ) / (Σwᵢ) - 加权平均
        truth = torch.zeros_like(truth)
        for i in range(n_models):
            truth += weights[i] * stacked_models[i]
        truth = truth / weight_sum

        # 4. 检查收敛 (Convergence Check)
        # 停止条件：||θ_new - θ_old||₂ < ε
        diff = torch.norm(truth - truth_old, p=2)
        if diff < epsilon:
            print(f"[TD] 收敛于第 {iteration + 1} 次迭代，真值变化：{diff.item():.8f}")
            break

    # 恢复真值模型
    truth_model_final = unflatten_model(truth, models[0])

    return weights, truth_model_final


def td_reputation_update(
    client_models: List[Dict],
    current_reputations: List[float],
    global_model: Dict,
    alpha: float = 0.7,  # EMA 平滑系数
    epsilon: float = EPSILON_DEFAULT
) -> List[float]:
    """
    使用 TD 算法更新客户端声誉

    参数:
        client_models: 所有客户端的模型参数列表
        current_reputations: 当前声誉列表
        global_model: 全局模型参数
        alpha: EMA 平滑系数
        epsilon: 收敛阈值

    返回:
        更新后的声誉列表
    """
    # 使用 TD 算法计算权重
    td_weights, _ = td_weight_estimation(
        models=client_models,
        truth_model=global_model,
        epsilon=epsilon
    )

    # 使用 EMA 平滑更新声誉
    updated_reputations = []
    for i in range(len(current_reputations)):
        new_reputation = alpha * td_weights[i] + (1 - alpha) * current_reputations[i]
        updated_reputations.append(new_reputation)

    return updated_reputations


def td_edge_server_reputation_update(
    edge_models: List[Dict],
    cloud_model: Dict,
    current_reputations: List[float],
    alpha: float = 0.7,
    epsilon: float = EPSILON_DEFAULT
) -> List[float]:
    """
    使用 TD 算法更新边缘服务器声誉（由云服务器计算）

    参数:
        edge_models: 所有边缘服务器的模型参数列表
        cloud_model: 云服务器全局模型
        current_reputations: 当前声誉列表
        alpha: EMA 平滑系数
        epsilon: 收敛阈值

    返回:
        更新后的声誉列表
    """
    # 使用 TD 算法计算边缘服务器权重
    td_weights, _ = td_weight_estimation(
        models=edge_models,
        truth_model=cloud_model,
        epsilon=epsilon
    )

    # 使用 EMA 平滑更新声誉
    updated_reputations = []
    for i in range(len(current_reputations)):
        new_reputation = alpha * td_weights[i] + (1 - alpha) * current_reputations[i]
        updated_reputations.append(new_reputation)

    return updated_reputations


def td_aggregate(
    models: List[Dict],
    global_model: Dict,
    reputations: List[float] = None,
    use_td_weights: bool = True
) -> Dict:
    """
    使用 TD 权重进行聚合

    参数:
        models: 所有模型参数字典列表
        global_model: 全局模型参数
        reputations: 客户端声誉列表（来自 history_reputation），如果为 None 则使用 TD 算法计算
        use_td_weights: 是否使用 TD 权重（否则使用均匀权重）

    返回:
        聚合后的模型参数
    """
    if len(models) == 0:
        return global_model

    if not use_td_weights:
        # 简单平均
        aggregated = {}
        for key in global_model.keys():
            aggregated[key] = torch.mean(torch.stack([m[key] for m in models]), dim=0)
        return aggregated

    # 如果提供了声誉列表，直接使用；否则使用 TD 算法计算
    if reputations is not None:
        weights = normalize_reputations(reputations)
    else:
        weights, _ = td_weight_estimation(models, global_model)

    aggregated = {}
    for key in global_model.keys():
        weighted_sum = torch.zeros_like(global_model[key])
        for i, model in enumerate(models):
            weighted_sum += weights[i] * model[key]
        aggregated[key] = weighted_sum

    return aggregated


def normalize_reputations(reputations: List[float]) -> List[float]:
    """
    将声誉值归一化为权重

    参数:
        reputations: 声誉列表

    返回:
        归一化后的权重列表
    """
    total = sum(reputations)
    if total > 0:
        return [r / total for r in reputations]
    else:
        return [1.0 / len(reputations)] * len(reputations)


def td_aggregate_hierarchical(
    client_models_per_edge: List[List[Dict]],
    edge_models: List[Dict],
    cloud_model: Dict,
    epsilon: float = EPSILON_DEFAULT
) -> Tuple[Dict, List[float], List[float]]:
    """
    层级式 TD 聚合算法（扩展到边缘服务器）

    流程:
    1. 在每个边缘服务器上使用 TD 算法聚合其客户端模型
    2. 在云服务器上使用 TD 算法聚合所有边缘服务器模型

    参数:
        client_models_per_edge: 每个边缘服务器的客户端模型列表
        edge_models: 所有边缘服务器的模型参数列表
        cloud_model: 云服务器全局模型
        epsilon: 收敛阈值

    返回:
        (聚合后的全局模型，边缘服务器权重列表，客户端权重列表（按边缘服务器分组）)
    """
    edge_client_weights = []

    # 1. 边缘服务器层级聚合
    for edge_idx, edge_client_models in enumerate(client_models_per_edge):
        if len(edge_client_models) > 0:
            _, truth_model = td_weight_estimation(
                models=edge_client_models,
                truth_model=edge_models[edge_idx],
                epsilon=epsilon
            )
            edge_models[edge_idx] = truth_model

    # 2. 云服务器层级聚合
    cloud_weights, cloud_model = td_weight_estimation(
        models=edge_models,
        truth_model=cloud_model,
        epsilon=epsilon
    )

    return cloud_model, cloud_weights, edge_client_weights


class TDReputationSystem:
    """
    基于 TD 算法的声誉系统

    用于管理客户端和边缘服务器的声誉值，并将其集成到 Stackelberg 博弈中
    """

    def __init__(
        self,
        epsilon: float = EPSILON_DEFAULT,
        alpha: float = 0.7,
        max_iterations: int = 100
    ):
        """
        初始化声誉系统

        参数:
            epsilon: TD 收敛阈值
            alpha: EMA 平滑系数
            max_iterations: TD 最大迭代次数
        """
        self.epsilon = epsilon
        self.alpha = alpha  # EMA 平滑系数
        self.max_iterations = max_iterations

    def update_client_reputation(
        self,
        client_models: List[Dict],
        edge_model: Dict,
        current_reputations: List[float]
    ) -> List[float]:
        """
        更新客户端声誉（边缘服务器执行）

        参数:
            client_models: 客户端模型列表
            edge_model: 边缘服务器模型（作为真值参考）
            current_reputations: 当前声誉列表

        返回:
            更新后的声誉列表
        """
        td_weights, _ = td_weight_estimation(
            models=client_models,
            truth_model=edge_model,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations
        )

        # EMA 平滑更新
        updated_reputations = []
        for i in range(len(current_reputations)):
            new_rep = self.alpha * td_weights[i] + (1 - self.alpha) * current_reputations[i]
            updated_reputations.append(new_rep)

        return updated_reputations

    def update_edge_reputation(
        self,
        edge_models: List[Dict],
        cloud_model: Dict,
        current_reputations: List[float]
    ) -> List[float]:
        """
        更新边缘服务器声誉（云服务器执行）

        参数:
            edge_models: 边缘服务器模型列表
            cloud_model: 云服务器全局模型（作为真值参考）
            current_reputations: 当前声誉列表

        返回:
            更新后的声誉列表
        """
        td_weights, _ = td_weight_estimation(
            models=edge_models,
            truth_model=cloud_model,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations
        )

        # EMA 平滑更新
        updated_reputations = []
        for i in range(len(current_reputations)):
            new_rep = self.alpha * td_weights[i] + (1 - self.alpha) * current_reputations[i]
            updated_reputations.append(new_rep)

        return updated_reputations

    def get_reputation_based_weights(
        self,
        models: List[Dict],
        reputations: List[float],
        truth_model: Dict
    ) -> List[float]:
        """
        获取基于声誉的聚合权重（用于 Stackelberg 博弈输入）

        参数:
            models: 模型列表
            reputations: 声誉列表
            truth_model: 真值模型

        返回:
            结合声誉和 TD 权重的混合权重列表
        """
        td_weights, _ = td_weight_estimation(
            models=models,
            truth_model=truth_model,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations
        )

        # 混合权重：结合 TD 权重和声誉值
        hybrid_weights = []
        for i in range(len(models)):
            # w_hybrid = TD 权重 × 声誉值
            hybrid_w = td_weights[i] * reputations[i]
            hybrid_weights.append(hybrid_w)

        # 归一化
        total = sum(hybrid_weights)
        if total > 0:
            hybrid_weights = [w / total for w in hybrid_weights]
        else:
            # 如果所有权重都为零，使用均匀权重
            hybrid_weights = [1.0 / len(models)] * len(models)

        return hybrid_weights


class StackelbergTDInput:
    """
    Stackelberg 博弈的 TD 输入生成器

    将 TD 声誉系统的输出转换为 Stackelberg 博弈的输入参数
    """

    def __init__(self, reputation_system: TDReputationSystem):
        """
        初始化输入生成器

        参数:
            reputation_system: TD 声誉系统实例
        """
        self.reputation_system = reputation_system

    def compute_client_input(
        self,
        client_models: List[Dict],
        edge_model: Dict,
        client_reputations: List[float],
        base_price: float
    ) -> Tuple[List[float], List[float]]:
        """
        计算 Stackelberg 博弈中客户端的输入

        参数:
            client_models: 客户端模型列表
            edge_model: 边缘服务器模型
            client_reputations: 客户端声誉列表
            base_price: 基础单价

        返回:
            (每个客户端的价格列表，每个客户端的贡献系数列表)
        """
        # 使用 TD 权重作为质量因子
        td_weights, _ = td_weight_estimation(
            models=client_models,
            truth_model=edge_model,
            epsilon=self.reputation_system.epsilon
        )

        # 计算每个客户端的价格 = 基础价格 × TD 权重 × 声誉
        prices = []
        contribution_ratios = []

        for i in range(len(client_models)):
            # 质量因子 = TD 权重 × 声誉
            quality_factor = td_weights[i] * client_reputations[i]
            # 价格与质量成正比
            price = base_price * quality_factor
            prices.append(price)
            # 贡献系数与声誉正相关
            contribution_ratios.append(min(1.0, client_reputations[i]))

        return prices, contribution_ratios

    def compute_edge_input(
        self,
        edge_models: List[Dict],
        cloud_model: Dict,
        edge_reputations: List[float],
        cloud_budget: float
    ) -> Tuple[List[float], Dict]:
        """
        计算 Stackelberg 博弈中边缘服务器的输入

        参数:
            edge_models: 边缘服务器模型列表
            cloud_model: 云服务器全局模型
            edge_reputations: 边缘服务器声誉列表
            cloud_budget: 云预算

        返回:
            (每个边缘服务器的收益分享列表，聚合后的全局模型)
        """
        # 使用 TD 权重计算边缘服务器的质量
        td_weights, aggregated_model = td_weight_estimation(
            models=edge_models,
            truth_model=cloud_model,
            epsilon=self.reputation_system.epsilon
        )

        # 计算每个边缘服务器的收益分享
        shares = []
        total_weight = sum(td_weights[i] * edge_reputations[i] for i in range(len(edge_models)))

        for i in range(len(edge_models)):
            # 分享与 (TD 权重 × 声誉) 成正比
            if total_weight > 0:
                share = cloud_budget * (td_weights[i] * edge_reputations[i]) / total_weight
            else:
                share = cloud_budget / len(edge_models)
            shares.append(share)

        return shares, aggregated_model
