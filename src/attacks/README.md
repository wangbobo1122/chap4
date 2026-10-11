# 拜占庭攻击模块使用说明

## 概述

本模块集成了多种针对联邦学习的拜占庭攻击方法，可在边缘服务器和云服务器两个层级注入攻击。

## 攻击方法

### 1. IPM (Inner Product Manipulation)
- **目标**: 针对 Krum/Trimmed-mean 防御机制
- **原理**: 最大化聚合结果与原始均值的距离
- **配置**: `attack_method='ipm'`

### 2. Fang 攻击
- **目标**: 针对 Krum 防御机制
- **原理**: 逐步减小 λ 直到能欺骗 Krum
- **配置**: `attack_method='fang'`

### 3. LIE (Little Is Enough)
- **目标**: 隐藏在统计分布边缘
- **原理**: 恶意更新略低于均值若干个标准差
- **配置**: `attack_method='lie'`

### 4. 简单攻击
| 攻击方法 | 配置 | 原理 |
|---------|------|------|
| Scaling | `attack_method='scaling'` | 放大梯度 (factor=10) |
| Mean | `attack_method='mean'` | 梯度反转 |
| Full Mean | `attack_method='full_mean'` | 完全抵消良性更新 |

## 配置方式

### 边缘服务器攻击配置

在 `options` 字典中添加以下配置：

```python
options = {
    # ... 其他配置 ...

    # 边缘服务器攻击配置
    'is_byzantine': True,          # 是否为恶意边缘服务器
    'attack_method': 'fang',       # 攻击方法：'fang', 'ipm', 'lie', 'scaling', 'mean', 'full_mean'
    'n_byzantine': 0.3,            # 恶意客户端比例 (0-1) 或具体数量
}
```

### 云服务器攻击配置

在 `options` 字典中添加以下配置：

```python
options = {
    # ... 其他配置 ...

    # 云服务器攻击配置
    'cloud_is_byzantine': True,     # 云服务器是否被攻击
    'cloud_attack_method': 'lie',   # 攻击方法
    'n_cloud_byzantine': 2,         # 恶意边缘服务器数量
}
```

## 使用示例

### 示例 1: 边缘服务器注入 Fang 攻击

```python
from src.fed_server.hierarchical_server import Hierarchical_server

options = {
    'num_of_clients': 100,
    'num_of_MEC': 10,
    'round_num': 100,
    # ... 其他配置 ...

    # 配置 30% 的客户端为恶意
    'is_byzantine': True,
    'attack_method': 'fang',
    'n_byzantine': 0.3,
}

server = Hierarchical_server(options, dataset, clients_label)
server.train()
```

### 示例 2: 云服务器注入 LIE 攻击

```python
options = {
    'num_of_clients': 100,
    'num_of_MEC': 10,
    'round_num': 100,
    # ... 其他配置 ...

    # 配置 2 个恶意边缘服务器
    'cloud_is_byzantine': True,
    'cloud_attack_method': 'lie',
    'n_cloud_byzantine': 2,
}

server = Hierarchical_server(options, dataset, clients_label)
server.train()
```

### 示例 3: 同时测试多种攻击方法

```python
attack_methods = ['fang', 'ipm', 'lie', 'scaling', 'mean', 'full_mean']

for method in attack_methods:
    options['attack_method'] = method
    options['is_byzantine'] = True
    options['n_byzantine'] = 0.2

    server = Hierarchical_server(options, dataset, clients_label)
    server.train()

    # 记录攻击效果
    print(f"Attack: {method}, Final Accuracy: {server.metrics.acc_on_g_test_data[-1]}")
```

## 攻击效果评估

攻击后，可通过以下指标评估攻击效果：

1. **测试准确率下降**: 比较攻击前后模型的测试准确率
2. **收敛速度**: 观察模型收敛是否变慢
3. **防御方法对比**: 比较不同聚合规则 (Mean/Median/Krum/Trimmed-mean) 的鲁棒性

## 文件结构

```
src/
├── attacks/
│   ├── __init__.py              # 模块初始化
│   └── byzantine_attacks.py     # 攻击方法实现
├── fed_MEC/
│   └── MEC_server.py            # 边缘服务器 (已集成攻击)
└── fed_server/
    └── hierarchical_server.py   # 云服务器 (已集成攻击)
```

## 核心 API

### 攻击函数

```python
from src.attacks.byzantine_attacks import (
    get_malicious_updates,      # 统一接口获取恶意更新
    fang_attack,                # Fang 攻击
    ipm_attack,                 # IPM 攻击
    lie_attack,                 # LIE 攻击
    scaling_attack,             # Scaling 攻击
    mean_attack,                # Mean 攻击
    full_mean_attack,           # Full Mean 攻击
    trimmed_mean,               # Trimmed Mean 聚合 (用于测试)
    multi_krum,                 # Multi-Krum 聚合 (用于测试)
)
```

### 使用统一接口

```python
malicious_updates = get_malicious_updates(
    all_updates=all_updates,      # 所有良性客户端的更新
    model_re=model_re,            # 全局模型参数
    n_attackers=n_attackers,      # 攻击者数量
    attack_method='fang'          # 攻击方法
)
```

## 注意事项

1. **攻击时机**: 攻击在聚合阶段注入，不影响客户端本地训练
2. **全知假设**: 攻击者假设知道所有良性更新的统计信息
3. **攻击比例**: 建议攻击比例不超过 30%，否则过于明显
4. **防御测试**: 可结合不同防御算法测试鲁棒性
