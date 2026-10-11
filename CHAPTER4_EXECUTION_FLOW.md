# 第四章代码执行流程

本文档对应实验文档第四章“动态攻击与资源衰减下的三层信誉自适应激励机制”。实现入口为 `main.py`，训练主循环为 `src/fed_server/hierarchical_server.py` 中的 `Hierarchical_server.train()`。

## 1. 启动阶段

1. `main.input_options(argv)` 解析参数并检查数据集、模型、客户端数、MEC 数、能量和带宽是否合法。
2. `setup_seed(seed)` 统一设置 Python、NumPy 和 PyTorch 随机种子。
3. `GetDataSet(dataset_name)` 从项目 `data/` 目录读取并归一化数据。
4. `choose_model(options)` 创建模型；`model_payload_bits(model)` 按实际传输的完整 `state_dict` 计算 `|theta|`，单位为 bit。
5. 数据划分器为每个客户端生成互斥索引。Dirichlet 划分保证索引不重复、不丢失，并受 `--seed` 控制。
6. `Hierarchical_server.__init__()` 创建云模型、MEC、客户端及其独立模型/优化器，固定攻击者集合，并初始化信誉、能量、位置和带宽。

`input_options()` 的输出为 `(client_label, options, dataset)`：

- `client_label[i]`：客户端 `i` 拥有的训练样本索引；
- `options`：全部实验参数；
- `dataset`：训练/测试数据及标签。

## 2. `train()` 每轮完整流程

### 步骤 0：初始评估

`evaluate_global()` 在测试集上计算随机初始化模型的准确率和交叉熵损失，作为攻击前和训练前基线。

### 步骤 1：冻结上一轮信息并选择参与者

- `round_index`：从 0 开始的全局轮次 `t`；
- `global_reference`：本轮开始时的全局模型 `Theta_global(t-1)`；
- `reputation_before`：本轮定价使用的客户端历史信誉 `H_i(t-1)`；
- `attack_active()`：按 always/delayed/intermittent/never 判断本轮是否攻击；
- `_select_participants()`：按 `c_fraction` 从每个 MEC 的固定客户端组中抽取参与者。

对每个参与客户端，`update_dynamic_cost()` 先计算：

- `E_comm = P_tx * |theta| / (B * log2(1 + SNR))`；
- `Q_max = floor((E_remaining - E_comm) / (kappa*f^2*C))`；
- `c_i = xi / (E_remaining - E_comm)`。

`Q_max` 是当前剩余能量允许的最大样本访问数，`c_i` 是客户端本轮单位边际成本。

### 步骤 2：三层 Stackelberg 逆向求解

`_stage1_cloud_price()` 依次调用各层最优响应：

1. `_satisfaction(t)` 得到 `lambda(t)=max(lambda0*gamma^t, lambda_min)`；
2. `_edge_response(edge, R)` 使用客户端 KKT 三段响应 `q_i*=clip(p_j*H_i/c_i, 0, Q_max,i)`，遍历饱和分段并求边缘最优价格；
3. 无约束候选使用文档式 (4.37) 的闭式 `R*`；
4. 有能量约束时，`_constrained_cloud_price()` 根据当前饱和集合计算 `A、B`，使用式 (4.52) 求广义 KKT 候选；
5. 有界网格及局部细化用于验证分段边界，并选择云效用最大的可行候选。

输出保存在：

- `global_base_price`：云端基础单价 `R(t)`；
- `_planned_prices[edge_id]`：边缘对客户端单价 `p_j(t)`；
- `_planned_quantities[client_id]`：客户端连续最优贡献量，执行时向下取整为样本访问数。

#### 贡献策略与聚合方法解耦

`--contribution_policy` 决定客户端训练多少数据，`--experiment_variant` 独立决定聚合、信誉和消融机制：

| 贡献策略 | 含义 |
|---|---|
| `auto` | 向后兼容；完整方案/消融使用 Stackelberg，`no_defense/hierfavg` 使用全量训练 |
| `stackelberg` | 显式使用定价与 KKT 贡献量 |
| `fixed_budget` | 所有方法共用每轮样本预算，用于严格防御对比 |
| `full` | 使用能量允许的最大训练量，用作准确率上界 |

`fixed_budget` 使用 `q_i(t)=min(Q_max,i(t), floor(Q_round*n_i/sum_k(n_k)))`。整数余额使用确定性最大余数法分配；某个客户端触及能量上限后，剩余预算会继续分配给其他可行客户端。

### 步骤 3：客户端本地训练与边缘端 TD

每个 MEC 先接收 `global_reference`。对每个 `local_step`：

1. `_stage3_client_contribution()` 将本轮计划贡献量分配到本地聚合步骤；
2. `client.local_train()` 调用 `local_update()`；
3. `local_update()` 精确抽取 `q_i` 次样本访问，执行前向传播、交叉熵、反向传播和优化器更新；
4. 它输出完整本地 `state_dict` 和统计字典，并保存本步骤按样本加权的平均真实梯度；
5. `_inject_attacks()` 对恶意客户端上传的模型更新实施攻击；`_inject_gradient_attacks()` 同步修改其上传梯度信号；
6. `_truth_discovery()` 迭代计算距离、原始对数权重和真值模型，使用绝对条件 `||truth_new-truth_old||_2 < epsilon_td` 停止；
7. TD 输出的最大值归一化权重作为 `w_i(t)`，聚合模型作为 `Theta_truth,j(t)`。

当 `local_num>1` 时，本地训练和边缘聚合可重复，但客户端信誉不会重复更新。代码汇总整轮梯度和损失，仅在全部本地步骤结束后执行一次 `_update_client_reputations()`。

客户端异常度变量为：

- `gradient_change = ||grad_i(t)-grad_i(t-1)||_2`；
- `loss_change = |L_i(t)-L_i(t-1)|`；
- `reputation_deviation = |H_i(t-1)-w_i(t)|`；
- `sigma`：上述有效分量分别除以本轮全局中位数后直接相加；
- `beta = beta0*exp(-attack_sensitivity*sigma)`；
- `history_reputation = beta*H_old+(1-beta)*w_i(t)`。

### 步骤 4：能量、支付和边缘效用结算

`settle_energy()` 按实际训练样本数和实际上传次数扣除计算/通信能量，若超过剩余能量立即抛错。

- 客户端式 (4.22)：`U_client = p_j*H_i*q_i - 0.5*c_i*q_i^2`；
- 边缘式 (4.25)：收入为 `R*H_cloud,j*sum(H_i*q_i)`，支出为 `p_j*sum(q_i)`。

PDF 的客户端收入与边缘支出定义并不守恒，代码分别严格按式 (4.22) 和式 (4.25) 记录。因此 `client_metrics.payment` 与 `edge_metrics.client_expenditure` 是两个不同口径。

### 步骤 5：云端 TD 与边缘信誉

1. 所有有效 MEC 上传边缘真值模型；
2. 若配置边缘攻击，先修改云端实际收到的边缘模型；
3. 云端 `_truth_discovery()` 输出 `W_j(t)` 和全局真值模型；
4. `_update_edge_reputations()` 使用云端连续两轮实际收到的边缘模型变化与 `|H_cloud,j(t-1)-W_j(t)|` 计算 `sigma_cloud,j(t)`；
5. 每个边缘信誉在一个全局轮次中只更新一次；单边缘时 TD 权重为 1，仍执行 EMA；
6. `set_model_parameters()` 将全局模型更新为云端 TD 真值。

### 步骤 6：评估、记录和收尾

每轮重新调用 `evaluate_global()`，计算云效用、累计能耗、累计支付、恶意支付比例、信誉检测率和能量约束违反量，并写入内存记录。

`round_metrics.csv` 还会记录本轮计划样本量 `planned_samples`、固定预算 `round_sample_budget` 和实际累计样本量 `cumulative_used_samples`。

训练全部结束后：

- `_write_csv()` 写出 `round_metrics.csv`、`client_metrics.csv`、`edge_metrics.csv`；
- `_save_results()` 写出 `summary.json`；
- 若 `save_checkpoint=true`，写出 `global_model.pt`。

## 3. 关键函数输入输出

| 函数 | 主要输入 | 输出/副作用 |
|---|---|---|
| `model_payload_bits` | PyTorch 模型 | 完整状态载荷 bit 数 |
| `dirichlet_split_noniid` | 标签、alpha、客户端数、seed | 每客户端互斥索引列表及其普通列表形式 |
| `max_feasible_samples` | 当前能量、链路和硬件参数 | `Q_max` |
| `update_dynamic_cost` | 当前能量、`E_comm` | `c_i(t)` |
| `plan_contribution` | 连续计划贡献量 | 截断后的整数 `q_i` |
| `local_update` | 轮次、本地步骤、`q_i` | 本地模型、loss/accuracy/样本数，并保存平均梯度 |
| `_truth_discovery` | 模型列表、共同参考模型 | 即时权重、真值模型、迭代次数 |
| `_update_client_reputations` | 客户端列表、`w_i(t)` | 更新 `sigma/beta/H_i` 和梯度/损失历史 |
| `_update_edge_reputations` | 边缘列表、收到的模型、`W_j(t)` | 更新 `sigma_cloud/beta/H_cloud` 和模型历史 |
| `_edge_response` | MEC、云价格 `R` | `p_j`、各客户端 `q_i`、有效贡献和边缘效用 |
| `_allocate_fixed_budget` | 活跃客户端、每轮总预算 | 按本地数据量分配且满足能量上限的 `q_i` |
| `_stage1_cloud_price` | 活跃客户端、轮次 | `R*`，并缓存 `p_j*、q_i*` |
| `evaluate_global` | 当前全局模型、测试集 | `(accuracy, mean_loss)` |

## 4. 推荐运行命令

```powershell
python main.py --dataset_name mnist --model_name mnist_cnn --gpu -1 `
  --round_num 100 --num_of_clients 50 --num_of_MEC 5 `
  --experiment_variant proposed --attack_schedule delayed `
  --attack_start_round 30 --n_client_byzantine 0.2
```

`attack_start_round=30` 表示内部第 0 到 29 轮不攻击，界面显示的第 31 轮开始攻击。

## 5. 可直接执行的对比实验

所有方法都使用同一个 `seed` 时，数据划分、客户端攻击者集合和边缘攻击者集合保持一致。推荐先用 2 轮冒烟，再用完整轮数。

### 5.1 先验证单次实验

```powershell
python main.py --dataset_name mnist --model_name mnist_cnn --gpu -1 `
  --round_num 2 --local_epoch 1 --num_of_clients 10 --num_of_MEC 2 `
  --attack_schedule never --client_is_byzantine false `
  --run_name smoke_mnist
```

### 5.2 固定预算防御主实验

```powershell
python run_chapter4_experiments.py `
  --seeds 30 31 32 `
  --variants proposed no_defense trimmed_mean coordinate_median `
  --attack_methods mean lie scaling full_mean `
  --contribution_policy fixed_budget `
  --round_sample_budget 10000 `
  --schedules never delayed `
  --output_dir results_fair_defense `
  -- --gpu -1 --round_num 100 --local_epoch 5 --num_of_clients 50 --num_of_MEC 5 `
     --attack_start_round 30 --n_client_byzantine 0.2
```

四种方法每轮共用 10000 次样本访问预算（除非能量硬约束已无法满足）。`never` 每个方法和 seed 只运行一次，该 clean 结果可与同 seed 的所有攻击方法配对。

### 5.3 完整机制消融实验

```powershell
python run_chapter4_experiments.py `
  --seeds 30 31 32 `
  --variants proposed fixed_ema edge_td_only cloud_td_only fixed_cost fixed_satisfaction fixed_price `
  --attack_methods mean lie scaling `
  --contribution_policy auto `
  --schedules never delayed `
  --output_dir results_mechanism_ablation `
  -- --gpu -1 --round_num 100 --local_epoch 5 --num_of_clients 50 --num_of_MEC 5 `
     --attack_start_round 30 --n_client_byzantine 0.2
```

### 5.4 全数据准确率上界

```powershell
python run_chapter4_experiments.py `
  --seeds 30 31 32 `
  --variants no_defense `
  --attack_methods mean `
  --contribution_policy full `
  --schedules never delayed `
  --output_dir results_full_data_upper_bound `
  -- --gpu -1 --round_num 100 --local_epoch 5 --num_of_clients 50 --num_of_MEC 5 `
     --attack_start_round 30 --n_client_byzantine 0.2
```

论文中应将这个结果标为 `FedAvg-Full`，只作为全数据准确率上界，不用它与动态贡献方法直接比较同轮准确率。

其中：

- `proposed`：完整机制；
- `no_defense`：普通分层 FedAvg，无 TD、无鲁棒聚合、无 Stackelberg；
- `hierfavg`：样本量加权分层 FedAvg；
- `trimmed_mean`：两层坐标截尾均值；
- `coordinate_median`：两层坐标中位数；
- `fixed_ema`：固定遗忘因子；
- `edge_td_only`/`cloud_td_only`：单层 TD 消融；
- `fixed_cost`/`fixed_satisfaction`：资源成本和满意度消融；
- `fixed_price`：固定 `--fixed_cloud_price`，客户端仍使用 KKT 响应。

### 5.5 边缘攻击实验

```powershell
python run_chapter4_experiments.py `
  --seeds 30 31 32 `
  --variants proposed no_defense cloud_td_only hierfavg `
  --attack_methods mean scaling `
  --output_dir results_edge_attack `
  -- --gpu -1 --round_num 100 --local_epoch 5 --num_of_clients 50 --num_of_MEC 5 `
     --client_is_byzantine false --edge_is_byzantine true `
     --n_edge_byzantine 0.2 --attack_schedule delayed --attack_start_round 30
```

### 5.6 资源衰减实验

固定方法和攻击设置，只改变 `--initial_energy`，例如：

```powershell
python main.py --dataset_name mnist --model_name mnist_cnn --gpu -1 `
  --experiment_variant proposed --initial_energy 5000 `
  --round_num 100 --num_of_clients 50 --num_of_MEC 5 `
  --attack_schedule never --run_name energy_5000
```

建议重复运行 `5000、10000、25000`，比较有效样本量、剩余电量、能耗和最终精度。

## 6. 汇总 CSV 和批量生成图片

所有运行结束后，在项目根目录执行：

```powershell
python analyze_chapter4_results.py --input_dir results_fair_defense
python plot_chapter4_results.py `
  --input_dir results_fair_defense `
  --attack_method mean `
  --schedule delayed `
  --contribution_policy fixed_budget
```

汇总脚本生成：

- `aggregate_summary.csv`：按方法、贡献策略、预算、攻击和日程分组的均值/标准差；
- `paired_robustness_runs.csv`：每个 seed 的 clean-attack 净精度下降；
- `paired_robustness_summary.csv`：配对下降的多 seed 均值/标准差。

图片默认写入 `results_fair_defense/plots/`，共 12 张。新增的 `11_accuracy_samples_energy.png` 比较准确率—累计样本访问量和准确率—累计能耗；`12_paired_attack_degradation.png` 比较各方法相对同 seed clean 运行的平均/最大净精度下降。
