# 第四章实现说明

本代码保留第四章已经确定的机制：客户端与边缘两级真值发现、基于异常度的自适应遗忘因子、剩余能量驱动的动态成本，以及 Cloud–Edge–Client 三阶段逐轮定价。实验变体只关闭相应组件，不改变完整方案的定义。

逐函数、逐变量的训练执行说明见 [`CHAPTER4_EXECUTION_FLOW.md`](CHAPTER4_EXECUTION_FLOW.md)。
全量实验完成后可用 `plot_chapter4_results.py` 自动生成论文图；绘图依赖见 `requirements_plotting.txt`。

## 替换位置

将本压缩包中的文件覆盖到原工程同名位置：

- `main.py`
- `src/fed_server/hierarchical_server.py`
- `src/fed_MEC/MEC_server.py`
- `src/fed_client/Client_server.py`
- `src/attacks/byzantine_attacks.py` 和 `src/attacks/td_algorithm.py` 为原依赖文件，一并放入以保证导入路径完整。

代码仍依赖原工程中的 `getdata.py`、`src/models/model.py`、`src/optimizers/gd.py` 和数据划分工具。

## 公平对比模式

`experiment_variant` 选择聚合/信誉方法，`contribution_policy` 独立选择训练数据量：

- `auto`：保留完整方案和原消融逻辑；
- `fixed_budget`：所有方法使用相同的每轮样本预算，用于严格防御对比；
- `full`：使用能量允许的全量数据，只作为准确率上界。

公平防御实验的直接命令：

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

`never` clean 运行每个方法和 seed 只执行一次，并自动与同 seed 的各攻击运行配对。

## 单次运行

```bash
python main.py \
  --experiment_variant proposed \
  --attack_schedule delayed \
  --attack_start_round 30 \
  --n_client_byzantine 0.2 \
  --client_attack_method mean \
  --seed 30
```

`attack_start_round=30` 使用从零开始的内部轮次，因此论文图中的第 31 轮开始攻击。恶意客户端在实验开始时一次性抽取，后续轮次保持身份不变。

## 消融与基线开关

| `experiment_variant` | 含义 |
|---|---|
| `proposed` | 完整第四章方案 |
| `fixed_ema` | 将自适应遗忘因子替换为固定 `beta0` |
| `edge_td_only` | 只在边缘端执行客户端 TD |
| `cloud_td_only` | 只在云端执行边缘 TD |
| `fixed_cost` | 冻结初始成本系数 |
| `fixed_satisfaction` | 冻结满意度参数 `lambda0` |
| `fixed_price` | 固定云端/边缘单价 `--fixed_cloud_price`，客户端仍按 KKT 响应 |
| `hierfavg` | 样本量加权的层次 FedAvg；仍执行相同能量可行性检查 |
| `trimmed_mean` | 两层均使用坐标截尾均值；仍使用同一资源与支付执行器 |
| `coordinate_median` | 两层均使用坐标中位数聚合，不使用 TD 信誉 |
| `no_defense` | 无 TD、无鲁棒聚合、无 Stackelberg 的普通分层 FedAvg 基线 |

批量运行三个随机种子：

```bash
python run_chapter4_experiments.py --seeds 30 31 32 \
  --variants proposed fixed_ema edge_td_only cloud_td_only fixed_cost fixed_satisfaction hierfavg trimmed_mean
python analyze_chapter4_results.py
```

建议将 `mean`、`lie`、`scaling` 分开运行；Fang 与 IPM 对组规模和防御器有额外前提。程序只在 Fang/IPM 的小组规模不足时采用确定性符号翻转，并在 `round_metrics.attack_fallback` 中显式记录；其他攻击异常会直接终止实验。

## 输出指标

每次运行在 `results_chapter4/<run_name>/` 生成：

- `round_metrics.csv`：全局准确率/损失、攻击及回退状态、价格、满意度、样本访问量、能耗、支付、恶意支付占比、云效用、信誉均值、诚实节点误惩罚率、TD 迭代数和约束违反量。
- `client_metrics.csv`：每轮客户端信誉、即时评分、遗忘因子、异常度三个分量、贡献量、成本、计算/通信能耗、支付和效用。
- `edge_metrics.csv`：边缘信誉、真值模型变化、信誉偏差、价格、加权贡献、收入、客户端支出与效用。
- `summary.json`：最终精度、最好精度、末五轮精度、攻击后最大精度下降、恢复延迟、达到目标精度的轮次、累计能耗、累计支付、攻击后的恶意支付占比和信誉检测延迟。
- `global_model.pt`：最终全局模型，可用 `--save_checkpoint false` 关闭。

实验后执行：

```powershell
python analyze_chapter4_results.py --input_dir results_fair_defense
python plot_chapter4_results.py `
  --input_dir results_fair_defense `
  --attack_method mean `
  --schedule delayed `
  --contribution_policy fixed_budget
```

除常规汇总外，会生成 `paired_robustness_runs.csv`、`paired_robustness_summary.csv`、准确率—累计样本/能耗图和 clean-attack 配对下降图。

论文主表至少报告三个随机种子的均值和标准差。攻击实验应画出第 31 轮前后的准确率、恶意/诚实信誉、恶意支付占比；资源实验应报告累计能耗、剩余电量、有效贡献量以及 `max_energy_constraint_violation`（应始终为 0）。

## 与外部论文代码对比

RAIM、CSFL 等外部方法不能仅用一个布尔开关可靠复现。应分别接入其作者代码，并统一数据划分、模型、客户端—边缘固定关联、攻击者集合、训练预算与能量可行性执行器；最终将它们的逐轮结果转换成这里三张 CSV 的字段再比较。`hierfavg` 与 `trimmed_mean` 已提供可直接运行的内部参考线。

## 建议的实验顺序

1. **有效性主实验**：`proposed / hierfavg / trimmed_mean`，攻击比例 20%，第 31 轮开始，分别运行 `mean / lie / scaling`。
2. **模块消融**：固定 `mean` 攻击，对比 `fixed_ema / edge_td_only / cloud_td_only / fixed_cost / fixed_satisfaction`。
3. **攻击强度**：恶意客户端比例取 10%、20%、30%；边缘攻击单独实验，避免同时改变两个变量。
4. **异构性敏感性**：Dirichlet 参数取 0.1、0.2、0.5；观察 TD 对正常 Non-IID 更新的误惩罚率。
5. **资源压力**：改变 `initial_energy`、`round_deadline` 和客户端参与比例；检查能量约束、有效样本访问量及达到目标精度所需能耗。
6. **参数敏感性**：改变 `beta0`、`attack_sensitivity`、`lambda_gamma` 和 `price_grid_step`；价格网格同时报告最终细化步长与运行时间。

外部论文对比应放在主表，模块消融单独成表。RAIM最接近信誉激励主线；HierFAVG用于验证普通分层聚合；Trimmed Mean用于区分“动态信誉激励”和“单轮鲁棒聚合”的效果。CSFL是第三章已有方法，更适合作为跨章节衔接参考，不应被描述成第四章动态攻击方法的同类算法。

