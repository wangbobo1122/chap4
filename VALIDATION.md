# 验证记录

- `python -m compileall -q main.py getdata.py src` 通过。
- `python -m unittest discover -s tests -v` 的 7 项测试通过，覆盖模型 bit 载荷、全部 CLI 数据划分不变量、单节点 TD、自适应 EMA、式 (4.28) 边缘定价和能耗/Qmax 公式。
- 已完成 1 轮无攻击 CPU 冒烟；训练、TD、能量结算、评估和三张 CSV 均正常。
- 已完成 2 轮、4 客户端、1 MEC、`local_num=2`、25% 客户端攻击的 CPU 冒烟；整轮梯度/损失汇总与“每全局轮只更新一次信誉”正常。
- 已完成 2 轮、4 客户端、2 MEC、50% 边缘攻击的 CPU 冒烟；云端 TD、边缘模型变化和边缘信誉记录正常。
- 两次冒烟的 `max_energy_constraint_violation` 均为 0，结果中未出现 NaN/Inf。
- 当前环境为 Python 3.13.11、PyTorch 2.10.0 CPU；主流程不再要求 torchvision。

