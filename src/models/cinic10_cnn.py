import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 保留原模型注释中用到的math，避免报错

num_classes = 10


# 仅修改类名为CINIC10_CNN，内部结构和CIFAR10_CNN完全一致
class CINIC10_CNN(nn.Module):
    def __init__(self):
        super(CINIC10_CNN, self).__init__()
        # 完全复用原CIFAR-10模型的层结构，参数无任何改动
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        # 完全保留原模型的注释代码，不做任何修改
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         stdv = 1. / math.sqrt(m.weight.size(1))
        #         m.weight.data.uniform_(-stdv, stdv)
        #         if m.bias is not None:
        #             m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # 前向传播逻辑和原CIFAR-10模型完全一致，无任何改动
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def get_model_size(self):
        # 修复原模型的bug：原代码中创建了新的CIFAR10_CNN实例，改为self（否则计算的是新实例参数）
        total_params = 0
        for name, param in self.named_parameters():  # 关键修改：self.named_parameters()
            layer_params = param.numel()
            total_params += layer_params
        total_params_kb = (total_params * 4) / 1024 / 1024
        return total_params_kb


if __name__ == '__main__':
    # 实例化CINIC10模型（结构和CIFAR10完全一致）
    model = CINIC10_CNN()

    # 打印各层参数量（和原CIFAR10模型输出一致）
    total_params = 0
    for name, param in model.named_parameters():
        layer_params = param.numel()
        total_params += layer_params
        print(f"{name}: {layer_params} parameters")

    # 打印总参数量（修复原模型bug后结果准确）
    total_params_mb = model.get_model_size()
    print(f"\nTotal model parameters: {total_params_mb:.2f} MB")

    # 验证输入输出兼容性（CINIC10和CIFAR10均为3x32x32输入）
    test_input = torch.randn(1, 3, 32, 32)  # 模拟单张CINIC10图像输入
    test_output = model(test_input)
    print(f"\nInput shape (CINIC10): {test_input.shape}")
    print(f"Output shape: {test_output.shape} (10分类，和CIFAR10一致)")