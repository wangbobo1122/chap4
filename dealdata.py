import os
import pickle
import numpy as np
from PIL import Image
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 建议安装 tqdm 显示进度条: pip install tqdm

# 1. 配置路径
CINIC10_RAW_DIR = os.path.join('data', 'cinic-10')
CINIC10_BATCH_DIR = os.path.join('data', 'cinic-10-batches-py')
os.makedirs(CINIC10_BATCH_DIR, exist_ok=True)

# 2. 定义类别
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def read_single_image(args):
    """读取单张图片的辅助函数，用于多线程"""
    path, label = args
    try:
        with Image.open(path) as img:
            # 统一转为 RGB (防止部分图片是 RGBA 或 L 模式)
            img = img.convert('RGB')
            img_np = np.array(img).astype('uint8')  # (32, 32, 3)
            img_np = img_np.transpose(2, 0, 1)  # (3, 32, 32) CHW
            return img_np, label, os.path.basename(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def create_batch(file_list, batch_label, save_path):
    """将文件列表转换为 CIFAR-style 字典并保存"""
    data_list = []
    labels_list = []
    filenames_list = []

    # 使用多线程加速图片读取
    # max_workers 可以根据你的 CPU 核心数调整，IO 密集型任务可以设大一点
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(read_single_image, file_list),
                            total=len(file_list),
                            desc=f"Processing {batch_label}"))

    # 过滤读取失败的图片（如果有）
    results = [r for r in results if r is not None]

    # 解包数据
    # 注意：这里直接用 list comprehension 比 np.stack 更快，最后再 reshape
    data_list = [r[0] for r in results]
    labels_list = [r[1] for r in results]
    filenames_list = [r[2] for r in results]

    # 转换为 NumPy 数组
    data_np = np.stack(data_list)  # (N, 3, 32, 32)

    # 构造字典
    batch_data = {
        'data': data_np.reshape(len(data_np), -1),  # Flatten to (N, 3072)
        'labels': labels_list,
        'batch_label': batch_label,
        'filenames': filenames_list
    }

    with open(save_path, 'wb') as f:
        pickle.dump(batch_data, f)
    print(f"Saved {save_path}: {len(data_np)} images.")


def convert_cinic10_optimized():
    print("Step 1: 扫描所有文件路径...")

    train_valid_files = []  # 存放 (path, label_idx)
    test_files = []

    # 遍历所有文件夹收集路径
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(CINIC10_RAW_DIR, split)
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir): continue

            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.png')]

            # 标记数据归属
            file_objs = [(f, cls_idx) for f in files]

            if split == 'test':
                test_files.extend(file_objs)
            else:
                train_valid_files.extend(file_objs)  # 合并 train 和 valid

    print(f"扫描完成。训练+验证集: {len(train_valid_files)} 张, 测试集: {len(test_files)} 张")

    # ---------------------------------------------------------
    # 关键步骤：随机打乱！
    # ---------------------------------------------------------
    print("Step 2: 随机打乱数据 (Shuffling)...")
    random.seed(42)  # 固定种子保证可复现
    random.shuffle(train_valid_files)
    # 测试集也可以打乱，虽不影响训练，但影响评估的随机性
    random.shuffle(test_files)

    # ---------------------------------------------------------
    # Step 3: 生成 Training Batches (5个)
    # ---------------------------------------------------------
    total_train = len(train_valid_files)
    batch_size = total_train // 5

    for b in range(1, 6):
        start = (b - 1) * batch_size
        end = b * batch_size if b < 5 else total_train
        current_batch_files = train_valid_files[start:end]

        save_path = os.path.join(CINIC10_BATCH_DIR, f'data_batch_{b}')
        create_batch(current_batch_files, f'training batch {b}', save_path)

    # ---------------------------------------------------------
    # Step 4: 生成 Test Batch
    # ---------------------------------------------------------
    save_path = os.path.join(CINIC10_BATCH_DIR, 'test_batch')
    create_batch(test_files, 'testing batch 1', save_path)

    print("所有转换完成！")


# 执行转换
if __name__ == '__main__':
    # 只有当文件不存在时才运行
    if not os.path.exists(os.path.join(CINIC10_BATCH_DIR, 'data_batch_1')):
        convert_cinic10_optimized()
    else:
        print("Batch files already exist.")

    # 测试读取（复用你原来的读取函数，这里省略不写，直接调用即可）
    # ...