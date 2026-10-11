# import numpy as np
#
# def split_imbalance(train_labels, number_of_clients):
#     each_client_data_number = int(len(train_labels) / number_of_clients)
#     np.random.seed(2024)
#     n_classes = train_labels.max() + 1
#     index_of_diff_class = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
#     print(index_of_diff_class)
#     index_own_by_clients = [[] for _ in range(number_of_clients)]
#     for i in range(number_of_clients):
#         assign_which_class = int(i / (number_of_clients / n_classes))
#         assign_which_class_area = int(i % (number_of_clients / n_classes))
#         index_own_by_clients[i].append(index_of_diff_class[assign_which_class] \
#                                        [assign_which_class_area * each_client_data_number: \
#                                         (assign_which_class_area + 1) * each_client_data_number])
#     index_own_by_clients = [np.array(idcs[0]) for idcs in index_own_by_clients]
#     return index_own_by_clients
import numpy as np
def split_random(train_labels, number_of_clients, seed=2025):
    """
    随机分配数据集给多个客户端（IID分配）
    Args:
        train_labels: 训练集的标签数组（形状 [n_samples]）
        number_of_clients: 客户端数量
        seed: 随机种子（保证可复现）
    Returns:
        index_own_by_clients: 列表，每个元素是分配给客户端的样本索引数组
    """
    np.random.seed(seed)
    # 获取所有样本的索引并打乱
    total_indices = np.arange(len(train_labels))
    np.random.shuffle(total_indices)  # 随机打乱

    # 计算每份的大小
    split_size = len(total_indices) // number_of_clients
    remainder = len(total_indices) % number_of_clients  # 余数

    # 均分索引
    index_own_by_clients = []
    start = 0
    for i in range(number_of_clients):
        end = start + split_size + (1 if i < remainder else 0)  # 处理余数
        client_indices = total_indices[start:end]
        index_own_by_clients.append(client_indices)
        start = end

    return index_own_by_clients

def split_imbalance(train_labels, number_of_clients, seed=2025):
    np.random.seed(seed)
    n_classes = train_labels.max() + 1
    index_of_diff_class = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # Create the each_client_data_number array
    array_length = number_of_clients // 10
    target_sum = len(index_of_diff_class[0])

    # Generate random integers that sum to target_sum
    if array_length > 0:
        # Create values using normal distribution that sum to target_sum
        mean = target_sum / array_length
        std_dev = mean * 0.3  # 30% of mean as standard deviation

        # Generate normal distribution values
        normal_values = np.random.normal(mean, std_dev, array_length)

        # Ensure all values are positive
        normal_values = np.maximum(normal_values, 1)

        # Normalize to sum to target_sum
        normal_values = normal_values / np.sum(normal_values) * target_sum
        each_client_data_number = normal_values.astype(int)

        # Adjust if there's any rounding difference
        current_sum = np.sum(each_client_data_number)
        if current_sum != target_sum:
            difference = target_sum - current_sum
            each_client_data_number[0] += difference
    else:
        each_client_data_number = np.array([target_sum])
    index_own_by_clients = [[] for _ in range(number_of_clients)]

    for i in range(number_of_clients):
        assign_which_class = int(i / (number_of_clients / n_classes))
        # Use the corresponding value from the array (cycling through if needed)
        data_number = each_client_data_number[i % len(each_client_data_number)]

        random_choose_index = np.random.choice(
            index_of_diff_class[assign_which_class][:],
            size=data_number,
            replace=False
        )
        index_own_by_clients[i] += random_choose_index.tolist()

    index_own_by_clients = [np.array(idcs) for idcs in index_own_by_clients]
    return index_own_by_clients




def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split  #
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst

def choose_two_digit(split_data_lst, uesr):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        if uesr == 199 or uesr == 198:
            lst = np.random.choice(available_digit, 2, replace=True).tolist()
        else:
            lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        print(available_digit)
    return lst

np.random.seed(2025)

def limit_class_samples(train_labels, target_class=None, max_samples=100, seed=2025):
    np.random.seed(seed)

    # Create a copy to avoid modifying original array
    modified_labels = train_labels.copy()

    # If target_class is not specified, choose randomly
    if target_class is None:
        target_class = np.random.randint(0, modified_labels.max() + 1)

    # Get indices of the target class
    class_indices = np.argwhere(modified_labels == target_class).flatten()

    # If already below limit, return original
    if len(class_indices) <= max_samples:
        return modified_labels

    # Randomly select samples to keep (others will be marked as -1)
    keep_indices = np.random.choice(class_indices, max_samples, replace=False)

    # Mark excess samples as -1
    excess_mask = (modified_labels == target_class) & ~np.isin(np.arange(len(modified_labels)), keep_indices)
    modified_labels[excess_mask] = -1

    return modified_labels


def split_imbalance_two(train_labels, number_of_clients,seed=2025):
    # Preserve the old method's strong label skew while guaranteeing that all
    # samples are assigned exactly once.  A low-alpha Dirichlet allocation is
    # the stable equivalent for arbitrary client/class counts.
    result, _ = dirichlet_split_noniid(train_labels, alpha=0.1, n_clients=number_of_clients, seed=seed)
    return result




def split(train_labels, number_of_clients, seed=2025):
    """
    Split data where:
    1. Each label is partitioned into number_of_clients equal parts
    2. Each client randomly selects 10 partitions from all available partitions (across all labels)
    3. Selected partitions are removed from the available pool
    Args:
        train_labels: Array of training labels (can be integers or other numeric types)
        number_of_clients: Number of clients to split into
        seed: Random seed for reproducibility

    Returns:
        client_idcs: List of numpy arrays where each array contains indices for a client's data
    """
    # The legacy partition-pool implementation left unused partitions behind.
    # Use a moderate Dirichlet allocation so method 4 has the same intended
    # heterogeneous-label behavior without dropping data.
    result, _ = dirichlet_split_noniid(train_labels, alpha=0.5, n_clients=number_of_clients, seed=seed)
    return result

def split_split(train_labels, number_of_clients, seed=2025):
    """
    Split train_labels into two groups:
    - First group uses split_average method (non-IID)
    - Second group uses random allocation (IID)

    Args:
        train_labels: Array of training labels
        number_of_clients: Total number of clients to split into
        seed: Random seed for reproducibility

    Returns:
        List of arrays where each array contains indices for a client's data
    """
    np.random.seed(seed)

    # Split clients into two groups (approximately half)
    clients_split1 = number_of_clients // 2
    clients_split2 = number_of_clients - clients_split1

    # Split the data indices into two parts
    total_indices = np.arange(len(train_labels))
    np.random.shuffle(total_indices)
    split_point = len(total_indices) // 2
    part1_indices = total_indices[:split_point]
    part2_indices = total_indices[split_point:]

    # First group uses split_average method
    part1_labels = train_labels[part1_indices]
    split_avg_idcs = split_imbalance_two(part1_labels, clients_split1, seed)

    # Second group uses random allocation (IID)
    part2_labels = train_labels[part2_indices]
    random_idcs = split_random(part2_labels, clients_split2, seed)

    # Combine the results by mapping back to original indices
    combined_client_idcs = []

    # Add split_average results (part1)
    for client_idcs in split_avg_idcs:
        combined_client_idcs.append(part1_indices[client_idcs])

    # Add random allocation results (part2)
    for client_idcs in random_idcs:
        combined_client_idcs.append(part2_indices[client_idcs])

    return combined_client_idcs


def split_average(train_labels, number_of_clients, seed=2025):
    """
    Split data into groups where each group contains only specific labels (2-7)
    and has significantly different data sizes.

    Args:
        train_labels: Array of training labels
        number_of_clients: Number of groups to create
        seed: Random seed for reproducibility

    Returns:
        List of arrays where each array contains indices for a client's data
    """
    np.random.seed(seed)
    n_classes = train_labels.max() + 1
    index_own_by_clients = [[] for _ in range(number_of_clients)]

    # Get indices for each class
    class_indices = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    for client in range(number_of_clients):
        # Randomly choose number of labels for this client (between 2-7)
        num_labels = np.random.randint(2, 8)

        # Randomly select which labels this client will have
        selected_labels = np.random.choice(n_classes, num_labels, replace=False)

        # Determine data size for this client (random between 0.5x to 3x average)
        avg_size = len(train_labels) / number_of_clients
        min_size = int(avg_size * 0.5)
        max_size = int(avg_size * 3)
        target_size = np.random.randint(min_size, max_size + 1)

        # Collect indices from selected labels until we reach target size
        collected_indices = []
        remaining_size = target_size

        # Shuffle the order of selected labels to randomize which classes contribute more
        np.random.shuffle(selected_labels)

        for label in selected_labels:
            indices = class_indices[label]
            if len(indices) == 0:
                continue

            # Take up to remaining_size samples from this class
            take = min(remaining_size, len(indices))
            selected = np.random.choice(indices, take, replace=False)
            collected_indices.extend(selected)
            remaining_size -= take

            if remaining_size <= 0:
                break

        # If we didn't get enough samples, try to get more from any of the selected labels
        if remaining_size > 0:
            while remaining_size > 0:
                for label in selected_labels:
                    indices = class_indices[label]
                    available = np.setdiff1d(indices, collected_indices)
                    if len(available) == 0:
                        continue

                    take = min(remaining_size, len(available))
                    selected = np.random.choice(available, take, replace=False)
                    collected_indices.extend(selected)
                    remaining_size -= take

                    if remaining_size <= 0:
                        break

        index_own_by_clients[client] = np.array(collected_indices)

    return index_own_by_clients

def split_iid(train_labels, number_of_clients, seed=2025):
    """
    将数据集平均分配给多个客户端（IID分配）
    Args:
        train_labels: 训练集的标签数组（形状 [n_samples]）
        number_of_clients: 客户端数量
        seed: 随机种子（保证可复现）
    Returns:
        index_own_by_clients: 列表，每个元素是分配给客户端的样本索引数组
    """
    np.random.seed(seed)
    # 1. 获取所有样本的索引并打乱
    total_indices = np.arange(len(train_labels))
    np.random.shuffle(total_indices)  # 随机打乱
    # 2. 计算每份的大小
    split_size = len(total_indices) // number_of_clients
    remainder = len(total_indices) % number_of_clients  # 余数
    # 3. 均分索引
    index_own_by_clients = []
    start = 0
    for i in range(number_of_clients):
        end = start + split_size + (1 if i < remainder else 0)  # 处理余数
        client_indices = total_indices[start:end]
        index_own_by_clients.append(client_indices)
        start = end
    return index_own_by_clients

def _validate_partition(client_idcs, n_samples, number_of_clients):
    """Check the invariant required by all Chapter 4 splitters."""
    if len(client_idcs) != number_of_clients:
        raise ValueError("splitter returned an incorrect number of clients")
    flat = np.concatenate([np.asarray(v, dtype=np.int64) for v in client_idcs]) if client_idcs else np.empty(0, dtype=np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= n_samples):
        raise ValueError("splitter produced an out-of-range sample index")
    if len(np.unique(flat)) != flat.size:
        raise ValueError("splitter produced duplicate sample indices")
    if flat.size != n_samples:
        raise ValueError(f"splitter lost samples: assigned {flat.size}, expected {n_samples}")


def split_NIID(train_labels, alpha, number_of_clients, seed=2025):
    """Mix an IID quarter with a Dirichlet three-quarter partition."""
    if number_of_clients < 2:
        return [np.arange(len(train_labels), dtype=np.int64)]
    rng = np.random.default_rng(seed)
    total_indices = np.arange(len(train_labels), dtype=np.int64)
    rng.shuffle(total_indices)
    split_point = len(total_indices) // 4
    part1_indices, part2_indices = total_indices[:split_point], total_indices[split_point:]
    clients_part1 = max(1, number_of_clients // 4)
    clients_part2 = number_of_clients - clients_part1
    iid_idcs = split_iid(train_labels[part1_indices], clients_part1, int(seed) + 1)
    noniid_idcs, _ = dirichlet_split_noniid(
        train_labels[part2_indices], alpha, clients_part2, seed=int(seed) + 2
    )
    result = [part1_indices[np.asarray(v, dtype=np.int64)] for v in iid_idcs]
    result.extend(part2_indices[np.asarray(v, dtype=np.int64)] for v in noniid_idcs)
    _validate_partition(result, len(train_labels), number_of_clients)
    return result


def dirichlet_split_noniid(train_labels, alpha, n_clients, seed=2025):
    """Partition every sample exactly once using a Dirichlet class allocation.

    The old implementation pre-added one random sample to each client and then
    split the complete class indices, duplicating those samples.  This version
    only cuts each shuffled class index array into mutually exclusive pieces.
    """
    labels = np.asarray(train_labels)
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1 if labels.size else 0
    client_idcs = [[] for _ in range(n_clients)]
    if not labels.size:
        return [np.empty(0, dtype=np.int64) for _ in range(n_clients)], [[] for _ in range(n_clients)]
    for class_id in range(n_classes):
        class_indices = np.flatnonzero(labels == class_id).astype(np.int64)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(n_clients, float(alpha)))
        cut_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        for client_id, piece in enumerate(np.split(class_indices, cut_points)):
            client_idcs[client_id].extend(piece.tolist())
    result = [np.asarray(indices, dtype=np.int64) for indices in client_idcs]
    # Keep non-empty clients when the data set is large enough, without
    # duplicating a sample.  Moving an existing sample preserves the partition.
    if len(labels) >= n_clients:
        empty = [i for i, indices in enumerate(result) if len(indices) == 0]
        for target in empty:
            donor = int(np.argmax([len(indices) for indices in result]))
            if len(result[donor]) <= 1:
                break
            result[target] = np.append(result[target], result[donor][-1])
            result[donor] = result[donor][:-1]
    _validate_partition(result, len(labels), n_clients)
    return result, [indices.tolist() for indices in result]
