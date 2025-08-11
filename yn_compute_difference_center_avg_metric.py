import json
import numpy as np
from pathlib import Path

# 基础路径（根据实际情况调整）
base_path = Path(
    '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task132_externalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1')

# 初始化指标存储：合并数据 + 每个fold的独立数据
metrics = {
    'Combined': {
        'Dice': [],
        'Jaccard': [],
        'Precision': [],
        'Recall': [],
        'Hausdorff Distance 95': []
    }
}

# 为每个fold创建单独的指标存储
for fold in range(5):
    metrics[f'fold_{fold}'] = {
        'Dice': [],
        'Jaccard': [],
        'Precision': [],
        'Recall': [],
        'Hausdorff Distance 95': []
    }

# 遍历所有折叠（0-4）
for fold in range(5):
    json_path = base_path / f'fold_{fold}/validation_raw/summary.json'

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"正在处理 fold_{fold}...")
        valid_samples = 0

        for result in data['results']['all']:
            category_1 = result.get('1', {})
            if not category_1:
                continue

            reference = result.get('reference', '')
            if not reference:
                continue

            filename = reference.split('/')[-1]

            # 筛选以YN开头的样本
            if filename.startswith('YN'):
                dice = category_1.get('Dice', 0.0)
                if not (0.0 <= dice <= 1.0):
                    continue

                # 添加到当前fold和合并列表
                metrics[f'fold_{fold}']['Dice'].append(dice)
                metrics[f'fold_{fold}']['Jaccard'].append(category_1['Jaccard'])
                metrics[f'fold_{fold}']['Precision'].append(category_1['Precision'])
                metrics[f'fold_{fold}']['Recall'].append(category_1['Recall'])
                metrics[f'fold_{fold}']['Hausdorff Distance 95'].append(category_1['Hausdorff Distance 95'])

                metrics['Combined']['Dice'].append(dice)
                metrics['Combined']['Jaccard'].append(category_1['Jaccard'])
                metrics['Combined']['Precision'].append(category_1['Precision'])
                metrics['Combined']['Recall'].append(category_1['Recall'])
                metrics['Combined']['Hausdorff Distance 95'].append(category_1['Hausdorff Distance 95'])

                valid_samples += 1

        print(f"fold_{fold} 处理完成，找到 {valid_samples} 个有效样本")

    except FileNotFoundError:
        print(f"警告: fold_{fold} 的JSON文件未找到，跳过此折叠")
        continue
    except Exception as e:
        print(f"错误: 处理 fold_{fold} 时发生异常: {e}")
        continue


# 计算单折叠内样本统计的函数（保留用于显示单折叠详情）
def calculate_single_fold_stats(metrics_list):
    if not metrics_list:
        return {'mean': 0.0, 'std': 0.0, 'count': 0}
    arr = np.array(metrics_list)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),
        'count': len(arr)
    }

# 计算五折之间统计的函数（核心修改点）
def calculate_cross_fold_stats(metrics_dict, metric_name):
    """计算五个折叠间的均值和标准差"""
    fold_means = []
    valid_folds = 0
    for fold in range(5):
        fold_key = f'fold_{fold}'
        if fold_key in metrics_dict and metrics_dict[fold_key][metric_name]:
            fold_mean = np.mean(metrics_dict[fold_key][metric_name])
            fold_means.append(fold_mean)
            valid_folds += 1
    if not fold_means:
        return {'cross_fold_mean': 0.0, 'cross_fold_std': 0.0, 'valid_folds': 0}
    return {
        'cross_fold_mean': np.mean(fold_means),  # 五折的平均均值
        'cross_fold_std': np.std(fold_means, ddof=1),  # 五折均值的标准差
        'valid_folds': valid_folds
    }


# 输出结果
print("\n==== 五折交叉验证统计结果 ====")
metrics_to_analyze = ['Dice', 'Jaccard', 'Precision', 'Recall', 'Hausdorff Distance 95']

# 1. 显示五折之间的统计（核心需求）
for metric in metrics_to_analyze:
    stats = calculate_cross_fold_stats(metrics, metric)
    if stats['valid_folds'] > 0:
        print(f"{metric}:")
        print(f"  五折平均均值: {stats['cross_fold_mean']:.4f}")
        print(f"  五折均值标准差: {stats['cross_fold_std']:.4f}")
        print(f"  有效折叠数: {stats['valid_folds']}/5")
    else:
        print(f"{metric}: 无有效数据")

# 2. 显示各折叠的详细统计（可选，用于参考）
print("\n==== 各折叠详细统计 ====")
for fold in range(5):
    fold_key = f'fold_{fold}'
    if fold_key in metrics:
        fold_data = metrics[fold_key]
        print(f"\n{fold_key}:")
        for metric in metrics_to_analyze:
            stats = calculate_single_fold_stats(fold_data[metric])
            if stats['count'] > 0:
                print(f"  {metric}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}, 样本数={stats['count']}")
            else:
                print(f"  {metric}: 无数据")

# 3. 显示五折统计汇总（修改后，使用五折平均和五折标准差）
print("\n==== 五折统计汇总 ====")
for metric in metrics_to_analyze:
    cross_stats = calculate_cross_fold_stats(metrics, metric)
    if cross_stats['valid_folds'] > 0:
        print(f"{metric}:")
        print(f"  五折平均均值: {cross_stats['cross_fold_mean']:.4f}")
        print(f"  五折均值标准差: {cross_stats['cross_fold_std']:.4f}")
        print(f"  有效折叠数: {cross_stats['valid_folds']}/5")
    else:
        print(f"{metric}: 无有效数据")