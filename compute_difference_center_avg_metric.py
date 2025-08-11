import json
import numpy as np
from pathlib import Path

# 基础路径（根据实际情况调整）
base_path = Path(
    '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1')

# 初始化指标存储：每个fold中ZJ和SY的独立数据
metrics = {
    f'Fold_{fold}': {
        'ZJ': {
            'Dice': [],
            'Jaccard': [],
            'Precision': [],
            'Recall': [],
            'Hausdorff Distance 95': []
        },
        'SY': {
            'Dice': [],
            'Jaccard': [],
            'Precision': [],
            'Recall': [],
            'Hausdorff Distance 95': []
        }
    } for fold in range(5)
}

# 遍历所有折叠（0-4）
for fold in range(5):
    json_path = base_path / f'fold_{fold}/validation_raw/summary.json'

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"正在处理 fold_{fold}...")
        valid_samples_zj = 0
        valid_samples_sy = 0

        for result in data['results']['all']:
            category_1 = result.get('1', {})
            if not category_1:
                continue

            reference = result.get('reference', '')
            if not reference:
                continue

            filename = reference.split('/')[-1]
            center = None

            # 判断属于ZJ还是SY
            if filename.startswith('ZJ'):
                center = 'ZJ'
            elif filename.startswith('SY'):
                center = 'SY'

            if not center:
                continue  # 跳过非目标中心的样本

            # 提取指标并过滤无效值
            dice = category_1.get('Dice', 0.0)
            if not (0.0 <= dice <= 1.0):
                continue

            # 存储到当前fold的对应中心
            metrics[f'Fold_{fold}'][center]['Dice'].append(dice)
            metrics[f'Fold_{fold}'][center]['Jaccard'].append(category_1['Jaccard'])
            metrics[f'Fold_{fold}'][center]['Precision'].append(category_1['Precision'])
            metrics[f'Fold_{fold}'][center]['Recall'].append(category_1['Recall'])
            metrics[f'Fold_{fold}'][center]['Hausdorff Distance 95'].append(category_1['Hausdorff Distance 95'])

            # 统计样本数
            if center == 'ZJ':
                valid_samples_zj += 1
            else:
                valid_samples_sy += 1

        print(f"fold_{fold} 处理完成，ZJ样本: {valid_samples_zj}, SY样本: {valid_samples_sy}")

    except FileNotFoundError:
        print(f"警告: fold_{fold} 的JSON文件未找到，跳过此折叠")
        continue
    except Exception as e:
        print(f"错误: 处理 fold_{fold} 时发生异常: {e}")
        continue


# 计算统计指标的函数 - 计算五折之间的均值和标准差
def calculate_fold_stats(metrics_dict, metric_name, center):
    """
    计算所有折叠间特定指标的统计数据

    参数:
    metrics_dict: 包含所有折叠数据的字典
    metric_name: 要计算的指标名称，如'Dice'
    center: 中心名称，如'ZJ'或'SY'

    返回:
    包含均值、标准差和有效折叠数的字典
    """
    fold_values = []
    valid_folds = 0

    for fold in range(5):
        fold_name = f'Fold_{fold}'
        if fold_name in metrics_dict:
            values = metrics_dict[fold_name][center][metric_name]
            if values:  # 只考虑有数据的折叠
                fold_mean = np.mean(values)
                fold_values.append(fold_mean)
                valid_folds += 1

    if not fold_values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'valid_folds': 0
        }

    return {
        'mean': np.mean(fold_values),
        'std': np.std(fold_values, ddof=1),  # 样本标准差
        'valid_folds': valid_folds
    }


# 输出结果：五折之间的统计数据
print("\n==== 五折交叉验证统计结果 ====")
metrics_to_display = ['Dice', 'Jaccard', 'Precision', 'Recall', 'Hausdorff Distance 95']

for center in ['ZJ', 'SY']:
    print(f"\n中心: {center}")
    for metric in metrics_to_display:
        stats = calculate_fold_stats(metrics, metric, center)
        if stats['valid_folds'] > 0:
            print(f"{metric}: 均值 = {stats['mean']:.4f}, 标准差 = {stats['std']:.4f}, 有效折叠数 = {stats['valid_folds']}/5")
        else:
            print(f"{metric}: 没有有效数据")

# 可选：输出每个折叠的详细统计数据
print("\n==== 各折叠详细统计结果 ====")
for fold in range(5):
    fold_name = f'Fold_{fold}'
    if fold_name in metrics:
        print(f"\n{fold_name}:")
        for center in ['ZJ', 'SY']:
            print(f"  中心: {center}")
            for metric in metrics_to_display:
                values = metrics[fold_name][center][metric]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1)
                    count = len(values)
                    print(f"    {metric}: 均值 = {mean_val:.4f}, 标准差 = {std_val:.4f}, 样本数 = {count}")
                else:
                    print(f"    {metric}: 没有数据")