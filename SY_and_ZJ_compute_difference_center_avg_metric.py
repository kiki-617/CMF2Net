import json
import numpy as np
from pathlib import Path

# 基础路径（根据实际情况调整）
base_path = Path(
    # '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task500_DCESegment/nnUNetTrainerV2__nnUNetPlansv2.1'
    # '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1'
    # '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1'
    '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_600/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1'
)

# 初始化指标存储：每个fold的结果
metrics = {}

# 为每个fold创建单独的指标存储
for fold in range(5):
    metrics[f'Fold_{fold}'] = {
        'Dice': [],
        'Jaccard': [],
        'Precision': [],
        'Recall': [],
        'Hausdorff Distance 95': []
    }

# 遍历所有折叠（0-4）
for fold in range(5):
    # 构建当前折叠的JSON文件路径
    json_path = base_path / f'fold_{fold}/validation_raw/summary.json'

    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理当前折叠的数据
        print(f"正在处理 fold_{fold}...")
        valid_samples = 0

        for result in data['results']['all']:
            category_1 = result.get('1', {})  # 假设类别1为目标类别
            if not category_1:
                continue  # 跳过无类别1的样本

            reference = result.get('reference', '')
            if not reference:
                continue  # 跳过无参考路径的样本

            filename = reference.split('/')[-1]  # 提取文件名

            # 判断文件名是否以ZJ或SY开头
            if filename.startswith(('ZJ', 'SY')):
                # 提取指标并过滤无效值
                dice = category_1.get('Dice', 0.0)
                if not (0.0 <= dice <= 1.0):
                    continue  # 过滤无效Dice值

                # 存储到当前fold结果
                metrics[f'Fold_{fold}']['Dice'].append(dice)
                metrics[f'Fold_{fold}']['Jaccard'].append(category_1['Jaccard'])
                metrics[f'Fold_{fold}']['Precision'].append(category_1['Precision'])
                metrics[f'Fold_{fold}']['Recall'].append(category_1['Recall'])
                metrics[f'Fold_{fold}']['Hausdorff Distance 95'].append(category_1['Hausdorff Distance 95'])

                valid_samples += 1

        print(f"fold_{fold} 处理完成，找到 {valid_samples} 个有效样本")

    except FileNotFoundError:
        print(f"警告: fold_{fold} 的JSON文件未找到，跳过此折叠")
        continue
    except Exception as e:
        print(f"错误: 处理 fold_{fold} 时发生异常: {e}")
        continue


# 计算统计指标的函数
def calculate_stats(metrics_list):
    if not metrics_list:
        return {
            'mean': 0.0,
            'std': 0.0,
            'count': 0
        }
    arr = np.array(metrics_list)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),  # 样本标准差（无偏估计）
        'count': len(arr)
    }


# 计算各折的统计数据
fold_statistics = {}
for fold in range(5):
    fold_name = f'Fold_{fold}'
    if fold_name not in metrics or not any(metrics[fold_name].values()):
        print(f"\n{fold_name} 无有效数据")
        continue

    fold_statistics[fold_name] = {
        metric: calculate_stats(values)
        for metric, values in metrics[fold_name].items()
    }

# 计算五折均值的标准差
metrics_list = ['Dice', 'Jaccard', 'Precision', 'Recall', 'Hausdorff Distance 95']
cross_fold_stats = {}

for metric in metrics_list:
    # 收集各折的均值
    fold_means = [
        fold_statistics[f'Fold_{fold}'][metric]['mean']
        for fold in range(5)
        if f'Fold_{fold}' in fold_statistics
    ]

    # 计算这些均值的统计数据
    cross_fold_stats[metric] = calculate_stats(fold_means)

# 输出结果
print("\n==== 各折叠单独统计结果 ====")
for fold in range(5):
    fold_name = f'Fold_{fold}'
    if fold_name not in fold_statistics:
        print(f"\n{fold_name} 无有效数据")
        continue

    fold_stats = fold_statistics[fold_name]
    count = fold_stats['Dice']['count']

    print(f"\n{fold_name}（{count}个样本）统计结果：")
    for metric, values in fold_stats.items():
        print(f"{metric} 均值: {values['mean']:.4f}, 标准差: {values['std']:.4f}")

# 输出五折交叉验证的统计结果
print("\n==== 五折交叉验证统计结果 ====")
print("\n各折均值的汇总统计：")
for metric, values in cross_fold_stats.items():
    print(f"{metric} 均值的均值: {values['mean']:.4f}, 均值的标准差: {values['std']:.4f}")
