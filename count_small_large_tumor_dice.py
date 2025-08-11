import pandas as pd
import json
import numpy as np
from collections import defaultdict
import os

# ======================
# 1. 读取CSV并构建标签映射
# ======================
csv_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data_base/nnUNet_raw_data/Task500_DCESegment/tumor_size_analysis.csv"
csv_data = pd.read_csv(csv_path)
csv_data["prefix"] = csv_data["filename"].str.split(".nii.gz").str[0]

# 构建前缀 → (category, center) 的映射
label_map = dict(zip(
    csv_data["prefix"],
    zip(csv_data["category"], csv_data["center"])  # (category, center)
))


# 定义新的分类映射函数
def remap_category(original_category):
    if original_category == "T≤2cm" or original_category == "2＜T≤5cm":
        return "T≤5cm"
    elif original_category == "T＞5cm":
        return "T＞5cm"
    return original_category  # 保持其他类别不变


# ======================
# 2. 定义函数：处理单个折的JSON数据
# ======================
def process_fold(json_path, label_map):
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # 初始化存储结构
    center_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    center_file_counts = defaultdict(lambda: defaultdict(int))

    for sample in json_data["results"]["all"]:
        test_filename = sample["test"].split("/")[-1].split(".nii.gz")[0]
        category_center = label_map.get(test_filename)
        if category_center is None:
            continue

        category, center = category_center
        # 应用新的分类映射
        new_category = remap_category(category)
        center_file_counts[center][new_category] += 1

        tumor_metrics = sample.get("1", {})
        target_metrics = {
            "Dice": tumor_metrics.get("Dice"),
            "Jaccard": tumor_metrics.get("Jaccard"),
            "HD95": tumor_metrics.get("Hausdorff Distance 95")
        }

        for metric_name, value in target_metrics.items():
            if value is not None and isinstance(value, (int, float)):
                center_metrics[center][new_category][metric_name].append(value)

    return center_metrics, center_file_counts


# ======================
# 3. 处理所有折的结果并计算折间统计量
# ======================
fold_paths = [
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task131_DCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
]

# 初始化存储所有折的统计数据
all_fold_stats = []

# 处理每一折并保存统计数据
for fold_path in fold_paths:
    fold_metrics, fold_counts = process_fold(fold_path, label_map)

    # 计算当前折的统计量
    fold_stats = {}
    for center, categories in fold_metrics.items():
        center_stats = {}
        for category, metrics in categories.items():
            category_stats = {
                "文件数量": fold_counts[center][category],
                "指标统计": {
                    metric: {
                        "平均值": np.mean(values),
                        "样本数": len(values)
                    } for metric, values in metrics.items()
                }
            }
            center_stats[category] = category_stats
        fold_stats[center] = center_stats

    all_fold_stats.append(fold_stats)

# ======================
# 4. 计算折间统计值（均值的均值和标准差）
# ======================
# 初始化最终统计结果
stats = {}

# 获取所有可能的中心和类别
all_centers = set()
all_categories = set()
for fold_stats in all_fold_stats:
    all_centers.update(fold_stats.keys())
    for center in fold_stats:
        all_categories.update(fold_stats[center].keys())

# 计算每个中心和类别的折间统计量
for center in all_centers:
    center_stats = {}
    for category in all_categories:
        # 检查该中心和类别是否存在于任何一折
        exists = False
        for fold_stats in all_fold_stats:
            if center in fold_stats and category in fold_stats[center]:
                exists = True
                break

        if not exists:
            continue

        # 收集所有折的指标均值
        metric_means = defaultdict(list)
        file_counts = []

        for fold_stats in all_fold_stats:
            if center in fold_stats and category in fold_stats[center]:
                fold_category_stats = fold_stats[center][category]
                file_counts.append(fold_category_stats["文件数量"])

                for metric, values in fold_category_stats["指标统计"].items():
                    metric_means[metric].append(values["平均值"])

        # 计算折间统计量，仅对Dice和Jaccard乘以100
        category_stats = {
            "文件数量": file_counts,
            "指标统计": {}
        }

        for metric, means in metric_means.items():
            if len(means) > 0:
                mean_value = np.mean(means)
                std_value = np.std(means, ddof=1) if len(means) > 1 else 0

                # 仅对Dice和Jaccard乘以100
                if metric in ["Dice", "Jaccard"]:
                    mean_value *= 100
                    std_value *= 100

                category_stats["指标统计"][metric] = {
                    "均值": mean_value,
                    "标准差": std_value,
                }

        center_stats[category] = category_stats

    if center_stats:  # 只添加有数据的中心
        stats[center] = center_stats


# ======================
# 5. 输出结果（紧凑美观格式，保留两位小数）
# ======================
def format_metric(value, metric_name):
    """根据指标类型格式化数值：Dice/Jaccard保留两位小数，HD95保留两位小数（但不乘以100）"""
    return f"{value:.2f}"  # 统一保留两位小数，但数值处理已在前面完成


# 定义固定的类别顺序，T≤5cm在前，T＞5cm在后
CATEGORY_ORDER = ["T≤5cm", "T＞5cm"]

# 构建紧凑美观的输出
compact_output = []

for center, center_data in stats.items():
    center_section = f"中心: {center}"
    compact_output.append(center_section)

    # 先按固定顺序输出已知类别
    for category in CATEGORY_ORDER:
        if category in center_data:
            category_data = center_data[category]
            # 计算文件数量的平均值并取整
            category_line = f"  类别: {category} (文件数: {category_data['文件数量']})"
            compact_output.append(category_line)

            metrics_line = "    "
            for metric, stats in category_data["指标统计"].items():
                mean = format_metric(stats["均值"], metric)
                std = format_metric(stats["标准差"], metric)
                metrics_line += f"{metric}: {mean}±{std}  "

            compact_output.append(metrics_line)
            compact_output.append("")  # 添加空行分隔不同类别

    # 输出不在固定顺序中的其他类别（如果有）
    other_categories = [cat for cat in center_data.keys() if cat not in CATEGORY_ORDER]
    for category in other_categories:
        category_data = center_data[category]
        # 计算文件数量的平均值并取整
        category_line = f"  类别: {category} (文件数: {category_data['文件数量']})"
        compact_output.append(category_line)

        metrics_line = "    "
        for metric, stats in category_data["指标统计"].items():
            mean = format_metric(stats["均值"], metric)
            std = format_metric(stats["标准差"], metric)
            metrics_line += f"{metric}: {mean}±{std}  "

        compact_output.append(metrics_line)
        compact_output.append("")  # 添加空行分隔不同类别

# 打印紧凑美观的输出
for line in compact_output:
    print(line)
