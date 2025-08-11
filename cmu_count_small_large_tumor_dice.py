import pandas as pd
import json
import numpy as np
from collections import defaultdict, OrderedDict
import os

# ======================
# 1. 读取CSV并构建标签映射
# ======================
csv_path = "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_raw/nnUNet_raw_data_base/nnUNet_raw_data/Task133_CMUexternalValDCET2Reg/tumor_size_analysis_with_folds.csv"
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
# 2. 定义函数：处理单个折的JSON数据并计算该折均值
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

    # 计算该折每个类别和指标的均值
    fold_means = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for center, categories in center_metrics.items():
        for category, metrics in categories.items():
            for metric_name, values in metrics.items():
                if values:
                    fold_means[center][category][metric_name] = np.mean(values)
                else:
                    fold_means[center][category][metric_name] = 0

    return fold_means, center_file_counts


# ======================
# 3. 处理所有折的结果并计算折间均值的标准差
# ======================
fold_paths = [
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task133_CMUexternalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task133_CMUexternalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task133_CMUexternalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task133_CMUexternalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
    "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task133_CMUexternalValDCET2Reg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
]

# 收集所有折的均值数据
all_fold_means = []
all_fold_counts = []

for fold_path in fold_paths:
    if not os.path.exists(fold_path):
        print(f"警告: 文件不存在 - {fold_path}")
        continue

    fold_means, fold_counts = process_fold(fold_path, label_map)
    all_fold_means.append(fold_means)
    all_fold_counts.append(fold_counts)

# ======================
# 4. 计算折间统计值（均值的均值和标准差）
# ======================
# 初始化最终统计结果
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# 获取所有可能的中心和类别
all_centers = set()
all_categories = set()

for fold_means in all_fold_means:
    all_centers.update(fold_means.keys())
    for center in fold_means:
        all_categories.update(fold_means[center].keys())

# 计算每个中心和类别的折间统计量
for center in all_centers:
    for category in all_categories:
        # 收集所有折的指标均值
        metric_values = defaultdict(list)
        file_counts = []

        for fold_idx, fold_means in enumerate(all_fold_means):
            if center in fold_means and category in fold_means[center]:
                # 收集该折的指标均值
                for metric in ["Dice", "Jaccard", "HD95"]:
                    metric_values[metric].append(fold_means[center][category][metric])

                # 收集该折的文件数量
                if center in all_fold_counts[fold_idx] and category in all_fold_counts[fold_idx][center]:
                    file_counts.append(all_fold_counts[fold_idx][center][category])

        # 计算折间统计量
        if metric_values and any(metric_values.values()):
            category_stats = {
                "文件数量": file_counts,
                "指标统计": {}
            }

            for metric, values in metric_values.items():
                if values:
                    # 计算均值的均值
                    mean_of_means = np.mean(values)
                    # 计算均值的标准差
                    std_of_means = np.std(values, ddof=1) if len(values) > 1 else 0

                    # 仅对Dice和Jaccard乘以100
                    if metric in ["Dice", "Jaccard"]:
                        mean_of_means *= 100
                        std_of_means *= 100

                    category_stats["指标统计"][metric] = {
                        "均值": mean_of_means,
                        "标准差": std_of_means,
                        "样本数": len(values)  # 折数
                    }

            stats[center][category] = category_stats


# ======================
# 5. 输出结果（紧凑美观格式，保留两位小数，调整类别顺序）
# ======================
def format_metric(value, metric_name):
    """根据指标类型格式化数值：Dice/Jaccard保留两位小数，HD95保留两位小数（但不乘以100）"""
    return f"{value:.2f}"  # 统一保留两位小数，但数值处理已在前面完成


# 构建紧凑美观的输出
compact_output = []

for center, center_data in stats.items():
    center_section = f"中心: {center}"
    compact_output.append(center_section)

    # 按指定顺序处理类别：先T≤5cm，再T＞5cm，最后其他类别
    ordered_categories = []

    # 先添加T≤5cm（如果存在）
    if "T≤5cm" in center_data:
        ordered_categories.append("T≤5cm")

    # 再添加T＞5cm（如果存在）
    if "T＞5cm" in center_data:
        ordered_categories.append("T＞5cm")

    # 添加其他类别（按原始顺序）
    for category in center_data:
        if category not in ordered_categories:
            ordered_categories.append(category)

    # 按排序后的类别顺序生成输出
    for category in ordered_categories:
        category_data = center_data[category]

        # 计算文件数量的平均值并取整
        category_line = f"  类别: {category} (总文件数: {category_data['文件数量']})"
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
