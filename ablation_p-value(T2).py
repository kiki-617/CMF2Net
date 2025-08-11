import os
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def extract_metrics_from_summary(summary_path: str) -> Dict[str, List[float]]:
    """从summary.json提取所有样本的指标（保留样本顺序，确保配对）"""
    all_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}

    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return all_metrics

    # 遍历样本时严格保留原始顺序（确保配对）
    for sample in data.get('results', {}).get('all', []):
        if not isinstance(sample, dict) or '1' not in sample:
            continue
        metrics = sample['1']
        all_metrics['DSC'].append(metrics.get('Dice', np.nan))
        all_metrics['IOU'].append(metrics.get('Jaccard', np.nan))
        all_metrics['SEN'].append(metrics.get('Recall', np.nan))
        all_metrics['PPV'].append(metrics.get('Precision', np.nan))
        all_metrics['HD95'].append(metrics.get('Hausdorff Distance 95', np.nan))

    return all_metrics


def calculate_significance(my_model_metrics: Dict[str, List[float]],
                           other_model_metrics: Dict[str, List[float]],
                           model_name: str) -> Dict[str, Dict[str, float]]:
    """
    计算配对样本的Wilcoxon符号秩检验
    前提：两组样本是一一配对的（同一批样本的两种模型结果）
    """
    results = {}
    for metric in ['DSC', 'IOU', 'SEN', 'PPV', 'HD95']:
        my_values = np.array(my_model_metrics[metric])
        other_values = np.array(other_model_metrics[metric])

        # 1. 过滤NaN值（同时删除两组中任一为NaN的样本，保证配对关系）
        mask = ~np.isnan(my_values) & ~np.isnan(other_values)
        my_clean = my_values[mask]
        other_clean = other_values[mask]

        # 2. 检查配对样本数量是否满足要求（符号秩检验至少需要1个非零差异）
        if len(my_clean) < 1:
            p_value = np.nan
            significant = False
            print(f"警告：{metric} 有效配对样本数为0，无法进行检验")
        else:
            # 3. 计算差异（用于符号秩检验）
            differences = my_clean - other_clean
            # 排除差异为0的样本（符号秩检验忽略无差异的样本）
            non_zero_mask = differences != 0
            if np.sum(non_zero_mask) < 1:
                p_value = 1.0  # 所有差异为0，无显著性
                significant = False
            else:
                # 4. 执行Wilcoxon符号秩检验（返回Z值和p值）
                _, p_value = stats.wilcoxon(differences[non_zero_mask])
                significant = p_value < 0.05

        results[metric] = {
            'p_value': p_value,
            'significant': significant,
            'valid_paired_samples': len(my_clean)  # 有效配对样本数
        }

    return results

def main():
    my_model_paths = [
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_400/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_400/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_400/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_400/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/cat_DS_400/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
    ]


    # 对比模型路径（保持不变）
    other_models = {
        "Single-modality(T2)": [
            '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json',
            '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json',
            '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json',
            '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json',
            '/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task501_T2Segment/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json',
        ],

        "Multi-modality+CFSG": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "Multi-modality+CMGA": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_500/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "Multi-modality+CFSG+CMGA": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "Multi-modality+CFSG+TDS": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMFF_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "Multi-modality+CMGA+TDS": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/CMGA_DS_1000/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        # 自己模型的五折路径（保持不变）
        "Ours" : [
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json"
        ],
    }

    # 1. 提取你的模型所有折的指标（保留样本顺序，确保配对）
    my_all_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for path in my_model_paths:
        fold_metrics = extract_metrics_from_summary(path)
        for metric in my_all_metrics:
            my_all_metrics[metric].extend(fold_metrics[metric])
    print(f"自己的模型：合并后总样本数 {len(my_all_metrics['DSC'])} 个")

    # 2. 与每个对比模型进行符号秩检验
    for model_name, model_paths in other_models.items():
        print(f"\n\n===== 与模型 {model_name} 的对比 =====")

        # 提取对比模型所有折的指标（保留样本顺序）
        other_all_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
        for path in model_paths:
            fold_metrics = extract_metrics_from_summary(path)
            for metric in other_all_metrics:
                other_all_metrics[metric].extend(fold_metrics[metric])
        print(f"{model_name}：合并后总样本数 {len(other_all_metrics['DSC'])} 个")

        # 检查样本数量是否匹配（配对样本必须数量相同）
        if len(my_all_metrics['DSC']) != len(other_all_metrics['DSC']):
            print(f"警告：样本数量不匹配（你的模型：{len(my_all_metrics['DSC'])}，对比模型：{len(other_all_metrics['DSC'])}），可能影响检验有效性")

        # 计算符号秩检验结果
        overall_results = calculate_significance(my_all_metrics, other_all_metrics, model_name)
        print("\n--- 符号秩检验结果 ---")
        for metric, result in overall_results.items():
            significant_text = "显著" if result['significant'] else "不显著"
            print(
                f"{metric}: 有效配对样本数={result['valid_paired_samples']}, p值={result['p_value']:.4f}, 显著性: {significant_text}")


if __name__ == "__main__":
    main()
