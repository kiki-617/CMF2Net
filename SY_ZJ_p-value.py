import os
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def extract_metrics_from_summary(summary_path: str) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]]]:
    """
    提取指标并保留样本ID（用于配对）
    返回：SY中心指标、ZJ中心指标、样本ID字典（{中心: [样本ID列表]}）
    """
    # 初始化指标和样本ID存储
    sy_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    zj_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    sample_ids = {'SY': [], 'ZJ': []}  # 存储每个中心的样本ID（用于后续排序配对）

    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return sy_metrics, zj_metrics, sample_ids

    for sample in data.get('results', {}).get('all', []):
        if not isinstance(sample, dict) or '1' not in sample:
            continue

        metrics = sample['1']
        reference_path = sample.get('reference', '')
        # 提取样本唯一ID（以参考文件名为ID，确保唯一性）
        sample_id = os.path.basename(reference_path).split('.')[0]  # 如从"SY_case001.nii.gz"提取"SY_case001"

        # 判断中心并存储指标和ID
        if 'SY' in os.path.basename(reference_path):
            center_metrics = sy_metrics
            center_id_list = sample_ids['SY']
        elif 'ZJ' in os.path.basename(reference_path):
            center_metrics = zj_metrics
            center_id_list = sample_ids['ZJ']
        else:
            continue  # 未知中心跳过

        # 存储指标和样本ID
        center_metrics['DSC'].append(metrics.get('Dice', np.nan))
        center_metrics['IOU'].append(metrics.get('Jaccard', np.nan))
        center_metrics['SEN'].append(metrics.get('Recall', np.nan))
        center_metrics['PPV'].append(metrics.get('Precision', np.nan))
        center_metrics['HD95'].append(metrics.get('Hausdorff Distance 95', np.nan))
        center_id_list.append(sample_id)  # 记录当前样本ID

    return sy_metrics, zj_metrics, sample_ids


def sort_metrics_by_id(metrics: Dict[str, List[float]], ids: List[str]) -> Tuple[Dict[str, List[float]], List[str]]:
    """按样本ID排序指标，保证同一中心内样本顺序固定（用于配对）"""
    # 组合ID和指标并按ID排序
    sorted_pairs = sorted(zip(ids, metrics['DSC'], metrics['IOU'], metrics['SEN'], metrics['PPV'], metrics['HD95']),
                          key=lambda x: x[0])  # 按样本ID升序排序
    # 拆分排序后的结果
    sorted_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    sorted_ids = []
    for pair in sorted_pairs:
        sid, dsc, iou, sen, ppv, hd95 = pair
        sorted_ids.append(sid)
        sorted_metrics['DSC'].append(dsc)
        sorted_metrics['IOU'].append(iou)
        sorted_metrics['SEN'].append(sen)
        sorted_metrics['PPV'].append(ppv)
        sorted_metrics['HD95'].append(hd95)
    return sorted_metrics, sorted_ids


def merge_folds_with_id(fold_metrics_list: List[Dict[str, List[float]]], fold_ids_list: List[List[str]]) -> Tuple[
    Dict[str, List[float]], List[str]]:
    """合并多折的指标，并按样本ID去重+排序（确保每个样本只保留一个结果）"""
    # 临时存储所有折的ID和指标（键：样本ID，值：指标列表）
    id_to_metrics = {}
    for fold_metrics, fold_ids in zip(fold_metrics_list, fold_ids_list):
        for sid, dsc, iou, sen, ppv, hd95 in zip(fold_ids,
                                                 fold_metrics['DSC'],
                                                 fold_metrics['IOU'],
                                                 fold_metrics['SEN'],
                                                 fold_metrics['PPV'],
                                                 fold_metrics['HD95']):
            if sid not in id_to_metrics:  # 去重：同一ID只保留首次出现的结果（或可改为取均值）
                id_to_metrics[sid] = (dsc, iou, sen, ppv, hd95)

    # 按ID排序并转换为列表
    sorted_sids = sorted(id_to_metrics.keys())
    merged_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for sid in sorted_sids:
        dsc, iou, sen, ppv, hd95 = id_to_metrics[sid]
        merged_metrics['DSC'].append(dsc)
        merged_metrics['IOU'].append(iou)
        merged_metrics['SEN'].append(sen)
        merged_metrics['PPV'].append(ppv)
        merged_metrics['HD95'].append(hd95)
    return merged_metrics, sorted_sids


def calculate_significance(my_model_metrics: Dict[str, List[float]],
                           my_model_ids: List[str],
                           other_model_metrics: Dict[str, List[float]],
                           other_model_ids: List[str],
                           center_name: str,
                           model_name: str) -> Dict[str, Dict[str, float]]:
    """
    用Wilcoxon符号秩检验（配对检验）计算显著性
    要求：my_model_ids和other_model_ids有共同的样本ID（配对基础）
    """
    results = {}
    # 1. 找到共同的样本ID（仅保留两个模型都有的样本）
    common_ids = list(set(my_model_ids) & set(other_model_ids))
    if not common_ids:
        print(f"警告：{center_name}中心无共同样本，无法进行配对检验")
        for metric in ['DSC', 'IOU', 'SEN', 'PPV', 'HD95']:
            results[metric] = {'p_value': np.nan, 'significant': False}
        return results

    # 2. 为共同样本提取两个模型的指标（按共同ID排序）
    my_dict = dict(zip(my_model_ids, zip(my_model_metrics['DSC'], my_model_metrics['IOU'], my_model_metrics['SEN'],
                                         my_model_metrics['PPV'], my_model_metrics['HD95'])))
    other_dict = dict(zip(other_model_ids,
                          zip(other_model_metrics['DSC'], other_model_metrics['IOU'], other_model_metrics['SEN'],
                              other_model_metrics['PPV'], other_model_metrics['HD95'])))

    my_paired = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    other_paired = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for sid in common_ids:
        my_vals = my_dict[sid]
        other_vals = other_dict[sid]
        for i, metric in enumerate(['DSC', 'IOU', 'SEN', 'PPV', 'HD95']):
            my_paired[metric].append(my_vals[i])
            other_paired[metric].append(other_vals[i])

    # 3. 对每个指标进行符号秩检验
    for metric in ['DSC', 'IOU', 'SEN', 'PPV', 'HD95']:
        my_values = np.array(my_paired[metric])
        other_values = np.array(other_paired[metric])

        # 过滤掉任一模型为NaN的样本（仅保留均为有效值的配对）
        valid_mask = ~np.isnan(my_values) & ~np.isnan(other_values)
        my_valid = my_values[valid_mask]
        other_valid = other_values[valid_mask]

        # 确保有足够的配对样本（至少2对）
        if len(my_valid) < 2:
            p_value = np.nan
            significant = False
        else:
            # Wilcoxon符号秩检验（配对检验）
            _, p_value = stats.wilcoxon(my_valid, other_valid)
            significant = p_value < 0.05

        results[metric] = {
            'p_value': p_value,
            'significant': significant
        }

    return results


def main():
    # 1. 提取我的模型的指标和样本ID（按中心+折数）
    my_sy_fold_metrics = []  # 存储SY中心各折的指标
    my_sy_fold_ids = []  # 存储SY中心各折的样本ID
    my_zj_fold_metrics = []  # 存储ZJ中心各折的指标
    my_zj_fold_ids = []  # 存储ZJ中心各折的样本ID

    my_model_paths = [
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json"
    ]

    for path in my_model_paths:
        sy_metrics, zj_metrics, sample_ids = extract_metrics_from_summary(path)
        my_sy_fold_metrics.append(sy_metrics)
        my_sy_fold_ids.append(sample_ids['SY'])
        my_zj_fold_metrics.append(zj_metrics)
        my_zj_fold_ids.append(sample_ids['ZJ'])

    # 2. 合并我的模型的指标（去重+按ID排序）
    my_sy_merged, my_sy_ids = merge_folds_with_id(my_sy_fold_metrics, my_sy_fold_ids)
    my_zj_merged, my_zj_ids = merge_folds_with_id(my_zj_fold_metrics, my_zj_fold_ids)
    print(f"我的模型 - SY中心有效样本数：{len(my_sy_ids)}；ZJ中心有效样本数：{len(my_zj_ids)}")

    # 3. 处理对比模型
    other_models = {
        "nnunet": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json"
        ],
        "nnformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "MAML": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "A2FSeg": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "PA-Net": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "mmformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task262_T2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task262_T2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task262_T2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task262_T2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task262_T2DCEReg/fold_4/summary.json",
        ],
        "Nestedformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task262_T2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task262_T2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task262_T2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task262_T2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task262_T2DCEReg/fold_4/summary.json",
        ],
    }

    # 4. 逐个对比模型计算显著性（配对检验）
    for model_name, model_paths in other_models.items():
        print(f"\n\n===== 与模型 {model_name} 的对比 =====")

        # 提取对比模型的指标和ID
        other_sy_fold_metrics = []
        other_sy_fold_ids = []
        other_zj_fold_metrics = []
        other_zj_fold_ids = []
        for path in model_paths:
            sy_metrics, zj_metrics, sample_ids = extract_metrics_from_summary(path)
            other_sy_fold_metrics.append(sy_metrics)
            other_sy_fold_ids.append(sample_ids['SY'])
            other_zj_fold_metrics.append(zj_metrics)
            other_zj_fold_ids.append(sample_ids['ZJ'])

        # 合并对比模型的指标（去重+按ID排序）
        other_sy_merged, other_sy_ids = merge_folds_with_id(other_sy_fold_metrics, other_sy_fold_ids)
        other_zj_merged, other_zj_ids = merge_folds_with_id(other_zj_fold_metrics, other_zj_fold_ids)
        print(f"{model_name} - SY中心有效样本数：{len(other_sy_ids)}；ZJ中心有效样本数：{len(other_zj_ids)}")

        # 5. SY中心配对检验
        print("\n--- SY中心（配对检验） ---")
        sy_results = calculate_significance(my_sy_merged, my_sy_ids, other_sy_merged, other_sy_ids, "SY", model_name)
        for metric, result in sy_results.items():
            significant_text = "显著" if result['significant'] else "不显著"
            print(f"{metric}: p值={result['p_value']:.4f}, 显著性: {significant_text}")

        # 6. ZJ中心配对检验
        print("\n--- ZJ中心（配对检验） ---")
        zj_results = calculate_significance(my_zj_merged, my_zj_ids, other_zj_merged, other_zj_ids, "ZJ", model_name)
        for metric, result in zj_results.items():
            significant_text = "显著" if result['significant'] else "不显著"
            print(f"{metric}: p值={result['p_value']:.4f}, 显著性: {significant_text}")


if __name__ == "__main__":
    main()
