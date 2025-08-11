import os
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def extract_metrics(summary_path: str, yn_folder_key: str, cmu_folder_key: str) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]], List[str], List[str], int, int]:
    """
    提取指标并保留样本ID（用于配对）
    返回：YN指标、CMU指标、YN样本ID列表、CMU样本ID列表、YN有效样本数、CMU有效样本数
    """
    yn_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    cmu_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    yn_sample_ids = []  # YN中心样本唯一ID（用于配对）
    cmu_sample_ids = []  # CMU中心样本唯一ID
    yn_valid_count = 0  # YN有效样本数（至少一个指标非NaN）
    cmu_valid_count = 0  # CMU有效样本数

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败 {summary_path}: {e}")
        return yn_metrics, cmu_metrics, yn_sample_ids, cmu_sample_ids, yn_valid_count, cmu_valid_count

    for sample in data.get('results', {}).get('all', []):
        if not isinstance(sample, dict) or '1' not in sample:
            continue

        metrics = sample['1']
        sample_path = sample.get('reference', '') or sample.get('prediction', '')
        # 提取样本唯一ID（以文件名作为标识，确保同一病例可配对）
        sample_id = os.path.basename(sample_path).split('.')[0]  # 如从"case001.nii.gz"提取"case001"

        # 判断中心
        if yn_folder_key in sample_path:
            center_metrics = yn_metrics
            center_ids = yn_sample_ids
            count_var = 'yn'
        else:
            center_metrics = cmu_metrics
            center_ids = cmu_sample_ids
            count_var = 'cmu'

        # 提取指标
        dsc = metrics.get('Dice', np.nan)
        iou = metrics.get('Jaccard', np.nan)
        sen = metrics.get('Recall', np.nan)
        ppv = metrics.get('Precision', np.nan)
        hd95 = metrics.get('Hausdorff Distance 95', np.nan)

        center_metrics['DSC'].append(dsc)
        center_metrics['IOU'].append(iou)
        center_metrics['SEN'].append(sen)
        center_metrics['PPV'].append(ppv)
        center_metrics['HD95'].append(hd95)
        center_ids.append(sample_id)  # 记录样本ID用于后续配对

        # 统计有效样本（至少一个指标非NaN）
        if not all(np.isnan([dsc, iou, sen, ppv, hd95])):
            if count_var == 'yn':
                yn_valid_count += 1
            else:
                cmu_valid_count += 1

    return yn_metrics, cmu_metrics, yn_sample_ids, cmu_sample_ids, yn_valid_count, cmu_valid_count


def sort_and_deduplicate(metrics: Dict[str, List[float]], ids: List[str]) -> Tuple[Dict[str, List[float]], List[str]]:
    """按样本ID排序并去重（同一ID保留首个出现的结果）"""
    # 组合ID和指标，按ID去重（保留第一个）
    id_metrics_map = {}
    for sid, dsc, iou, sen, ppv, hd95 in zip(ids, metrics['DSC'], metrics['IOU'], metrics['SEN'], metrics['PPV'],
                                             metrics['HD95']):
        if sid not in id_metrics_map:
            id_metrics_map[sid] = (dsc, iou, sen, ppv, hd95)

    # 按ID排序
    sorted_sids = sorted(id_metrics_map.keys())
    sorted_metrics = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for sid in sorted_sids:
        dsc, iou, sen, ppv, hd95 = id_metrics_map[sid]
        sorted_metrics['DSC'].append(dsc)
        sorted_metrics['IOU'].append(iou)
        sorted_metrics['SEN'].append(sen)
        sorted_metrics['PPV'].append(ppv)
        sorted_metrics['HD95'].append(hd95)
    return sorted_metrics, sorted_sids


def collect_all_folds_metrics(paths: List[str], yn_key: str, cmu_key: str) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]], List[str], List[str], int, int]:
    """
    合并所有折的指标，按样本ID去重+排序（确保配对）
    返回：YN合并指标、CMU合并指标、YN样本ID、CMU样本ID、YN总样本数、CMU总样本数
    """
    # 临时存储各折的指标和ID
    yn_all_metrics = []
    yn_all_ids = []
    cmu_all_metrics = []
    cmu_all_ids = []
    total_yn_count = 0
    total_cmu_count = 0

    for path in paths:
        if not os.path.exists(path):
            print(f"警告：路径不存在 {path}")
            continue
        # 提取当前折的指标、ID和样本数
        yn_metrics, cmu_metrics, yn_ids, cmu_ids, yn_count, cmu_count = extract_metrics(path, yn_key, cmu_key)
        yn_all_metrics.append(yn_metrics)
        yn_all_ids.extend(yn_ids)
        cmu_all_metrics.append(cmu_metrics)
        cmu_all_ids.extend(cmu_ids)
        total_yn_count += yn_count
        total_cmu_count += cmu_count

    # 合并YN指标（按ID去重+排序）
    yn_combined = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for metric in yn_combined:
        for fold_metrics in yn_all_metrics:
            yn_combined[metric].extend(fold_metrics[metric])
    yn_sorted, yn_sorted_ids = sort_and_deduplicate(yn_combined, yn_all_ids)

    # 合并CMU指标（按ID去重+排序）
    cmu_combined = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for metric in cmu_combined:
        for fold_metrics in cmu_all_metrics:
            cmu_combined[metric].extend(fold_metrics[metric])
    cmu_sorted, cmu_sorted_ids = sort_and_deduplicate(cmu_combined, cmu_all_ids)

    return yn_sorted, cmu_sorted, yn_sorted_ids, cmu_sorted_ids, total_yn_count, total_cmu_count


def calculate_pvalue(my_metrics: Dict[str, List[float]], my_ids: List[str],
                     other_metrics: Dict[str, List[float]], other_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """
    使用Wilcoxon符号秩检验（配对检验）计算p值
    仅对两个模型共有的样本ID（配对样本）进行检验
    """
    results = {}
    # 找到共同的样本ID（配对基础）
    common_ids = list(set(my_ids) & set(other_ids))
    if not common_ids:
        print("警告：无共同样本ID，无法进行配对检验")
        for metric in ['DSC', 'IOU', 'SEN', 'PPV', 'HD95']:
            results[metric] = {'p_value': np.nan, 'significant': False}
        return results

    # 构建ID到指标的映射（用于快速提取配对数据）
    my_id_map = dict(zip(my_ids, zip(my_metrics['DSC'], my_metrics['IOU'], my_metrics['SEN'], my_metrics['PPV'],
                                     my_metrics['HD95'])))
    other_id_map = dict(zip(other_ids,
                            zip(other_metrics['DSC'], other_metrics['IOU'], other_metrics['SEN'], other_metrics['PPV'],
                                other_metrics['HD95'])))

    # 提取配对样本的指标
    my_paired = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    other_paired = {'DSC': [], 'IOU': [], 'SEN': [], 'PPV': [], 'HD95': []}
    for sid in common_ids:
        my_vals = my_id_map[sid]
        other_vals = other_id_map[sid]
        for i, metric in enumerate(['DSC', 'IOU', 'SEN', 'PPV', 'HD95']):
            my_paired[metric].append(my_vals[i])
            other_paired[metric].append(other_vals[i])

    # 对每个指标进行符号秩检验
    for metric in ['DSC', 'IOU', 'SEN', 'PPV', 'HD95']:
        my_vals = np.array(my_paired[metric])
        other_vals = np.array(other_paired[metric])

        # 过滤掉任一模型为NaN的配对样本
        valid_mask = ~np.isnan(my_vals) & ~np.isnan(other_vals)
        my_valid = my_vals[valid_mask]
        other_valid = other_vals[valid_mask]

        # 确保至少有2对有效样本
        if len(my_valid) < 2:
            p_value = np.nan
            significant = False
        else:
            # Wilcoxon符号秩检验（配对样本检验）
            _, p_value = stats.wilcoxon(my_valid, other_valid)
            significant = p_value < 0.05

        results[metric] = {
            'p_value': round(p_value, 4),
            'significant': significant
        }
    return results


def main():
    # 配置参数
    YN_FOLDER_KEY = "YN"
    CMU_FOLDER_KEY = "CMU"

    # 自己模型路径
    MY_MODEL_PATHS = [
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
        "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
    ]

    # 提取自己模型的合并指标和样本ID（用于配对）
    my_yn, my_cmu, my_yn_ids, my_cmu_ids, my_yn_total, my_cmu_total = collect_all_folds_metrics(MY_MODEL_PATHS,
                                                                                                YN_FOLDER_KEY,
                                                                                                CMU_FOLDER_KEY)
    print(f"自己的模型 - YN中心有效样本数：{len(my_yn_ids)}（总样本数{my_yn_total}）；CMU中心有效样本数：{len(my_cmu_ids)}（总样本数{my_cmu_total}）")

    # 对比模型
    COMPARE_MODELS = {
        "nnunet": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "nnformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_4/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "MAML": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "A2FSeg": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "PA-Net": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_0/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_1/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_2/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_3/validation_raw/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1/fold_4/validation_raw/summary.json",
        ],
        "mmformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task263_externalValT2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task263_externalValT2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task263_externalValT2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task263_externalValT2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task263_externalValT2DCEReg/fold_4/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg/fold_4/summary.json",
        ],
        "Nestedformer": [
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task263_externalValT2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task263_externalValT2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task263_externalValT2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task263_externalValT2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task263_externalValT2DCEReg/fold_4/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg/fold_0/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg/fold_1/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg/fold_2/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg/fold_3/summary.json",
            "/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg/fold_4/summary.json",
        ],
    }

    # 对比每个模型（使用配对检验）
    for model_name, model_paths in COMPARE_MODELS.items():
        print(f"\n\n===== 与模型 {model_name} 的对比 =====")
        # 提取对比模型的合并指标、ID和样本数
        other_yn, other_cmu, other_yn_ids, other_cmu_ids, other_yn_total, other_cmu_total = collect_all_folds_metrics(
            model_paths, YN_FOLDER_KEY, CMU_FOLDER_KEY)
        print(
            f"{model_name} - YN中心有效样本数：{len(other_yn_ids)}（总样本数{other_yn_total}）；CMU中心有效样本数：{len(other_cmu_ids)}（总样本数{other_cmu_total}）")

        # YN中心配对检验
        print("\n--- YN中心（配对检验） ---")
        yn_common = len(set(my_yn_ids) & set(other_yn_ids))
        print(f"YN中心共同样本数（用于配对）：{yn_common}")
        yn_results = calculate_pvalue(my_yn, my_yn_ids, other_yn, other_yn_ids)
        for metric, res in yn_results.items():
            sig = "显著" if res['significant'] else "不显著"
            print(f"{metric}：p值={res['p_value']:.4f}，{sig}")

        # CMU中心配对检验
        print("\n--- CMU中心（配对检验） ---")
        cmu_common = len(set(my_cmu_ids) & set(other_cmu_ids))
        print(f"CMU中心共同样本数（用于配对）：{cmu_common}")
        cmu_results = calculate_pvalue(my_cmu, my_cmu_ids, other_cmu, other_cmu_ids)
        for metric, res in cmu_results.items():
            sig = "显著" if res['significant'] else "不显著"
            print(f"{metric}：p值={res['p_value']:.4f}，{sig}")


if __name__ == "__main__":
    main()
