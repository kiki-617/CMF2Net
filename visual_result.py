import os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 在导入pyplot之前设置非交互式后端
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import random
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import re
import argparse

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class SegmentationVisualizer:
    def __init__(self, base_dir, output_dir, num_cases=20):  # 默认为20个病例
        """
        初始化分割可视化工具

        参数:
            base_dir: 数据根目录
            output_dir: 输出图像保存目录
            num_cases: 需要选择的病例数，默认为20
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.num_cases = num_cases  # 改为20
        self.model_dirs = {}  # {模型名: {病例标识: 预测路径}}
        self.gt_dirs = {}  # {模型名: {病例标识: GT路径}}
        self.best_slices = []
        # 标记需要跨fold查找的模型
        self.cross_fold_models = ["nnUNet", "nnFormer"]
        # 标记特殊目录结构的模型（无validation_raw）
        self.special_dir_models = ["mmFormer", "NestedFormer"]
        # 标记没有GT文件夹的模型
        self.no_gt_models = ["mmFormer", "NestedFormer"]

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 定义模型名称和对应的目录路径
        self.model_configs = {
            "YourModel": {
                "internal": os.path.join(base_dir,
                                         "MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "MMFAFM_BCS/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1")
            },
            "nnUNet": {
                "internal": os.path.join(base_dir,
                                         "nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task262_T2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task263_externalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "nnunet/nnUNet/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/nnUNetTrainerV2__nnUNetPlansv2.1")
            },
            "nnFormer": {
                "internal": os.path.join(base_dir,
                                         "nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task262_T2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task263_externalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "nnFormer-main/DATASET/nnFormer_trained_models/nnFormer/3d_fullres/Task264_CMUexternalValT2DCEReg/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1")
            },
            "MAML": {
                "internal": os.path.join(base_dir,
                                         "MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "MAML-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAMLTrainerV2__nnUNetPlansv2.1")
            },
            "A2FSeg": {
                "internal": os.path.join(base_dir,
                                         "A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "A2FSeg-main/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/MAML3_channelTrainerV2BreastRegions__nnUNetPlansv2.1")
            },
            "PA-Net": {
                "internal": os.path.join(base_dir,
                                         "PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task262_T2DCEReg/DUNetTrainer__nnUNetPlansv2.1"),
                "external_ync": os.path.join(base_dir,
                                             "PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task263_externalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1"),
                "external_cmu": os.path.join(base_dir,
                                             "PA-Net-master/data/RESULTS_FOLDER/nnUNet/3d_fullres/Task264_CMUexternalValT2DCEReg/DUNetTrainer__nnUNetPlansv2.1")
            },
            "mmFormer": {
                "internal": os.path.join(base_dir, "mmFormer-main/pred_file/Task262_T2DCEReg"),
                "external_ync": os.path.join(base_dir, "mmFormer-main/pred_file/Task263_externalValT2DCEReg"),
                "external_cmu": os.path.join(base_dir, "mmFormer-main/pred_file/Task264_CMUexternalValT2DCEReg")
            },
            "NestedFormer": {
                "internal": os.path.join(base_dir, "NestedFormer-main/output_dir/Task262_T2DCEReg"),
                "external_ync": os.path.join(base_dir, "NestedFormer-main/output_dir/Task263_externalValT2DCEReg"),
                "external_cmu": os.path.join(base_dir, "NestedFormer-main/output_dir/Task264_CMUexternalValT2DCEReg")
            }
        }

    def load_data(self):
        """
        加载所有模型的预测结果和GT：
        - 普通模型：从fold下的validation_raw加载
        - 特殊模型(mmFormer/NestedFormer)：直接从fold目录加载（无validation_raw）
        - 所有模型的GT：统一使用你的模型的GT文件夹
        """
        for model_name, config in self.model_configs.items():
            self.model_dirs[model_name] = {}
            self.gt_dirs[model_name] = {}

            # 获取你的模型的GT目录配置
            your_model_config = self.model_configs["YourModel"]

            # 处理内部验证数据
            if model_name in self.cross_fold_models:
                # nnUNet和nnFormer：跨所有fold收集case（不绑定fold）
                all_cases = {}  # 临时存储{case_id: (pred_path, gt_path)}
                for fold in range(5):
                    # 根据模型类型确定预测目录
                    if model_name in self.special_dir_models:
                        # 特殊模型：直接使用fold目录
                        pred_dir = os.path.join(config["internal"], f"fold_{fold}")
                    else:
                        # 普通模型：使用fold下的validation_raw
                        pred_dir = os.path.join(config["internal"], f"fold_{fold}/validation_raw")

                    if not os.path.exists(pred_dir):
                        continue

                    # 收集该折中所有case
                    pred_files = glob(os.path.join(pred_dir, "*.nii.gz"))
                    for pred_file in pred_files:
                        case_id = os.path.basename(pred_file).replace(".nii.gz", "")

                        # 统一使用你的模型的GT文件
                        if model_name in self.no_gt_models:
                            # 特殊模型（无GT文件夹）使用你的模型的GT
                            your_gt_dir = os.path.join(your_model_config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")
                        else:
                            # 普通模型使用自己的GT（如果有）
                            gt_dir = os.path.join(config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(gt_dir, f"{case_id}.nii.gz")

                        # 备选GT路径（如果自己的GT不存在）
                        if not os.path.exists(gt_file):
                            your_gt_dir = os.path.join(your_model_config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")

                        if os.path.exists(gt_file) and case_id not in all_cases:
                            all_cases[case_id] = (pred_file, gt_file)

                # 存储为{case_id: pred_path}，不包含fold信息
                for case_id, (pred_path, gt_path) in all_cases.items():
                    key = f"internal_{case_id}"  # 标识格式：internal_病例ID
                    self.model_dirs[model_name][key] = pred_path
                    self.gt_dirs[model_name][key] = gt_path

            else:
                # 你的模型和其他模型：按fold存储（同fold有相同case）
                for fold in range(5):
                    # 根据模型类型确定预测目录
                    if model_name in self.special_dir_models:
                        # 特殊模型：直接使用fold目录（无validation_raw）
                        pred_dir = os.path.join(config["internal"], f"fold_{fold}")
                    else:
                        # 普通模型：使用fold下的validation_raw
                        pred_dir = os.path.join(config["internal"], f"fold_{fold}/validation_raw")

                    if not os.path.exists(pred_dir):
                        continue

                    # 收集该折中所有case
                    pred_files = glob(os.path.join(pred_dir, "*.nii.gz"))
                    for pred_file in pred_files:
                        case_id = os.path.basename(pred_file).replace(".nii.gz", "")

                        # 统一使用你的模型的GT文件
                        if model_name in self.no_gt_models:
                            # 特殊模型（无GT文件夹）使用你的模型的GT
                            your_gt_dir = os.path.join(your_model_config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")
                        else:
                            # 普通模型使用自己的GT（如果有）
                            gt_dir = os.path.join(config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(gt_dir, f"{case_id}.nii.gz")

                        # 备选GT路径（如果自己的GT不存在）
                        if not os.path.exists(gt_file):
                            your_gt_dir = os.path.join(your_model_config["internal"], f"fold_{fold}/gt_niftis")
                            gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")

                        if os.path.exists(gt_file):
                            # 标识格式：internal_foldX_病例ID（确保同fold匹配）
                            key = f"internal_fold{fold}_{case_id}"
                            self.model_dirs[model_name][key] = pred_file
                            self.gt_dirs[model_name][key] = gt_file

            # 处理外部验证数据（只使用fold_0）
            for source in ["external_ync", "external_cmu"]:
                source_name = "YNCH" if source == "external_ync" else "CMU"
                fold = 0  # 固定使用fold_0，确保所有模型使用同一折

                # 外部验证数据的fold处理
                # 特殊处理mmFormer和NestedFormer的外部验证路径
                if model_name in self.special_dir_models:
                    # 这两个模型没有validation_raw目录
                    pred_dir = os.path.join(config[source], f"fold_{fold}")
                else:
                    # 其他模型有validation_raw目录
                    pred_dir = os.path.join(config[source], f"fold_{fold}", "validation_raw")

                # 如果路径不存在，尝试其他可能的路径
                if not os.path.exists(pred_dir):
                    # 尝试不包含validation_raw的路径
                    pred_dir = os.path.join(config[source], f"fold_{fold}")
                    if not os.path.exists(pred_dir):
                        # 尝试直接使用源目录
                        pred_dir = config[source]
                        if not os.path.exists(pred_dir):
                            continue

                pred_files = glob(os.path.join(pred_dir, "*.nii.gz"))
                for pred_file in pred_files:
                    case_id = os.path.basename(pred_file).replace(".nii.gz", "")

                    # 统一使用你的模型的GT文件
                    if model_name in self.no_gt_models:
                        # 特殊模型（无GT文件夹）使用你的模型的外部GT
                        your_gt_dir = os.path.join(your_model_config[source], f"fold_{fold}", "gt_niftis")
                        if not os.path.exists(your_gt_dir):
                            your_gt_dir = os.path.join(your_model_config[source], "gt_niftis")
                        gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")
                    else:
                        # 普通模型使用自己的GT（如果有）
                        if model_name == "nnUNet":
                            gt_dir = os.path.join(config[source], f"fold_{fold}", "gt_niftis")
                            if not os.path.exists(gt_dir):
                                gt_dir = os.path.join(config[source], "gt_niftis")
                            gt_file = os.path.join(gt_dir, f"{case_id}.nii.gz")
                        else:
                            gt_dir = os.path.join(self.base_dir, f"GT_files/{source_name}/gt_niftis")
                            gt_file = os.path.join(gt_dir, f"{case_id}.nii.gz")

                    # 备选GT路径（如果自己的GT不存在）
                    if not os.path.exists(gt_file):
                        your_gt_dir = os.path.join(your_model_config[source], f"fold_{fold}", "gt_niftis")
                        if not os.path.exists(your_gt_dir):
                            your_gt_dir = os.path.join(your_model_config[source], "gt_niftis")
                        gt_file = os.path.join(your_gt_dir, f"{case_id}.nii.gz")

                    # 备选GT路径2 - 保留通用GT文件夹，但仅限外部验证数据
                    if not os.path.exists(gt_file):
                        gt_dir = os.path.join(self.base_dir, "GT_files/general/gt_niftis")
                        gt_file = os.path.join(gt_dir, f"{case_id}.nii.gz")

                    if os.path.exists(gt_file):
                        key = f"external_{source_name}_{case_id}"
                        self.model_dirs[model_name][key] = pred_file
                        self.gt_dirs[model_name][key] = gt_file

    def find_common_cases(self):
        """查找所有模型共有的case"""
        # 1. 单独筛选内部验证的common cases
        internal_common = []
        your_internal_cases = [k for k in self.model_dirs["YourModel"].keys() if k.startswith("internal_fold")]

        for your_case_key in your_internal_cases:
            # 解析你的模型case信息：internal_foldX_caseID → (foldX, caseID)
            fold_part, case_id = your_case_key.split("_", 2)[1], your_case_key.split("_", 2)[2]

            # 检查其他非跨fold模型是否在同fold有该case
            other_model_ok = True
            for model_name in self.model_configs.keys():
                if model_name == "YourModel" or model_name in self.cross_fold_models:
                    continue  # 跳过自己和跨fold模型

                # 检查特殊模型是否有该病例
                if model_name in self.special_dir_models:
                    model_case_key = f"internal_{fold_part}_{case_id}"
                    if model_case_key not in self.model_dirs[model_name]:
                        other_model_ok = False
                        break

                # 非跨fold模型的case标识应为：internal_foldX_caseID
                model_case_key = f"internal_{fold_part}_{case_id}"
                if model_case_key not in self.model_dirs[model_name]:
                    other_model_ok = False
                    break

            if not other_model_ok:
                continue

            # 检查跨fold模型（nnUNet/nnFormer）是否有该case（任意fold）
            cross_model_ok = True
            for model_name in self.cross_fold_models:
                # 跨fold模型的case标识为：internal_caseID
                model_case_key = f"internal_{case_id}"
                if model_case_key not in self.model_dirs[model_name]:
                    cross_model_ok = False
                    break

            if cross_model_ok:
                internal_common.append(your_case_key)  # 以你的模型case标识为基准

        # 2. 单独筛选外部验证的common cases
        external_common = []
        your_external_cases = [k for k in self.model_dirs["YourModel"].keys() if k.startswith("external_")]

        for ext_case_key in your_external_cases:
            # 检查所有模型是否有该外部case
            all_have = True
            for model_name in self.model_configs.keys():
                if ext_case_key not in self.model_dirs[model_name]:
                    all_have = False
                    break
            if all_have:
                external_common.append(ext_case_key)

        # 合并内部和外部的common cases（完全分离）
        return internal_common + external_common

    def get_matching_case_key(self, model_name, your_case_key):
        """根据模型类型返回匹配的case标识"""
        if model_name in self.cross_fold_models:
            # 提取case_id（保留内部/外部标识）
            if your_case_key.startswith("internal_fold"):
                case_id = your_case_key.split("_", 2)[2]
                return f"internal_{case_id}"
            elif your_case_key.startswith("external_"):
                return your_case_key  # 外部验证case直接返回
        else:
            # 非跨fold模型直接使用你的模型case标识（同fold）
            return your_case_key

    def calculate_dsc(self, pred, gt):
        """计算Dice相似系数"""
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        if np.sum(gt_flat) == 0:
            return 1.0 if np.sum(pred_flat) == 0 else 0.0
        return f1_score(gt_flat, pred_flat, average='binary')

    def find_best_slices(self, common_cases):
        """寻找最佳切片（分析所有病例，不限制数量）"""
        print("正在分析所有数据，寻找最佳切片...")
        results = []
        # 取消数量限制，分析所有病例
        cases_to_analyze = common_cases  # 不再取前100个，而是全部分析

        for your_case_key in tqdm(cases_to_analyze, desc="分析病例"):
            # 加载你的模型的GT
            try:
                gt_path = self.gt_dirs["YourModel"][your_case_key]
                gt_img = nib.load(gt_path)
                gt_data = gt_img.get_fdata()
            except Exception as e:
                print(f"加载你的模型病例{your_case_key}的GT失败: {e}")
                continue

            # 找到有肿瘤的切片范围
            z_indices = np.where(np.sum(gt_data, axis=(0, 1)) > 0)[0]
            if len(z_indices) == 0:
                continue

            # 分析中间50%的切片（肿瘤区域的核心部分）
            start_idx = max(0, int(z_indices[0] + 0.25 * (z_indices[-1] - z_indices[0])))
            end_idx = min(z_indices[-1], int(z_indices[0] + 0.75 * (z_indices[-1] - z_indices[0])))

            for z in range(start_idx, end_idx + 1):
                gt_slice = gt_data[:, :, z]
                if np.sum(gt_slice) == 0:
                    continue

                # 计算所有模型的DSC
                dsc_scores = {}
                for model_name in self.model_configs.keys():
                    try:
                        # 获取该模型匹配的case标识
                        model_case_key = self.get_matching_case_key(model_name, your_case_key)

                        # 检查特殊模型的预测文件是否存在
                        if model_name in self.special_dir_models:
                            if not os.path.exists(self.model_dirs[model_name][model_case_key]):
                                print(f"{model_name} 病例 {model_case_key} 切片 {z} 的预测文件不存在，跳过")
                                dsc_scores[model_name] = 0.0
                                continue

                        # 加载预测结果
                        pred_path = self.model_dirs[model_name][model_case_key]
                        pred_img = nib.load(pred_path)
                        pred_data = pred_img.get_fdata()
                        pred_slice = pred_data[:, :, z]
                        pred_binary = (pred_slice > 0.5).astype(np.uint8)
                        # 计算DSC
                        dsc = self.calculate_dsc(pred_binary, gt_slice)
                        dsc_scores[model_name] = dsc
                    except Exception as e:
                        print(f"处理{model_name}模型的{your_case_key}切片{z}时出错: {e}")
                        dsc_scores[model_name] = 0.0

                # 计算你的模型与其他模型的差异
                your_model_dsc = dsc_scores.get("YourModel", 0.0)
                max_diff = 0.0
                avg_diff = 0.0
                valid_models = 0

                for model_name, dsc in dsc_scores.items():
                    if model_name != "YourModel":
                        valid_models += 1
                        diff = your_model_dsc - dsc
                        avg_diff += diff
                        if diff > max_diff:
                            max_diff = diff

                if valid_models > 0:
                    avg_diff /= valid_models

                results.append({
                    'case_key': your_case_key,
                    'slice': z,
                    'your_model_dsc': your_model_dsc,
                    'max_dsc_diff': max_diff,
                    'avg_dsc_diff': avg_diff,
                    'tumor_area': np.sum(gt_slice),
                    'dsc_scores': dsc_scores
                })

        # 排序并选择最佳切片
        if not results:
            print("未找到符合条件的切片!")
            return

        # 排序逻辑：优先平均DSC优势，然后是肿瘤面积，最后是模型自身DSC
        results.sort(key=lambda x: (x['avg_dsc_diff'], x['tumor_area'], x['your_model_dsc']), reverse=True)

        # 去重（确保不同病例），选择20个
        selected_case_ids = set()
        self.best_slices = []
        for result in results:
            # 提取带来源的唯一标识（修改点：保留内部/外部标识）
            if result['case_key'].startswith("internal_fold"):
                # 格式：internal_病例ID
                unique_id = f"internal_{result['case_key'].split('_', 2)[2]}"
            else:
                # 格式：external_来源_病例ID
                source_part = result['case_key'].split("_", 2)[1]  # 如“YNCH”或“CMU”
                case_id_part = result['case_key'].split("_", 2)[2]
                unique_id = f"external_{source_part}_{case_id_part}"

            if unique_id not in selected_case_ids:
                self.best_slices.append(result)
                selected_case_ids.add(unique_id)
                if len(self.best_slices) >= self.num_cases:  # 现在是20个
                    break

    def visualize_comparison(self):
        """可视化对比"""
        print(f"正在生成{len(self.best_slices)}个最佳病例的可视化结果...")

        for i, result in enumerate(self.best_slices):
            your_case_key = result['case_key']
            slice_z = result['slice']

            # 创建保存目录
            case_dir = os.path.join(self.output_dir, f"case_{i + 1}_{your_case_key}")
            os.makedirs(case_dir, exist_ok=True)

            # 加载原始图像（基于你的模型GT路径）
            try:
                gt_path = self.gt_dirs["YourModel"][your_case_key]
                image_dir = os.path.dirname(os.path.dirname(gt_path))
                if "gt_niftis" in image_dir:
                    image_dir = image_dir.replace("gt_niftis", "imagesTs")

                case_id = os.path.basename(gt_path).replace(".nii.gz", "")
                image_file = None
                patterns = [f"{case_id}_0000.nii.gz", f"{case_id}.nii.gz", f"{case_id[:-4]}_0000.nii.gz"]
                for pattern in patterns:
                    potential_file = os.path.join(image_dir, pattern)
                    if os.path.exists(potential_file):
                        image_file = potential_file
                        break

                # 加载图像数据
                if image_file:
                    image_img = nib.load(image_file)
                    image_data = image_img.get_fdata()
                    if len(image_data.shape) == 4:
                        image_data = image_data[:, :, :, 0]
                else:
                    print(f"警告: 未找到{your_case_key}的原始图像，使用空白背景")
                    image_data = np.zeros(nib.load(gt_path).get_fdata().shape[:3])
            except Exception as e:
                print(f"加载{your_case_key}的原始图像失败: {e}")
                image_data = np.zeros((256, 256, 100))

            # 提取当前切片
            if len(image_data.shape) >= 3 and slice_z < image_data.shape[2]:
                image_slice = image_data[:, :, slice_z]
            else:
                slice_shape = nib.load(self.gt_dirs["YourModel"][your_case_key]).get_fdata().shape[:2]
                image_slice = np.zeros(slice_shape)

            # 调整图像尺寸与GT匹配
            gt_shape = nib.load(self.gt_dirs["YourModel"][your_case_key]).get_fdata().shape[:2]
            if image_slice.shape[:2] != gt_shape:
                from scipy.ndimage import zoom
                zoom_factor = (gt_shape[0] / image_slice.shape[0], gt_shape[1] / image_slice.shape[1])
                image_slice = zoom(image_slice, zoom_factor, order=1)

            # 创建多模型对比图
            num_models = len(self.model_configs)
            # 限制图宽度，避免过宽
            fig_width = min(5 * num_models, 50)  # 最大宽度限制为50英寸
            fig, axes = plt.subplots(1, num_models, figsize=(fig_width, 5))
            fig.suptitle(
                f"病例: {your_case_key}\n切片: {slice_z} | 你的模型DSC: {result['your_model_dsc']:.4f}",
                fontsize=14
            )

            # 逐个模型绘制
            for j, (model_name, ax) in enumerate(zip(self.model_configs.keys(), axes)):
                try:
                    # 获取匹配的case标识
                    model_case_key = self.get_matching_case_key(model_name, your_case_key)

                    # 检查特殊模型的预测文件
                    if model_name in self.special_dir_models:
                        if model_case_key not in self.model_dirs[model_name]:
                            print(f"{model_name} 病例 {model_case_key} 不存在，跳过可视化")
                            ax.set_title(f"{model_name}\n(病例不存在)", fontsize=10)
                            ax.axis('off')
                            continue

                        pred_path = self.model_dirs[model_name][model_case_key]
                        if not os.path.exists(pred_path):
                            print(f"{model_name} 病例 {model_case_key} 预测文件不存在，跳过可视化")
                            ax.set_title(f"{model_name}\n(预测文件缺失)", fontsize=10)
                            ax.axis('off')
                            continue

                    # 加载预测和GT
                    pred_path = self.model_dirs[model_name][model_case_key]
                    pred_img = nib.load(pred_path)
                    pred_data = pred_img.get_fdata()
                    pred_slice = pred_data[:, :, slice_z]
                    pred_binary = (pred_slice > 0.5).astype(np.uint8)

                    gt_path = self.gt_dirs[model_name][model_case_key]
                    gt_img = nib.load(gt_path)
                    gt_data = gt_img.get_fdata()
                    gt_slice = gt_data[:, :, slice_z]

                    # 计算DSC
                    dsc = self.calculate_dsc(pred_binary, gt_slice)

                    # 创建叠加图
                    overlay = self.create_overlay(image_slice, pred_binary, gt_slice)

                    # 显示
                    ax.imshow(overlay)
                    ax.set_title(f"{model_name}\nDSC: {dsc:.4f}", fontsize=10)
                    ax.axis('off')

                    # 保存单个模型图
                    self.save_single_model_fig(case_dir, model_name, your_case_key, slice_z, overlay, dsc)

                except Exception as e:
                    print(f"生成{model_name}模型的可视化时出错: {e}")
                    ax.set_title(f"{model_name}\n(加载失败)", fontsize=10)
                    ax.axis('off')

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.5, label='真实肿瘤'),
                Patch(facecolor='blue', alpha=0.5, label='正确分割'),
                Patch(facecolor='red', alpha=0.5, label='错误分割')
            ]
            fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)

            # 保存对比图
            plt.savefig(
                os.path.join(case_dir, f"all_models_comparison_slice_{slice_z}.png"),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

            # 保存DSC表格
            dsc_df = pd.DataFrame({
                '模型': list(result['dsc_scores'].keys()),
                'DSC值': [f"{v:.4f}" for v in result['dsc_scores'].values()]
            })
            dsc_df.to_csv(os.path.join(case_dir, f"dsc_scores.csv"), index=False, encoding='utf-8-sig')

            print(f"已保存病例 {i + 1}/{len(self.best_slices)} 的可视化结果")

    def save_single_model_fig(self, case_dir, model_name, case_key, slice_z, overlay, dsc):
        """保存单个模型的可视化结果"""
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 8))
        ax_single.imshow(overlay)
        ax_single.set_title(f"{model_name}\n病例: {case_key} | 切片: {slice_z}\nDSC: {dsc:.4f}", fontsize=12)
        ax_single.axis('off')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.5, label='真实肿瘤'),
            Patch(facecolor='blue', alpha=0.5, label='正确分割'),
            Patch(facecolor='red', alpha=0.5, label='错误分割')
        ]
        ax_single.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(
            os.path.join(case_dir, f"{model_name}_slice_{slice_z}.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig_single)

    def create_overlay(self, image, pred, gt):
        """创建彩色叠加图"""
        if np.max(image) > 0:
            image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        else:
            image_norm = image

        overlay = np.stack([image_norm] * 3, axis=-1)

        true_positives = np.logical_and(pred == 1, gt == 1)
        false_positives = np.logical_and(pred == 1, gt == 0)
        false_negatives = np.logical_and(pred == 0, gt == 1)

        overlay[gt == 1, 1] = np.maximum(overlay[gt == 1, 1], 0.8)  # 绿色：真实肿瘤
        overlay[true_positives, 2] = np.maximum(overlay[true_positives, 2], 0.8)  # 蓝色：正确分割
        overlay[false_positives, 0] = np.maximum(overlay[false_positives, 0], 0.8)  # 红色：假阳性
        overlay[false_negatives, 0] = np.maximum(overlay[false_negatives, 0], 0.8)  # 红色：假阴性

        return overlay

    def run(self):
        """运行完整流程"""
        print("开始加载数据...")
        self.load_data()

        print("寻找所有模型共有的病例...")
        common_cases = self.find_common_cases()
        print(f"找到{len(common_cases)}个所有模型都包含的病例")

        # 新增：打印内部和外部病例的数量
        internal_count = sum(1 for case in common_cases if case.startswith("internal_"))
        external_count = sum(1 for case in common_cases if case.startswith("external_"))
        print(f"其中：内部验证病例{internal_count}个，外部验证病例{external_count}个")

        if not common_cases:
            print("没有找到所有模型都共有的病例！请检查数据路径是否正确。")
            return

        self.find_best_slices(common_cases)

        if not self.best_slices:
            print("未找到符合条件的最佳切片！")
            return

        print(f"\n最佳切片选择结果（共{len(self.best_slices)}个）:")
        for i, result in enumerate(self.best_slices):
            print(
                f"{i + 1}. 病例: {result['case_key']}, 切片: {result['slice']}, "
                f"你的模型DSC: {result['your_model_dsc']:.4f}, "
                f"平均DSC优势: {result['avg_dsc_diff']:.4f}"
            )

        self.visualize_comparison()
        print(f"\n可视化完成！结果已保存到: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='医学图像分割模型可视化对比工具（适配特殊目录结构）')
    parser.add_argument('--data_dir', type=str, default='/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2', help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/media/lyq/4dbd4ed9-dd80-4bb0-8276-9178451541d2/MMFAFM_BCS/T2_segmentation_visualization_results',
                        help='输出结果目录')
    parser.add_argument('--num_cases', type=int, default=30, help='要选择的病例数，默认为20')  # 改为20

    args = parser.parse_args()

    visualizer = SegmentationVisualizer(
        base_dir=args.data_dir,
        output_dir=args.output_dir,
        num_cases=args.num_cases
    )

    visualizer.run()


if __name__ == "__main__":
    main()